# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Correctness test for a DECOUPLED / strided GTP rank placement.

Hypothesis under test
---------------------
GTP weight-sharding keys off the *within-group* rank (``gtp_group.rank()``), not
the global rank, so the GTP process group may be carved from arbitrary
(non-contiguous, strided) global ranks without affecting numerics.

Setup
-----
world = 4.  Two phases, both fed the SAME input on every rank (so the DP
replicate dimension is trivially in sync and needs no all-reduce):

  Phase 1 (baseline):  GTP=1, DP=4         -> full weight on every rank.
  Phase 2 (decoupled): GTP=2, DP=2 with gtp_order_anchor="dp"
                       -> GTP axis is injected AFTER dp (outer), so the GTP shard
                          groups are STRIDED: {0,2} and {1,3} (not {0,1}{2,3}).
                          Exercises the integrated parallel_state knob.

If the per-step loss trajectory of Phase 2 matches Phase 1, a strided GTP group
shards + all-gathers + reduce-scatters correctly -> decoupling GTP placement
from a contiguous layout is numerically safe.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.experimental.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

from megatron.experimental.gtp import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


def _worker_gtp_strided_correctness(rank, world_size, port):
    """Baseline (GTP=1, DP=4) vs strided GTP=2 (groups {0,2},{1,3}; DP=2)."""
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    from megatron.core import parallel_state as ps
    from megatron.experimental.gtp.generalized_tensor_parallelism import get_global_GTP_cache
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    HIDDEN = 4096
    NUM_HEADS = 32
    FFN_HIDDEN = 16384
    NUM_LAYERS = 2
    SEQ = 32
    BATCH = 1
    LR = 0.01
    STEPS = 10
    dtype = torch.bfloat16
    recipe = MXFP8BlockScaling()

    def make_config():
        return TransformerConfig(
            num_attention_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            ffn_hidden_size=FFN_HIDDEN,
            add_bias_linear=False,
            params_dtype=dtype,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bias_dropout_fusion=False,
            fp8='e4m3',
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def make_transformer_stack(config, pg_collection):
        spec = get_gpt_layer_with_transformer_engine_spec()
        return torch.nn.ModuleList(
            [
                spec.module(
                    config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection
                )
                for i in range(NUM_LAYERS)
            ]
        )

    def run_step(layers, x):
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            for layer in layers:
                x, _ = layer(x, attention_mask=None)
        return x.mean()

    # ---- Phase 1: Baseline — GTP=1 (DP=4); same input on all ranks ----
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp'])
    config = make_config()
    layers = make_transformer_stack(config, pg_collection)
    for layer in layers:
        layer.cuda()
    for p in layers.parameters():
        dist.broadcast(p.data, src=0)
    saved_weights = {n: p.data.clone() for n, p in layers.named_parameters()}

    baseline_losses = []
    for step in range(STEPS):
        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)
        loss = run_step(layers, x)
        if rank == 0:
            baseline_losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for p in layers.parameters():
                if p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}
    get_global_GTP_cache().clear()  # drop shape-keyed buffers so neighbors don't reuse stale sizes
    FP8GlobalStateManager.reset()

    # ---- Phase 2: strided GTP=2 (world = TP1 * GTP2 * CP1 * DP2) ----
    # gtp_order_anchor="dp" injects GTP AFTER dp -> GTP shard groups are strided {0,2},{1,3}.
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_remat_size=2,
        gtp_order_anchor="dp",
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp'])
    config = make_config()
    layers_gtp = make_transformer_stack(config, pg_collection)
    for layer in layers_gtp:
        layer.cuda()

    gtp_group = ps.get_generalized_tensor_parallel_remat_group()
    gtp_size = gtp_group.size()
    gtp_rank = gtp_group.rank()
    gtp_global_ranks = ps.get_generalized_tensor_parallel_remat_global_ranks()
    assert gtp_size == 2, f"GTP shard group size should be 2, got {gtp_size}"
    # The whole point: confirm the group is STRIDED, not contiguous.
    assert sorted(gtp_global_ranks) in ([0, 2], [1, 3]), (
        f"expected a strided GTP group {{0,2}} or {{1,3}}, got {sorted(gtp_global_ranks)}"
    )

    gtp_params = [p for p in layers_gtp.parameters() if isinstance(p, GTPShardedParam)]
    assert len(gtp_params) > 0, "GTP not active: no GTPShardedParam found"

    for name, p in layers_gtp.named_parameters():
        full = saved_weights[name]
        if isinstance(p, GTPShardedParam):
            shard_size = p.shape[0]
            p.data.copy_(full[gtp_rank * shard_size : (gtp_rank + 1) * shard_size])
        else:
            p.data.copy_(full)

    for p in layers_gtp.parameters():
        if isinstance(p, GTPShardedParam):
            p.main_grad = torch.zeros(p.shape, dtype=dtype, device='cuda')

    gtp_losses = []
    for step in range(STEPS):
        for p in layers_gtp.parameters():
            if isinstance(p, GTPShardedParam):
                p.main_grad.zero_()
        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)
        loss = run_step(layers_gtp, x)
        if rank == 0:
            gtp_losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for p in layers_gtp.parameters():
                if isinstance(p, GTPShardedParam):
                    p.data.sub_((LR / gtp_size) * p.main_grad)
                elif p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}
    get_global_GTP_cache().clear()  # drop shape-keyed buffers so neighbors don't reuse stale sizes

    if rank == 0:
        assert len(baseline_losses) == STEPS and len(gtp_losses) == STEPS
        print(f"\nStrided GTP global ranks for this run: {sorted(gtp_global_ranks)}", flush=True)
        for step, (lb, lg) in enumerate(zip(baseline_losses, gtp_losses)):
            print(f"Step {step:2d}: baseline={lb:.6f}  strided_gtp={lg:.6f}", flush=True)
        torch.testing.assert_close(
            torch.tensor(gtp_losses), torch.tensor(baseline_losses), atol=1e-5, rtol=1e-5
        )


class TestGTPStridedPlacement:
    def test_strided_gtp_loss_trajectory_matches_baseline(self):
        """Strided GTP=2 ({0,2},{1,3}) per-step losses must match no-GTP baseline."""
        _requires_mxfp8()
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_gtp_strided_correctness, 4)
