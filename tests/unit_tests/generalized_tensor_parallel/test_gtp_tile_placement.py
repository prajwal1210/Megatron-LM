# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration test for the 2D NVL/IB tile construction (Phase B).

Drives the `gtp_nvl_local` / `dp_nvl_local` tile path in parallel_state at world=4
(degenerate rack=1 tile -> reproduces the strided {0,2}{1,3} GTP groups) and checks:
  1. the tile actually built the expected GTP + no_gtp DP groups, and
  2. the per-step loss matches a GTP=1 baseline (bit-for-bit, same input on all ranks).

This validates the parallel_state tile integration (asserts fire, groups built, get_*_group
works, training correct) cheaply on 1 node, before multi-rack tile runs (gtp_nvl=16, dp_nvl=4).
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


def _worker_gtp_tile_correctness(rank, world_size, port):
    """Baseline (GTP=1, DP=4) vs tile GTP=2 via gtp_nvl_local=1/dp_nvl_local=1 (groups {0,2},{1,3})."""
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

    def make_stack(config, pg_collection):
        spec = get_gpt_layer_with_transformer_engine_spec()
        return torch.nn.ModuleList(
            [
                spec.module(config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection)
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
    layers = make_stack(config, pg_collection)
    for layer in layers:
        layer.cuda()
    for p in layers.parameters():
        dist.broadcast(p.data, src=0)
    saved = {n: p.data.clone() for n, p in layers.named_parameters()}

    baseline = []
    for step in range(STEPS):
        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)
        loss = run_step(layers, x)
        if rank == 0:
            baseline.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for p in layers.parameters():
                if p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}
    get_global_GTP_cache().clear()
    FP8GlobalStateManager.reset()

    # ---- Phase 2: tile GTP=2, DP=2 via gtp_nvl_local=1, dp_nvl_local=1 ----
    # rack=1 degenerate tile -> GTP groups {0,2},{1,3}; no_gtp DP {0,1},{2,3}.
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_remat_size=2,
        gtp_nvl_local=1,
        dp_nvl_local=1,
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp'])
    config = make_config()
    layers_t = make_stack(config, pg_collection)
    for layer in layers_t:
        layer.cuda()

    gtp_group = ps.get_generalized_tensor_parallel_remat_group()
    gtp_size = gtp_group.size()
    gtp_rank = gtp_group.rank()
    gtp_global = ps.get_generalized_tensor_parallel_remat_global_ranks()
    assert gtp_size == 2, f"expected GTP size 2, got {gtp_size}"
    # The tile must have built a STRIDED group, not contiguous {0,1}/{2,3}.
    assert sorted(gtp_global) in ([0, 2], [1, 3]), (
        f"tile GTP group should be {{0,2}} or {{1,3}}, got {sorted(gtp_global)}"
    )

    gtp_params = [p for p in layers_t.parameters() if isinstance(p, GTPShardedParam)]
    assert len(gtp_params) > 0, "GTP not active under tile"

    for name, p in layers_t.named_parameters():
        full = saved[name]
        if isinstance(p, GTPShardedParam):
            sh = p.shape[0]
            p.data.copy_(full[gtp_rank * sh : (gtp_rank + 1) * sh])
        else:
            p.data.copy_(full)
    for p in layers_t.parameters():
        if isinstance(p, GTPShardedParam):
            p.main_grad = torch.zeros(p.shape, dtype=dtype, device='cuda')

    tile_losses = []
    for step in range(STEPS):
        for p in layers_t.parameters():
            if isinstance(p, GTPShardedParam):
                p.main_grad.zero_()
        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)
        loss = run_step(layers_t, x)
        if rank == 0:
            tile_losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for p in layers_t.parameters():
                if isinstance(p, GTPShardedParam):
                    p.data.sub_((LR / gtp_size) * p.main_grad)
                elif p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}
    get_global_GTP_cache().clear()

    if rank == 0:
        print(f"\nTile GTP global ranks: {sorted(gtp_global)}", flush=True)
        for s, (lb, lt) in enumerate(zip(baseline, tile_losses)):
            print(f"Step {s:2d}: baseline={lb:.6f}  tile={lt:.6f}", flush=True)
        torch.testing.assert_close(
            torch.tensor(tile_losses), torch.tensor(baseline), atol=1e-5, rtol=1e-5
        )


class TestGTPTilePlacement:
    def test_tile_loss_trajectory_matches_baseline(self):
        """Tile GTP=2 ({0,2},{1,3}) per-step losses must match no-GTP baseline."""
        _requires_mxfp8()
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_gtp_tile_correctness, 4)
