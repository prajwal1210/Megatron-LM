# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the GTP symmetric-memory (NVLS) path.

Test groups
-----------
- TestRegisteredLifoPool    - LIFO recycling, keying, tagging, capture guard (single process)
- TestSymmFlagStamping      - gtp_smr / needs_nccl_mem from is_expert x config flags
- TestConfigureRecipeSymm   - symm kwargs land in GTP_CONFIG + fp32-accum interaction warning
- TestWgradSendBufferSplit  - get_wgrad_tensor / _prepare_wgrad_reduce_scatter_inputs routing:
                              unpadded symm scratch comes from the registered LIFO (zero-copy
                              send), padded symm copies into a registered padded buffer, and
                              the release paths recycle both
- TestSymmBackwardNumerics  - symm-flagged backward produces the same main_grad as plain GTP
- TestRealPoolRegistration  - register_gtp_symm_pool + gtp_symm_pool_ctx + a
                              collective on pool memory; skips where NCCL window
                              registration is unsupported

The buffer-routing tests monkeypatch ``is_gtp_symm_pool_registered`` in BOTH consuming
namespaces
(the gtp module's binding for routing decisions AND gtp_symmetric_memory's own for the LIFO's
branch), so LIFO allocations genuinely flow through ``gtp_symm_pool_ctx`` into the group's
ncclMemAlloc pool — just without the (collective, environment-dependent) window registration,
which TestRealPoolRegistration covers for real.

Multi-GPU tests skip when ``torch.distributed.get_world_size()`` != 4.
"""

import logging

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
import megatron.core.tensor_parallel.gtp_symmetric_memory as gtp_symm
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTP_CONFIG,
    GTPShardedParam,
)
from megatron.core.tensor_parallel.gtp_symmetric_memory import (
    RegisteredLifoPool,
    deregister_gtp_symm_pools,
    gtp_symm_pool_ctx,
    is_gtp_symm_pool_registered,
    register_gtp_symm_pool,
    symmetric_wgrad_pool,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


class _StubGroup:
    """Minimal stand-in for a dist process group (never used for real comms)."""

    def __init__(self, name="stub_gtp_symm_group", size=2, rank=0):
        self.group_name = name
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


_CONFIG_FIELDS = (
    "gtp_nccl_ub",
    "egtp_nccl_ub",
    "use_distributed_optimizer",
    "reduce_scatter_with_fp32_accumulation",
    "pad_for_alignment",
    "check_param_states",
    "calculate_per_token_loss",
)


@pytest.fixture(autouse=True)
def _restore_gtp_config():
    """Snapshot/restore the GTP_CONFIG fields these tests mutate."""
    saved = {f: getattr(GTP_CONFIG, f) for f in _CONFIG_FIELDS}
    yield
    for f, v in saved.items():
        setattr(GTP_CONFIG, f, v)


# ---------------------------------------------------------------------------
# RegisteredLifoPool (single process)
# ---------------------------------------------------------------------------


class TestRegisteredLifoPool:
    def test_alloc_returns_tagged_view(self):
        pool = RegisteredLifoPool()
        group = _StubGroup()
        buf = pool.alloc((8, 4), torch.bfloat16, "cuda", group)
        assert tuple(buf.shape) == (8, 4)
        assert buf.dtype == torch.bfloat16
        assert getattr(buf, "_gtp_symm_group", None) is group

    def test_free_then_alloc_recycles_lifo(self):
        pool = RegisteredLifoPool()
        group = _StubGroup()
        a = pool.alloc((8, 4), torch.bfloat16, "cuda", group)
        b = pool.alloc((8, 4), torch.bfloat16, "cuda", group)
        a_ptr, b_ptr = a.data_ptr(), b.data_ptr()
        assert a_ptr != b_ptr
        pool.free(a)
        pool.free(b)
        # LIFO: last freed comes back first; storage identity is preserved.
        assert pool.alloc((8, 4), torch.bfloat16, "cuda", group).data_ptr() == b_ptr
        assert pool.alloc((8, 4), torch.bfloat16, "cuda", group).data_ptr() == a_ptr
        # A 1-D key serves any shape with that numel.
        pool.free(a)
        c = pool.alloc((4, 8), torch.bfloat16, "cuda", group)
        assert c.data_ptr() == a_ptr and tuple(c.shape) == (4, 8)

    def test_keying_isolates_dtype_numel_group(self):
        pool = RegisteredLifoPool()
        g1, g2 = _StubGroup("g1"), _StubGroup("g2")
        a = pool.alloc((8,), torch.bfloat16, "cuda", g1)
        pool.free(a)
        assert pool.alloc((8,), torch.float32, "cuda", g1).data_ptr() != a.data_ptr()
        assert pool.alloc((16,), torch.bfloat16, "cuda", g1).data_ptr() != a.data_ptr()
        assert pool.alloc((8,), torch.bfloat16, "cuda", g2).data_ptr() != a.data_ptr()
        assert pool.alloc((8,), torch.bfloat16, "cuda", g1).data_ptr() == a.data_ptr()

    def test_free_untagged_is_noop(self):
        pool = RegisteredLifoPool()
        pool.free(torch.empty(8, device="cuda"))
        assert not pool._free  # nothing entered the free lists

    def test_capture_guard_raises_on_empty_bucket(self, monkeypatch):
        pool = RegisteredLifoPool()
        group = _StubGroup()
        warm = pool.alloc((8,), torch.bfloat16, "cuda", group)
        pool.free(warm)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        # Pop from the warmed bucket is capture-safe...
        got = pool.alloc((8,), torch.bfloat16, "cuda", group)
        assert got.data_ptr() == warm.data_ptr()
        # ...but a fresh allocation during capture must fail loudly.
        with pytest.raises(RuntimeError, match="during CUDA-graph capture"):
            pool.alloc((8,), torch.bfloat16, "cuda", group)


# ---------------------------------------------------------------------------
# gtp_smr / needs_nccl_mem stamping (single process)
# ---------------------------------------------------------------------------


class TestSymmFlagStamping:
    @pytest.mark.parametrize(
        "gtp_flag,egtp_flag,is_expert,expect",
        [
            (True, False, False, True),  # dense param, dense flag on
            (True, False, True, False),  # expert param must NOT take the dense flag
            (False, True, True, True),  # expert param, expert flag on
            (False, True, False, False),  # dense param must NOT take the expert flag
            (False, False, False, False),
        ],
    )
    def test_stamping_follows_is_expert(self, gtp_flag, egtp_flag, is_expert, expect):
        GTP_CONFIG.gtp_nccl_ub = gtp_flag
        GTP_CONFIG.egtp_nccl_ub = egtp_flag
        p = GTPShardedParam(torch.randn(4, 4, device="cuda", dtype=torch.bfloat16))
        gtp_module._gtp_attach_attrs(p, _StubGroup(), is_expert=is_expert)
        assert p.gtp_smr is expect
        assert p.needs_nccl_mem is expect


# ---------------------------------------------------------------------------
# configure_gtp_remat_from_recipe: symm kwargs + fp32-accum warning
# ---------------------------------------------------------------------------


class TestConfigureRecipeSymm:
    def test_symm_kwargs_land_in_config_and_register(self, monkeypatch, caplog):
        registered = []
        monkeypatch.setattr(
            gtp_module, "register_gtp_symm_pool", lambda group: registered.append(group)
        )
        with caplog.at_level(logging.WARNING, logger=gtp_module.logger.name):
            gtp_module.configure_gtp_remat_from_recipe(
                gtp_nccl_ub=True,
                egtp_nccl_ub=False,
                use_distributed_optimizer=True,
                reduce_scatter_with_fp32_accumulation=True,
            )
        assert GTP_CONFIG.gtp_nccl_ub is True
        assert GTP_CONFIG.egtp_nccl_ub is False
        assert GTP_CONFIG.use_distributed_optimizer is True
        assert len(registered) == 1  # dense group only
        if not dist.is_initialized() or dist.get_rank() == 0:
            assert any("all-gather side only" in r.message for r in caplog.records)

    def test_no_warning_without_fp32_accum(self, monkeypatch, caplog):
        monkeypatch.setattr(gtp_module, "register_gtp_symm_pool", lambda group: None)
        with caplog.at_level(logging.WARNING, logger=gtp_module.logger.name):
            gtp_module.configure_gtp_remat_from_recipe(gtp_nccl_ub=True, egtp_nccl_ub=True)
        assert not any("all-gather side only" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_wgrad_tensor / _prepare_wgrad_reduce_scatter_inputs routing (world 4)
# ---------------------------------------------------------------------------


def _worker_wgrad_split(rank, world_size, port):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    group = dist.new_group(list(range(world_size)))
    # pad_for_alignment=16, world 4 -> alignment 64: out=128 -> pad 0; out=100 -> pad 28.
    aligned = _make_gtp_linear(64, 128, group, dtype)
    padded = _make_gtp_linear(64, 100, group, dtype)
    wa, wp = aligned.weight, padded.weight
    assert wa.pad_length == 0 and wp.pad_length > 0
    for w in (wa, wp):
        w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")

    saved_pred = gtp_module.is_gtp_symm_pool_registered
    saved_symm_pred = gtp_symm.is_gtp_symm_pool_registered
    # Patch BOTH consuming namespaces: gtp_module's binding drives the routing decisions;
    # gtp_symm's own drives RegisteredLifoPool.alloc's pool-vs-plain branch, so LIFO
    # allocations genuinely land in the group's ncclMemAlloc pool.
    gtp_module.is_gtp_symm_pool_registered = lambda g: g is group
    gtp_symm.is_gtp_symm_pool_registered = lambda g: g is group
    try:
        # Not symm-stamped -> plain scratch even with the pool "registered".
        t = wa.get_wgrad_tensor()
        assert getattr(t, "_gtp_symm_group", None) is None
        wa.gtp_smr = True
        wp.gtp_smr = True

        # Unpadded symm weight: GEMM scratch IS the registered send buffer (zero-copy).
        assert group.group_name not in gtp_symm._pools
        t = wa.get_wgrad_tensor()
        assert getattr(t, "_gtp_symm_group", None) is group
        assert tuple(t.shape) == tuple(wa._unsharded_shape)
        # Positive proof the allocation routed through gtp_symm_pool_ctx: the group's
        # pool was created by this alloc.
        assert group.group_name in gtp_symm._pools
        prepared = wa._prepare_wgrad_reduce_scatter_inputs([t])
        assert prepared[0] is t

        # Release recycles it through the registered LIFO (same storage comes back).
        wa._wgrad_input_bufs = [t]
        ptr = t.data_ptr()
        wa._release_comm_scratch()
        t2 = symmetric_wgrad_pool.alloc(tuple(wa._unsharded_shape), dtype, t.device, group)
        assert t2.data_ptr() == ptr
        symmetric_wgrad_pool.free(t2)

        # Padded symm weight: plain scratch; _prepare copies into a registered padded
        # buffer with a zeroed tail and tracks it for release.
        g = wp.get_wgrad_tensor()
        assert getattr(g, "_gtp_symm_group", None) is None
        g.normal_()
        prepared = wp._prepare_wgrad_reduce_scatter_inputs([g])
        buf = prepared[0]
        assert buf is not g
        assert getattr(buf, "_gtp_symm_group", None) is group
        assert tuple(buf.shape) == tuple(wp._unsharded_shape_padded)
        n = wp._unsharded_shape[0]
        assert torch.equal(buf[:n], g)
        assert torch.count_nonzero(buf[n:]) == 0
        assert wp._wgrad_sendbufs == [buf]
        wp._release_wgrad_sendbufs()
        assert wp._wgrad_sendbufs is None

        # Foreign (untagged) unpadded wgrad: alias miss falls to the copy branch.
        f = torch.randn(tuple(wa._unsharded_shape), dtype=dtype, device="cuda")
        prepared = wa._prepare_wgrad_reduce_scatter_inputs([f])
        assert prepared[0] is not f
        assert getattr(prepared[0], "_gtp_symm_group", None) is group
        assert torch.equal(prepared[0], f)
        wa._release_wgrad_sendbufs()

        # fp32-accum turns the RS into an all-to-all (group > 2): symm send is skipped.
        GTP_CONFIG.reduce_scatter_with_fp32_accumulation = True
        t = wa.get_wgrad_tensor()
        assert getattr(t, "_gtp_symm_group", None) is None
    finally:
        gtp_module.is_gtp_symm_pool_registered = saved_pred
        gtp_symm.is_gtp_symm_pool_registered = saved_symm_pred
        GTP_CONFIG.reduce_scatter_with_fp32_accumulation = False
        # Drop the LIFO buffers and the (unregistered) pools created via gtp_symm_pool_ctx.
        deregister_gtp_symm_pools()


class TestWgradSendBufferSplit:
    def test_send_buffer_routing(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_wgrad_split, 4)


# ---------------------------------------------------------------------------
# Backward numerics: symm buffers must not change the reduced gradient (world 4)
# ---------------------------------------------------------------------------


def _worker_symm_backward_numerics(rank, world_size, port):
    dtype = torch.bfloat16
    group = dist.new_group(list(range(world_size)))
    inp = torch.randn(16, 64, dtype=dtype, device="cuda")
    gout = torch.randn(16, 128, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)
    dist.broadcast(gout, src=0)

    saved_pred = gtp_module.is_gtp_symm_pool_registered
    saved_symm_pred = gtp_symm.is_gtp_symm_pool_registered
    grads = {}
    try:
        for symm in (False, True):
            # Fresh chain state per phase so both layers are standalone chain heads
            # (sync RS) instead of linking into one chain with a dangling async RS.
            gtp_module.reset_gtp_state()
            torch.manual_seed(0)
            layer = _make_gtp_linear(64, 128, group, dtype)
            w = layer.weight
            w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")
            pred = (lambda g: g is group) if symm else saved_pred
            gtp_module.is_gtp_symm_pool_registered = pred
            gtp_symm.is_gtp_symm_pool_registered = pred if symm else saved_symm_pred
            w.gtp_smr = symm
            x = inp.clone().requires_grad_(True)
            out = layer(x, is_first_microbatch=True)
            out.backward(gout)
            grads[symm] = w.main_grad.clone()
    finally:
        gtp_module.is_gtp_symm_pool_registered = saved_pred
        gtp_symm.is_gtp_symm_pool_registered = saved_symm_pred
        deregister_gtp_symm_pools()

    # Same inputs, same weights, same collective — the send buffer's provenance must
    # not change the reduced gradient.
    torch.testing.assert_close(grads[True], grads[False], atol=0.0, rtol=0.0)


class TestSymmBackwardNumerics:
    def test_main_grad_matches_plain_gtp(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_symm_backward_numerics, 4)


# ---------------------------------------------------------------------------
# Real pool registration + collective on pool memory (world 4)
# ---------------------------------------------------------------------------


def _worker_real_pool_registration(rank, world_size, port):
    group = dist.new_group(list(range(world_size)))
    try:
        register_gtp_symm_pool(group)
    except Exception as e:  # NCCL window registration unsupported in this environment
        pytest.skip(f"NCCL symmetric registration unavailable: {e}")
    try:
        assert is_gtp_symm_pool_registered(group)
        # Idempotent re-register must not raise or re-issue the warmup.
        register_gtp_symm_pool(group)
        with gtp_symm_pool_ctx(group):
            src = torch.full((4,), float(rank), device="cuda")
            out = torch.empty(4 * world_size, device="cuda")
        dist.all_gather_into_tensor(out, src, group=group)
        expected = torch.repeat_interleave(
            torch.arange(world_size, device="cuda", dtype=torch.float32), 4
        )
        assert torch.equal(out, expected)
    finally:
        # Mandatory: leftover windows abort the ProcessGroupNCCL destructor at teardown.
        deregister_gtp_symm_pools()
    assert not is_gtp_symm_pool_registered(group)


class TestRealPoolRegistration:
    def test_register_alloc_collective_deregister(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_real_pool_registration, 4)
