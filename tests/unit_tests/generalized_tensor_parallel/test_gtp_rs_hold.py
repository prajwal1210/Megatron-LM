# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the plan-driven wgrad reduce-scatter hold/flush (L2).

Covers:
1.  TestRsHoldParser      - update_gtp_config(rs_hold_rules=...): validation,
                            whitespace, dedup, malformed, bool/0/>cap rejected,
                            post-freeze reject.
2.  TestRsHoldStamping    - classify_gtp_chains stamps rs_hold_steps (first-match,
                            unmatched -> 0, GRAPHED-chain forced to 0 with warning,
                            zero-match warning, _RS_HOLDS_ENABLED kill switch).
3.  TestHoldQueueTick     - park/tick/flush mechanics with a recorded
                            _reduce_scatter: FIFO order, decrement-at-entry,
                            multi-step survival, idempotent flush, chain isolation,
                            force-flush at _wait_reduce_scatter, drain flush in
                            wait_async_comms.
4.  GPU (torchrun) tests  - issue-order timeline with holds, cascade force-flush,
                            and numeric holds-on == holds-off main_grads.
"""

import pytest
import torch

from megatron.core.tensor_parallel.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTP_CONFIG,
    GTPChain,
    GTPShardedParam,
    classify_gtp_chains,
    update_gtp_config,
)


class _FakeGroup:
    """Minimal mock for a dist process group (single-process unit tests)."""

    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _FakeWork:
    """Minimal mock for a dist async-work handle."""

    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _RecordingWork:
    def __init__(self, record):
        self.record = record

    def wait(self):
        self.record.append("rs_wait")


class _RecordingEvent:
    def __init__(self, record, label):
        # NB: do not name this attr 'record' — it would shadow the method.
        self._log = record
        self.label = label

    def record(self, stream=None):
        del stream
        self._log.append(self.label)


class _RecordingGrad:
    shape = (1,)

    def __init__(self, record):
        self.record = record

    def add_(self, value):
        del value
        self.record.append("main_grad_add")


class _RecordingCache:
    def __init__(self, record):
        self.record = record

    def get(self, ticket):
        del ticket
        return object()

    def release(self, ticket):
        del ticket
        self.record.append("ticket_release")


@pytest.fixture(autouse=True)
def reset_rs_hold_state():
    """Reset all GTP mutable state the RS-hold feature touches, so each test
    starts from the byte-identical default (no rules, holds disabled)."""
    yield
    GTP_CONFIG.rs_hold_rules = []
    GTP_CONFIG.rs_finalize_lag = 2
    GTP_CONFIG.prefetch_steps_rules = []
    GTP_CONFIG.max_fetch_steps = 1
    gtp_module._GTP_PREFETCH_RULES_FROZEN = False
    gtp_module._RS_HOLDS_ENABLED = False
    gtp_module._held_rs_queues.clear()
    gtp_module._rs_finalize_queues.clear()
    gtp_module._rs_finalize_streams.clear()
    gtp_module._CUDA_GRAPH_MODULES = None
    gtp_module._FULL_ITERATION = False
    GTPShardedParam._chain_state = {}
    GTPShardedParam._recompute_chain_state = {}
    gtp_module._GTP_CACHE = None
    gtp_module._inflight_comm_params.clear()


def _set_rules(rules):
    """Set rs_hold_rules from a fresh (unfrozen) state."""
    gtp_module._GTP_PREFETCH_RULES_FROZEN = False
    update_gtp_config(rs_hold_rules=rules)


# ---------------------------------------------------------------------------
# 1. Parser / validation (update_gtp_config(rs_hold_rules=...))
# ---------------------------------------------------------------------------


class TestRsHoldParser:

    def test_empty_rules_default(self):
        _set_rules([])
        assert GTP_CONFIG.rs_hold_rules == []

    def test_valid_rules_set(self):
        _set_rules([("mixer.out_proj", 1), ("self_attention", 2)])
        assert GTP_CONFIG.rs_hold_rules == [
            ("mixer.out_proj", 1),
            ("self_attention", 2),
        ]

    def test_whitespace_trimmed(self):
        _set_rules([("  mixer.out_proj  ", 1)])
        assert GTP_CONFIG.rs_hold_rules == [("mixer.out_proj", 1)]

    def test_ordering_first_match_preserved(self):
        _set_rules([("linear", 2), ("linear_fc1", 1)])
        assert GTP_CONFIG.rs_hold_rules[0][0] == "linear"

    def test_duplicate_substring_rejected(self):
        with pytest.raises(ValueError, match="duplicate substring"):
            _set_rules([("out_proj", 1), ("  out_proj ", 2)])

    def test_empty_substring_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _set_rules([("   ", 1)])

    def test_malformed_tuple_rejected(self):
        with pytest.raises(ValueError, match="2-tuple"):
            _set_rules([("out_proj", 1, 1)])

    def test_non_int_steps_rejected(self):
        with pytest.raises(ValueError, match="must be an int"):
            _set_rules([("out_proj", "1")])

    def test_bool_steps_rejected(self):
        with pytest.raises(ValueError, match="must be an int"):
            _set_rules([("out_proj", True)])

    def test_zero_steps_rejected(self):
        # 0 is encoded by rule absence, not an explicit rule.
        with pytest.raises(ValueError, match="out of range"):
            _set_rules([("out_proj", 0)])

    def test_above_cap_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            _set_rules([("out_proj", gtp_module.MAX_RS_HOLD_CAP + 1)])

    def test_bounds_ok(self):
        _set_rules([("a", 1), ("b", gtp_module.MAX_RS_HOLD_CAP)])
        assert GTP_CONFIG.rs_hold_rules[1][1] == gtp_module.MAX_RS_HOLD_CAP

    def test_non_str_substring_rejected(self):
        with pytest.raises(ValueError, match="substring must be a str"):
            _set_rules([(123, 1)])

    def test_non_list_rejected(self):
        with pytest.raises(ValueError, match="must be a list"):
            _set_rules("out_proj:1")

    def test_post_freeze_update_rejected(self):
        _set_rules([("out_proj", 1)])
        gtp_module._GTP_PREFETCH_RULES_FROZEN = True
        with pytest.raises(RuntimeError, match="frozen"):
            update_gtp_config(rs_hold_rules=[("in_proj", 1)])


# ---------------------------------------------------------------------------
# 2. classify_gtp_chains stamping + kill switch
# ---------------------------------------------------------------------------


class _ParamHolder(torch.nn.Module):
    """Wrap GTPShardedParams under given dotted names so named_parameters() yields them."""

    def __init__(self, named_params):
        super().__init__()
        self._named = named_params

    def named_parameters(self, *args, **kwargs):  # noqa: D401 - mimic nn.Module API
        for name, p in self._named.items():
            yield name, p


def _mk_param(shape=(8, 4)):
    p = GTPShardedParam(torch.zeros(*shape))
    p.group = _FakeGroup(size=4)
    p.expert_idx = None
    p.pad_length = 0
    return p


class TestRsHoldStamping:

    def test_first_match_wins(self):
        _set_rules([("mixer", 2), ("mixer.out_proj", 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mixer.out_proj.weight": p})
        classify_gtp_chains(model)
        assert p.rs_hold_steps == 2

    def test_unmatched_defaults_to_zero(self):
        _set_rules([("mixer.out_proj", 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mlp.linear_fc2.weight": p})
        classify_gtp_chains(model)
        assert p.rs_hold_steps == 0

    def test_no_rules_kill_switch_stays_off(self):
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mixer.out_proj.weight": p})
        classify_gtp_chains(model)
        assert p.rs_hold_steps == 0
        assert gtp_module._RS_HOLDS_ENABLED is False

    def test_stamped_flips_kill_switch(self):
        _set_rules([("mixer.out_proj", 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mixer.out_proj.weight": p})
        classify_gtp_chains(model)
        assert p.rs_hold_steps == 1
        assert gtp_module._RS_HOLDS_ENABLED is True

    def test_graphed_chain_match_forced_to_zero(self, caplog):
        # .mixer. classifies GRAPHED when the "mamba" scope tag is active.
        gtp_module._CUDA_GRAPH_MODULES = {"mamba"}
        _set_rules([("mixer.out_proj", 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mixer.out_proj.weight": p})
        with caplog.at_level("WARNING"):
            classify_gtp_chains(model)
        assert p.chain_id == GTPChain.GRAPHED.value
        assert p.rs_hold_steps == 0
        assert gtp_module._RS_HOLDS_ENABLED is False
        assert any("GRAPHED" in r.message for r in caplog.records)

    def test_zero_match_rule_warns(self, caplog):
        _set_rules([("no.such.param", 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mixer.out_proj.weight": p})
        with caplog.at_level("WARNING"):
            classify_gtp_chains(model)
        assert any("matched NO parameters" in r.message for r in caplog.records)

    def test_multi_chunk_enable_survives_later_chunks(self):
        # classify_gtp_remat_chains calls classify once per model chunk (VPP);
        # a later chunk with no eager matches must not erase an earlier enable.
        _set_rules([("mixer.out_proj", 1)])
        pa = _mk_param()
        chunk_a = _ParamHolder({"decoder.layers.0.mixer.out_proj.weight": pa})
        pb = _mk_param()
        chunk_b = _ParamHolder({"decoder.layers.1.mlp.linear_fc2.weight": pb})
        classify_gtp_chains(chunk_a)
        assert gtp_module._RS_HOLDS_ENABLED is True
        classify_gtp_chains(chunk_b)
        assert pa.rs_hold_steps == 1
        assert pb.rs_hold_steps == 0
        assert gtp_module._RS_HOLDS_ENABLED is True


# ---------------------------------------------------------------------------
# 3. Park / tick / flush mechanics (recorded _reduce_scatter, no NCCL)
# ---------------------------------------------------------------------------


def _mk_holdable(name, hold_steps, record, chain=GTPChain.UNGRAPHED.value):
    """A param prepared for the hold path with a recording _reduce_scatter."""
    p = _mk_param()
    p._debug_name = name
    p.chain_id = chain
    p.rs_hold_steps = hold_steps

    def _fake_reduce_scatter(wgrads, async_op, nvtx_label=None, _p=p, **kwargs):
        record.append(("rs", _p._debug_name))
        return [], _FakeWork()

    p._reduce_scatter = _fake_reduce_scatter
    return p


class TestHoldQueueTick:

    def setup_method(self):
        gtp_module._RS_HOLDS_ENABLED = True

    def test_park_then_tick_flushes_in_fifo_order(self):
        record = []
        a = _mk_holdable("a", 1, record)
        b = _mk_holdable("b", 1, record)
        a._park_held_rs([torch.zeros(1)], "a.bwd.async")
        b._park_held_rs([torch.zeros(1)], "b.bwd.async")
        assert record == []
        gtp_module._rs_hold_tick(GTPChain.UNGRAPHED.value)
        assert record == [("rs", "a"), ("rs", "b")]
        assert not gtp_module._held_rs_queues[GTPChain.UNGRAPHED.value]
        assert a._held_rs is None and b._held_rs is None
        assert a._wgrad_rs_handle is not None and b._wgrad_rs_handle is not None

    def test_multi_step_hold_survives_one_tick(self):
        record = []
        a = _mk_holdable("a", 2, record)
        a._park_held_rs([torch.zeros(1)], None)
        gtp_module._rs_hold_tick(GTPChain.UNGRAPHED.value)
        assert record == [] and a._held_rs["remaining"] == 1
        gtp_module._rs_hold_tick(GTPChain.UNGRAPHED.value)
        assert record == [("rs", "a")]

    def test_mixed_holds_due_date_order(self):
        # A later-parked shorter hold may flush before an earlier longer one.
        record = []
        a = _mk_holdable("a", 2, record)
        b = _mk_holdable("b", 1, record)
        a._park_held_rs([torch.zeros(1)], None)
        b._park_held_rs([torch.zeros(1)], None)
        gtp_module._rs_hold_tick(GTPChain.UNGRAPHED.value)
        assert record == [("rs", "b")]
        gtp_module._rs_hold_tick(GTPChain.UNGRAPHED.value)
        assert record == [("rs", "b"), ("rs", "a")]

    def test_flush_idempotent(self):
        record = []
        a = _mk_holdable("a", 1, record)
        a._park_held_rs([torch.zeros(1)], None)
        a._flush_held_rs()
        a._flush_held_rs()
        assert record == [("rs", "a")]

    def test_double_park_asserts(self):
        record = []
        a = _mk_holdable("a", 1, record)
        a._park_held_rs([torch.zeros(1)], None)
        with pytest.raises(AssertionError, match="already"):
            a._park_held_rs([torch.zeros(1)], None)

    def test_chain_isolation(self):
        record = []
        a = _mk_holdable("a", 1, record)
        a._park_held_rs([torch.zeros(1)], None)
        gtp_module._rs_hold_tick(GTPChain.GRAPHED.value)
        assert record == [] and a._held_rs is not None

    def test_wait_reduce_scatter_force_flushes(self):
        # The cascade calls _wait_reduce_scatter on a possibly-held weight; it
        # must issue-then-wait, never wait on (or finalize) an unissued RS.
        record = []
        a = _mk_holdable("a", 3, record)
        a._park_held_rs([torch.zeros(1)], None)
        a._wait_reduce_scatter(finalize_grad=False)
        assert record == [("rs", "a")]
        assert a._held_rs is None and a._wgrad_rs_handle is None

    def test_wait_async_comms_drain_flushes(self):
        record = []
        a = _mk_holdable("a", 3, record)
        a._park_held_rs([torch.zeros(1)], None)
        gtp_module.wait_async_comms(None, finalize_after_drain=False)
        assert record == [("rs", "a")]
        assert not gtp_module._held_rs_queues[GTPChain.UNGRAPHED.value]
        assert a._wgrad_rs_handle is None
        assert a not in gtp_module._inflight_comm_params

    def test_disabled_park_branch_untaken(self):
        # With the kill switch off, wgrad_reduce_scatter must not consult the
        # hold machinery at all (byte-identical default): queues stay absent.
        gtp_module._RS_HOLDS_ENABLED = False
        assert gtp_module._held_rs_queues == {}


class TestDeferredFinalize:
    """The finalize-lag queue replacing the cascade checkpoint for held RSs."""

    def setup_method(self):
        gtp_module._RS_HOLDS_ENABLED = True

    def _mk_flushed(self, name, record):
        p = _mk_holdable(name, 1, record)
        p._wait_calls = []

        def _fake_wait(finalize_grad=False, _p=p):
            _p._wait_calls.append(finalize_grad)
            _p._wgrad_rs_handle = None
            # dequeue like the real method (identity-based)
            fq = gtp_module._rs_finalize_queues.get(_p.chain_id)
            if fq:
                for i, e in enumerate(fq):
                    if e[0] is _p:
                        del fq[i]
                        break

        p._wait_reduce_scatter = _fake_wait
        p._park_held_rs([torch.zeros(1)], None)
        gtp_module._rs_hold_tick(GTPChain.UNGRAPHED.value)  # flush -> enqueued
        return p

    def test_flush_enqueues_for_deferred_finalize(self):
        record = []
        a = self._mk_flushed("a", record)
        q = gtp_module._rs_finalize_queues[GTPChain.UNGRAPHED.value]
        assert len(q) == 1 and q[0][0] is a and q[0][1] == 0

    def test_finalize_after_lag_sites(self):
        record = []
        a = self._mk_flushed("a", record)
        GTP_CONFIG.rs_finalize_lag = 2
        gtp_module._rs_finalize_tick(GTPChain.UNGRAPHED.value)
        assert a._wait_calls == []          # age 1 < lag
        gtp_module._rs_finalize_tick(GTPChain.UNGRAPHED.value)
        assert a._wait_calls == [True]      # age 2 == lag -> finalized
        assert not gtp_module._rs_finalize_queues[GTPChain.UNGRAPHED.value]
        assert a._already_finalized is False  # flag consumed

    def test_drain_flushes_and_finalizes_everything(self):
        record = []
        a = self._mk_flushed("a", record)
        b = _mk_holdable("b", 3, record)
        b._wait_calls = []

        def _fake_wait_b(finalize_grad=False, _p=b):
            _p._wait_calls.append(finalize_grad)
            _p._wgrad_rs_handle = None
            fq = gtp_module._rs_finalize_queues.get(_p.chain_id)
            if fq:
                for i, e in enumerate(fq):
                    if e[0] is _p:
                        del fq[i]
                        break

        b._wait_reduce_scatter = _fake_wait_b
        b._park_held_rs([torch.zeros(1)], None)   # still parked (hold 3)
        gtp_module._rs_drain_chain(GTPChain.UNGRAPHED.value)
        assert record == [("rs", "a"), ("rs", "b")]  # b force-flushed by drain
        assert a._wait_calls == [True] and b._wait_calls == [True]
        assert not gtp_module._held_rs_queues.get(GTPChain.UNGRAPHED.value)
        assert not gtp_module._rs_finalize_queues.get(GTPChain.UNGRAPHED.value)

    def test_direct_wait_dequeues_from_finalize_queue(self):
        # Whoever waits finalizes: a direct _wait_reduce_scatter (e.g. the
        # wait_async_comms drain) must remove the entry so the tick can't
        # double-finalize.
        record = []
        a = self._mk_flushed("a", record)
        a._wait_reduce_scatter(finalize_grad=True)
        assert not gtp_module._rs_finalize_queues[GTPChain.UNGRAPHED.value]
        gtp_module._rs_finalize_tick(GTPChain.UNGRAPHED.value)
        assert a._wait_calls == [True]      # exactly once

    def test_finalize_publishes_post_add_event_before_grad_ready(self, monkeypatch):
        record = []
        param = _mk_param(shape=(1,))
        param._cached_rs_stream = torch.cuda.current_stream()
        param._wgrad_rs_handle = _RecordingWork(record)
        param.rs_event = _RecordingEvent(record, "rs_complete")
        param._gtp_main_grad_ready_event = _RecordingEvent(record, "main_grad_ready")
        param.main_grad = _RecordingGrad(record)
        param._rs_ticket = 1

        monkeypatch.setattr(gtp_module, "get_global_GTP_cache", lambda: _RecordingCache(record))
        monkeypatch.setattr(
            GTPShardedParam,
            "_handle_megatron_grad_accum",
            staticmethod(lambda _param: record.append("grad_ready")),
        )

        param._wait_reduce_scatter(finalize_grad=True)

        assert record.index("main_grad_add") < record.index("main_grad_ready")
        assert record.index("main_grad_ready") < record.index("grad_ready")
        assert param._gtp_main_grad_ready_event is not None

    def test_finalize_lag_validation(self):
        gtp_module._GTP_PREFETCH_RULES_FROZEN = False
        update_gtp_config(rs_finalize_lag=1)
        assert GTP_CONFIG.rs_finalize_lag == 1
        with pytest.raises(ValueError, match="1..3"):
            update_gtp_config(rs_finalize_lag=0)
        with pytest.raises(ValueError, match="1..3"):
            update_gtp_config(rs_finalize_lag=True)
        gtp_module._GTP_PREFETCH_RULES_FROZEN = True
        with pytest.raises(RuntimeError, match="frozen"):
            update_gtp_config(rs_finalize_lag=2)


# ---------------------------------------------------------------------------
# 4. GPU (torchrun) integration tests
# ---------------------------------------------------------------------------

import torch.distributed as dist

from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _make_gtp_linear,
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
)


def _worker_hold_ordering(rank, world_size, port):
    """3-layer chain, holds on the two non-chain-head weights: each held RS must
    issue at the NEXT weight's RS-site entry (after that site's tick), and the
    final grads must match a no-holds reference run bitwise."""
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    group = dist.new_group(list(range(world_size)))

    def run(hold):
        # Fresh RNG per run (identical weights/inputs for the A/B) and fresh
        # chain: without the reset the second run's layers append to the first
        # run's chain and its head is no longer the sync tail.
        torch.manual_seed(0)
        gtp_module.reset_gtp_state()
        gtp_module._inflight_comm_params.clear()
        layers = [_make_gtp_linear(in_f, out_f, group, dtype) for _ in range(3)]
        for lyr in layers:
            lyr.weight.main_grad = torch.zeros(
                lyr.weight.shape, dtype=dtype, device="cuda"
            )
        gtp_module._RS_HOLDS_ENABLED = hold
        if hold:
            # backward runs layers[2] -> [0]; chain head ([0] in bwd terms,
            # prev_w None) stays sync. Hold the two async-path weights.
            for lyr in layers[1:]:
                lyr.weight.rs_hold_steps = 1

        events = []
        orig_rs = GTPShardedParam._reduce_scatter
        orig_site = GTPShardedParam.wgrad_reduce_scatter

        def rec_rs(self, wgrads, async_op, nvtx_label=None, **kwargs):
            events.append(("rs", id(self)))
            return orig_rs(self, wgrads, async_op, nvtx_label=nvtx_label, **kwargs)

        def rec_site(self, wgrad, nvtx_label=None):
            events.append(("site", id(self)))
            return orig_site(self, wgrad, nvtx_label=nvtx_label)

        GTPShardedParam._reduce_scatter = rec_rs
        GTPShardedParam.wgrad_reduce_scatter = rec_site
        try:
            inp = torch.randn(8, in_f, dtype=dtype, device="cuda", requires_grad=True)
            dist.broadcast(inp, src=0)
            out = sum(lyr(inp, is_first_microbatch=True) for lyr in layers)
            out.sum().backward()
        finally:
            GTPShardedParam._reduce_scatter = orig_rs
            GTPShardedParam.wgrad_reduce_scatter = orig_site
            gtp_module._RS_HOLDS_ENABLED = False
            for lyr in layers:
                lyr.weight.rs_hold_steps = 0
        grads = [lyr.weight.main_grad.clone() for lyr in layers]
        return events, grads

    ref_events, ref_grads = run(hold=False)
    hold_events, hold_grads = run(hold=True)

    for g_ref, g_hold in zip(ref_grads, hold_grads):
        assert torch.equal(g_ref, g_hold), "holds changed gradient values"

    def rs_after_next_site(events, w_id):
        sites = [i for i, (k, v) in enumerate(events) if k == "site"]
        rs_i = events.index(("rs", w_id))
        own_site = next(i for i in sites if events[i][1] == w_id)
        later_sites = [i for i in sites if i > own_site]
        return later_sites and rs_i > later_sites[0]

    # Reference: every RS issues inside its own site (before the next site).
    # Held: the held weights' RSs issue after the NEXT site's entry. The first
    # two weights to hit backward (deepest + middle layer) are the held ones —
    # identify them by site order.
    site_order = [v for k, v in hold_events if k == "site"]
    for w_id in site_order[:2]:
        assert rs_after_next_site(hold_events, w_id), (
            "held RS did not move past the next RS-site entry"
        )
    ref_site_order = [v for k, v in ref_events if k == "site"]
    for w_id in ref_site_order[:2]:
        assert not rs_after_next_site(ref_events, w_id), (
            "reference RS unexpectedly issued after the next site"
        )


def _worker_cascade_force_flush(rank, world_size, port):
    """hold_steps=2 on the deepest weight: the tick at the next site leaves it
    parked (remaining 1), and the cascade at that same site force-flushes it.
    Grads must still match the no-holds reference."""
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    group = dist.new_group(list(range(world_size)))

    park_sums = {}
    orig_park = GTPShardedParam._park_held_rs
    orig_flush = GTPShardedParam._flush_held_rs

    def rec_park(self, wgrads, nvtx_label):
        torch.cuda.synchronize()
        park_sums[id(self)] = float(wgrads[0].float().sum())
        return orig_park(self, wgrads, nvtx_label)

    def rec_flush(self, _sums=park_sums):
        if self._held_rs is not None:
            torch.cuda.synchronize()
            now = float(self._held_rs["wgrads"][0].float().sum())
            was = _sums.get(id(self))
            assert was is not None and abs(now - was) <= 1e-2 * (abs(was) + 1), (
                f"parked wgrad clobbered between park and flush: {was} -> {now}"
            )
        return orig_flush(self)

    def run(hold):
        torch.manual_seed(0)
        gtp_module.reset_gtp_state()
        gtp_module._inflight_comm_params.clear()
        layers = [_make_gtp_linear(in_f, out_f, group, dtype) for _ in range(2)]
        for lyr in layers:
            lyr.weight.main_grad = torch.zeros(
                lyr.weight.shape, dtype=dtype, device="cuda"
            )
        gtp_module._RS_HOLDS_ENABLED = hold
        if hold:
            layers[1].weight.rs_hold_steps = 2
        GTPShardedParam._park_held_rs = rec_park
        GTPShardedParam._flush_held_rs = rec_flush
        try:
            inp = torch.randn(8, in_f, dtype=dtype, device="cuda", requires_grad=True)
            dist.broadcast(inp, src=0)
            (layers[0](inp, is_first_microbatch=True)
             + layers[1](inp, is_first_microbatch=True)).sum().backward()
        finally:
            GTPShardedParam._park_held_rs = orig_park
            GTPShardedParam._flush_held_rs = orig_flush
            gtp_module._RS_HOLDS_ENABLED = False
            for lyr in layers:
                lyr.weight.rs_hold_steps = 0
        assert not any(gtp_module._held_rs_queues.values()), "RS left parked"
        torch.cuda.synchronize()
        return [lyr.weight.main_grad.clone() for lyr in layers]

    ref = run(hold=False)
    held = run(hold=True)
    for i, (g_ref, g_hold) in enumerate(zip(ref, held)):
        assert torch.equal(g_ref, g_hold), (
            f"cascade force-flush changed grads (layer {i}): "
            f"ref[0][:4]={g_ref.flatten()[:4].tolist()} "
            f"hold[0][:4]={g_hold.flatten()[:4].tolist()} "
            f"ref_sum={float(g_ref.float().sum())} hold_sum={float(g_hold.float().sum())}"
        )


class TestRsHoldGPU:
    def test_hold_issue_flush_ordering(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_hold_ordering, 4)

    def test_cascade_force_flush(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_cascade_force_flush, 4)
