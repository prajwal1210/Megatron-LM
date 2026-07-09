# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the per-weight-type GTP prefetch lookahead.

Covers (all CPU-runnable, single-process):
1.  TestPrefetchStepsParser   - update_gtp_config(prefetch_steps_rules=...): validation,
                                whitespace, first-match ordering, dedup, malformed,
                                range 1..3, atomic max_fetch_steps, post-freeze reject.
2.  TestClassifyStamping      - classify_gtp_chains stamps next/prev_fetch_steps
                                (first-match, unmatched -> 1:1, embedding bwd opt-out).
3.  TestWeightsToPrefetch     - _weights_to_prefetch active-window walk: variable depths,
                                ineligible holes skipped, issued-exactly-once, and the
                                load-bearing regression: a drained-but-unconsumed target
                                (handle=None, _already_ag_drained=True, pending=True) is
                                still counted active and NOT re-issued.
4.  TestGenerationKeyedCache  - GTPWeightCache generation keying: max_fetch_steps=3 ->
                                4 concurrent same-key AG tickets get distinct buffers;
                                fwd/bwd don't alias (bf16); RS keys unchanged;
                                max_fetch_steps==1 uses the original key/alloc path.

These do not require CUDA / a process group: GTPShardedParam is constructed on CPU
tensors with a mock group, mirroring the single-process _FakeGroup tests in
test_gtp_basics.py.
"""

import pytest
import torch

from megatron.core.tensor_parallel.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTP_CONFIG,
    GTPShardedParam,
    GTPWeightCache,
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


@pytest.fixture(autouse=True)
def reset_prefetch_state():
    """Reset all GTP mutable state the lookahead feature touches.

    Extends the gtp_test_utils reset (which only clears _chain_state) to also reset
    the new GTPRematConfig fields, the module-level freeze flag, the recompute chain
    map, the global buffer cache, and the in-flight param set — so each test starts
    from the byte-identical default (no rules, max_fetch_steps==1, not frozen).
    """
    yield
    GTP_CONFIG.prefetch_steps_rules = []
    GTP_CONFIG.max_fetch_steps = 1
    gtp_module._GTP_PREFETCH_RULES_FROZEN = False
    GTPShardedParam._chain_state = {}
    GTPShardedParam._recompute_chain_state = {}
    gtp_module._GTP_CACHE = None
    gtp_module._inflight_comm_params.clear()


def _set_rules(rules):
    """Set prefetch_steps_rules from a fresh (unfrozen) state."""
    gtp_module._GTP_PREFETCH_RULES_FROZEN = False
    update_gtp_config(prefetch_steps_rules=rules)


# ---------------------------------------------------------------------------
# 1. Parser / validation (update_gtp_config(prefetch_steps_rules=...))
# ---------------------------------------------------------------------------


class TestPrefetchStepsParser:

    def test_empty_rules_default(self):
        _set_rules([])
        assert GTP_CONFIG.prefetch_steps_rules == []
        assert GTP_CONFIG.max_fetch_steps == 1

    def test_valid_rules_set(self):
        _set_rules([("mixer.in_proj", 3, 1), ("self_attention", 2, 2)])
        assert GTP_CONFIG.prefetch_steps_rules == [
            ("mixer.in_proj", 3, 1),
            ("self_attention", 2, 2),
        ]

    def test_max_fetch_steps_derived_atomically(self):
        # max over all next/prev across all rules.
        _set_rules([("a", 2, 1), ("b", 1, 3)])
        assert GTP_CONFIG.max_fetch_steps == 3
        _set_rules([("a", 2, 2)])
        assert GTP_CONFIG.max_fetch_steps == 2
        _set_rules([("a", 1, 1)])
        assert GTP_CONFIG.max_fetch_steps == 1

    def test_whitespace_trimmed(self):
        _set_rules([("  mixer.in_proj  ", 2, 1)])
        assert GTP_CONFIG.prefetch_steps_rules == [("mixer.in_proj", 2, 1)]

    def test_ordering_first_match_preserved(self):
        # Ordering is preserved (first-match-wins is enforced at classify time).
        _set_rules([("linear", 3, 1), ("linear_fc1", 1, 1)])
        assert GTP_CONFIG.prefetch_steps_rules[0][0] == "linear"
        assert GTP_CONFIG.prefetch_steps_rules[1][0] == "linear_fc1"

    def test_duplicate_substring_rejected(self):
        with pytest.raises(ValueError, match="duplicate substring"):
            _set_rules([("in_proj", 2, 1), ("in_proj", 1, 1)])

    def test_duplicate_after_trim_rejected(self):
        with pytest.raises(ValueError, match="duplicate substring"):
            _set_rules([("in_proj", 2, 1), ("  in_proj ", 1, 1)])

    def test_empty_substring_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _set_rules([("   ", 2, 1)])

    def test_malformed_tuple_rejected(self):
        with pytest.raises(ValueError, match="3-tuple"):
            _set_rules([("in_proj", 2)])

    def test_non_int_steps_rejected(self):
        with pytest.raises(ValueError, match="must be an int"):
            _set_rules([("in_proj", "2", 1)])

    def test_bool_steps_rejected(self):
        # bool is an int subclass but never a valid step count.
        with pytest.raises(ValueError, match="must be an int"):
            _set_rules([("in_proj", True, 1)])

    def test_steps_below_range_rejected(self):
        with pytest.raises(ValueError, match="out of"):
            _set_rules([("in_proj", 0, 1)])

    def test_steps_above_range_rejected(self):
        with pytest.raises(ValueError, match="out of"):
            _set_rules([("in_proj", 4, 1)])

    def test_steps_at_range_bounds_ok(self):
        _set_rules([("a", 1, 3), ("b", 3, 1)])
        assert GTP_CONFIG.max_fetch_steps == 3

    def test_non_str_substring_rejected(self):
        with pytest.raises(ValueError, match="substring must be a str"):
            _set_rules([(123, 2, 1)])

    def test_non_list_rejected(self):
        with pytest.raises(ValueError, match="must be a list"):
            _set_rules("in_proj:2:1")

    def test_max_fetch_steps_not_settable_directly(self):
        with pytest.raises(ValueError, match="derived from prefetch_steps_rules"):
            update_gtp_config(max_fetch_steps=2)

    def test_post_freeze_update_rejected(self):
        _set_rules([("in_proj", 2, 1)])
        # Simulate classify / first-ticket having committed the schedule.
        gtp_module._GTP_PREFETCH_RULES_FROZEN = True
        with pytest.raises(RuntimeError, match="frozen"):
            update_gtp_config(prefetch_steps_rules=[("out_proj", 1, 1)])


# ---------------------------------------------------------------------------
# 2. classify_gtp_chains stamping
# ---------------------------------------------------------------------------


class _ParamHolder(torch.nn.Module):
    """Wrap GTPShardedParams under given dotted names so named_parameters() yields them."""

    def __init__(self, named_params):
        super().__init__()
        # Register under flat names via a ModuleDict-ish trick: store as attributes
        # on nested modules so the dotted name matches the key.
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


class TestClassifyStamping:

    def test_first_match_wins(self):
        _set_rules([("mixer", 3, 1), ("mixer.in_proj", 1, 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mixer.in_proj.weight": p})
        classify_gtp_chains(model)
        # "mixer" matches first (earlier in the ordered list) -> 3:1, NOT 1:1.
        assert (p.next_fetch_steps, p.prev_fetch_steps) == (3, 1)

    def test_specific_before_general(self):
        _set_rules([("mixer.in_proj", 3, 2), ("mixer", 1, 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mixer.in_proj.weight": p})
        classify_gtp_chains(model)
        assert (p.next_fetch_steps, p.prev_fetch_steps) == (3, 2)

    def test_unmatched_defaults_to_one(self):
        _set_rules([("mixer.in_proj", 3, 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mlp.linear_fc2.weight": p})
        classify_gtp_chains(model)
        assert (p.next_fetch_steps, p.prev_fetch_steps) == (1, 1)

    def test_no_rules_all_default(self):
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.self_attention.linear_qkv.weight": p})
        classify_gtp_chains(model)
        assert (p.next_fetch_steps, p.prev_fetch_steps) == (1, 1)

    def test_embedding_bwd_optout_preserved(self):
        _set_rules([("embedding", 2, 2)])
        p = _mk_param()
        model = _ParamHolder({"embedding.word_embeddings.weight": p})
        classify_gtp_chains(model)
        # steps still stamped, but the bwd prefetch opt-out must be preserved.
        assert (p.next_fetch_steps, p.prev_fetch_steps) == (2, 2)
        assert p._need_weight_prefetch_bwd is False

    def test_classify_freezes_rules(self):
        _set_rules([("mixer", 2, 1)])
        p = _mk_param()
        model = _ParamHolder({"decoder.layers.0.mixer.in_proj.weight": p})
        classify_gtp_chains(model)
        assert gtp_module._GTP_PREFETCH_RULES_FROZEN is True
        with pytest.raises(RuntimeError, match="frozen"):
            update_gtp_config(prefetch_steps_rules=[("x", 1, 1)])


# ---------------------------------------------------------------------------
# 3. _weights_to_prefetch active-window walk
# ---------------------------------------------------------------------------


def _build_chain(n):
    """Build a linked chain of n GTPShardedParams (next_w / prev_w wired)."""
    params = [_mk_param() for _ in range(n)]
    for i in range(n - 1):
        params[i].next_w = params[i + 1]
        params[i + 1].prev_w = params[i]
    return params


class TestWeightsToPrefetch:

    def test_depth_one_returns_single_neighbor(self):
        c = _build_chain(4)
        out = c[0]._weights_to_prefetch("next_w", 1)
        assert out == [c[1]]

    def test_depth_one_returns_empty_if_pending(self):
        c = _build_chain(4)
        c[1]._prefetch_pending_consume = True
        # next_w already in flight -> nothing new to issue, but window is full.
        out = c[0]._weights_to_prefetch("next_w", 1)
        assert out == []

    def test_depth_three_fills_window(self):
        c = _build_chain(5)
        out = c[0]._weights_to_prefetch("next_w", 3)
        assert out == [c[1], c[2], c[3]]

    def test_window_refill_one_slot(self):
        # Head issues 3; mark them pending. Consuming head, the next consumer (c[1])
        # with depth 3 should only need to issue the one slot that drained (c[4]).
        c = _build_chain(6)
        for w in (c[1], c[2], c[3]):
            w._prefetch_pending_consume = True
        out = c[1]._weights_to_prefetch("next_w", 3)
        # c[2], c[3] pending (count, not issue); c[4] not pending -> issue.
        assert out == [c[4]]

    def test_ineligible_hole_skipped_not_counted(self):
        # c[2] opts out of prefetch entirely; it must be skipped without consuming a
        # window slot, so depth 2 from c[0] issues c[1] and c[3] (not c[1], c[2]).
        c = _build_chain(5)
        c[2]._need_weight_prefetch = False
        out = c[0]._weights_to_prefetch("next_w", 2)
        assert out == [c[1], c[3]]

    def test_bwd_optout_skipped_when_need_bwd(self):
        # c[2] has bwd opt-out (like embedding). With need_bwd=True it is skipped
        # without counting; with need_bwd=False it counts normally.
        c = _build_chain(5)
        c[2]._need_weight_prefetch_bwd = False
        out_bwd = c[4]._weights_to_prefetch("prev_w", 2, need_bwd=True)
        # walking prev_w from c[4]: c[3] (count+issue), c[2] (skip), c[1] (count+issue)
        assert out_bwd == [c[3], c[1]]
        # Reset pending so the fwd-direction check is independent.
        for w in c:
            w._prefetch_pending_consume = False
        out_fwd = c[4]._weights_to_prefetch("prev_w", 2, need_bwd=False)
        # c[3] (issue), c[2] (issue) — opt-out ignored when not need_bwd.
        assert out_fwd == [c[3], c[2]]

    def test_variable_depths_distinct_per_consumer(self):
        # Each consumer's own steps drives its window (the consumed weight's steps).
        c = _build_chain(6)
        assert c[0]._weights_to_prefetch("next_w", 3) == [c[1], c[2], c[3]]
        for w in c:
            w._prefetch_pending_consume = False
        assert c[0]._weights_to_prefetch("next_w", 1) == [c[1]]
        for w in c:
            w._prefetch_pending_consume = False
        assert c[0]._weights_to_prefetch("next_w", 2) == [c[1], c[2]]

    def test_each_target_issued_exactly_once_over_window_walk(self):
        # Simulate depths [3,1,2,1] across consumers; mark each issued target pending
        # as it would be at the real issue site, and assert no target is returned twice.
        c = _build_chain(8)
        depths = [3, 1, 2, 1]
        issued = []
        for i, d in enumerate(depths):
            for tgt in c[i]._weights_to_prefetch("next_w", d):
                assert tgt not in issued, f"{c.index(tgt)} issued twice"
                issued.append(tgt)
                tgt._prefetch_pending_consume = True  # mirror the issue site

    def test_drained_unconsumed_target_not_reissued(self):
        """LOAD-BEARING regression.

        A target that wait_async_comms() drained mid-iteration has handle=None and
        _already_ag_drained=True, but is still pending consume. The active-window walk
        must count it active and NOT re-issue it. A handle-only dedup (the hier bug)
        would re-issue here -> double collective + clobbered buffer.
        """
        c = _build_chain(4)
        # c[1] was prefetched, then drained by wait_async_comms:
        c[1]._prefetch_pending_consume = True
        c[1]._prefetch_handle = None  # nulled by _wait_param_gather
        c[1]._already_ag_drained = True  # set by wait_async_comms
        # Depth-1 walk from head: c[1] is pending -> not re-issued, window full.
        out = c[0]._weights_to_prefetch("next_w", 1)
        assert out == []
        # Depth-2 walk: c[1] pending (counts), c[2] free -> only c[2] issued.
        out2 = c[0]._weights_to_prefetch("next_w", 2)
        assert out2 == [c[2]]

    def test_pending_with_null_handle_undrained_asserts(self):
        """The invariant guard: pending=True with handle=None and NOT drained is a
        bug (a prefetched weight can't lose its handle without being drained)."""
        c = _build_chain(3)
        c[1]._debug_name = "c1"
        c[1]._prefetch_pending_consume = True
        c[1]._prefetch_handle = None
        c[1]._already_ag_drained = False
        with pytest.raises(AssertionError, match="invariant violated"):
            c[0]._weights_to_prefetch("next_w", 1)


# ---------------------------------------------------------------------------
# 4. Generation-keyed buffer cache
# ---------------------------------------------------------------------------


class TestGenerationKeyedCache:

    def _param(self, shape=(8, 4), gtp_size=2, expert_idx=None):
        p = GTPShardedParam(torch.zeros(*shape))
        p.group = _FakeGroup(size=gtp_size)
        p.expert_idx = expert_idx
        p.pad_length = 0
        p._quantizer = None
        return p

    def test_default_max1_distinct_only_via_empty_pool(self):
        # max_fetch_steps == 1: gen-keying OFF. Two concurrent same-key reserves get
        # distinct buffers (pool empty), but after release the buffer is reused — the
        # exact pre-feature behavior.
        assert GTP_CONFIG.max_fetch_steps == 1
        cache = GTPWeightCache()
        p = self._param()
        t1 = cache.reserve(p, torch.bfloat16, fwd=True)
        b1 = cache.get(t1)
        t2 = cache.reserve(p, torch.bfloat16, fwd=True)
        b2 = cache.get(t2)
        assert b1 is not b2  # concurrent -> distinct
        # keys are identical (no gen suffix) in the default path.
        assert cache._slots[t1].key == cache._slots[t2].key
        cache.release(t1)
        t3 = cache.reserve(p, torch.bfloat16, fwd=True)
        b3 = cache.get(t3)
        assert b3 is b1  # reused from pool after release

    def test_max1_key_unchanged(self):
        # The slot key in default mode equals the raw _get_cache_key (no suffix).
        assert GTP_CONFIG.max_fetch_steps == 1
        cache = GTPWeightCache()
        p = self._param()
        t = cache.reserve(p, torch.bfloat16, fwd=True)
        assert cache._slots[t].key == p._get_cache_key(torch.bfloat16, True, False)

    def test_max3_four_concurrent_distinct_buffers(self):
        _set_rules([("x", 3, 3)])  # max_fetch_steps = 3 -> n_gen = 4
        assert GTP_CONFIG.max_fetch_steps == 3
        cache = GTPWeightCache()
        p = self._param()
        tickets = [cache.reserve(p, torch.bfloat16, fwd=True) for _ in range(4)]
        bufs = [cache.get(t) for t in tickets]
        # 4 concurrent same-base-key AG tickets -> 4 distinct physical buffers.
        ids = {id(b) for b in bufs}
        assert len(ids) == 4
        # 5th wraps round-robin to gen 0 -> reuses the first (gen-0) buffer slot key,
        # but since none released, get() allocates from the (empty) gen-0 pool again.
        # The key, at least, must repeat gen 0:
        keys = [cache._slots[t].key for t in tickets]
        gens = [k[-1] for k in keys]
        assert gens == [0, 1, 2, 3]

    def test_max3_fwd_bwd_no_alias_bf16(self):
        # bf16 base key omits direction; in enhanced mode fwd/bwd must NOT share a
        # buffer bucket (recompute-fwd AG vs dgrad-bwd AG of the same weight concurrent).
        _set_rules([("x", 3, 3)])
        cache = GTPWeightCache()
        p = self._param()
        tf = cache.reserve(p, torch.bfloat16, fwd=True)
        tb = cache.reserve(p, torch.bfloat16, fwd=False)
        bf = cache.get(tf)
        bb = cache.get(tb)
        assert bf is not bb
        # Keys must differ (direction-scoped), even at the same generation index.
        assert cache._slots[tf].key != cache._slots[tb].key

    def test_max3_rs_keys_unchanged(self):
        # Reduce-scatter is never depth-prefetched: its key must carry no gen suffix
        # even when max_fetch_steps > 1.
        _set_rules([("x", 3, 3)])
        cache = GTPWeightCache()
        p = self._param()
        t = cache.reserve(p, torch.bfloat16, fwd=True, reduce_scatter=True)
        assert cache._slots[t].key == p._get_cache_key(torch.bfloat16, True, True)

    def test_max2_two_concurrent_then_reuse(self):
        _set_rules([("x", 2, 1)])  # max_fetch_steps = 2 -> n_gen = 3
        assert GTP_CONFIG.max_fetch_steps == 2
        cache = GTPWeightCache()
        p = self._param()
        keys = [cache._slots[cache.reserve(p, torch.bfloat16, fwd=True)].key for _ in range(3)]
        gens = [k[-1] for k in keys]
        assert gens == [0, 1, 2]
        # 4th wraps to gen 0.
        k4 = cache._slots[cache.reserve(p, torch.bfloat16, fwd=True)].key
        assert k4[-1] == 0

    def test_reserve_freezes_rules(self):
        # First ticket commits the schedule (gen sizing reads max_fetch_steps).
        cache = GTPWeightCache()
        p = self._param()
        cache.reserve(p, torch.bfloat16, fwd=True)
        assert gtp_module._GTP_PREFETCH_RULES_FROZEN is True
