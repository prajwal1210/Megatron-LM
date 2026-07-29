# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GTP symmetric memory: NCCL window registration for GTP communication buffers.

This module keeps one ``ncclMemAlloc``-backed ``torch.cuda.MemPool`` per GTP process
group. Once ``register_gtp_symm_pool(group)`` registers a pool on its group, PyTorch's
ProcessGroupNCCL hook window-registers every allocation made inside
``gtp_symm_pool_ctx(group)``, which lets NCCL run its symmetric / NVLS kernels on
those buffers.

Three parts:
  - Pool lifecycle: create, register, query, and tear down the per-group pools,
    plus the allocation context ``gtp_symm_pool_ctx``. The GTP weight cache uses the
    context for all-gather output buffers.
  - DDP buffer registration: ``register_ddp_buffers_on_gtp_groups`` and its undo put
    the DDP param buffer (the all-gather input) in the window too.
  - ``symmetric_wgrad_pool`` (a ``RegisteredLifoPool``): recycled, window-registered
    send buffers for the wgrad reduce-scatter.

The launcher must set ``NCCL_NVLS_ENABLE=1`` and
``TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0`` before ``init_process_group``.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from contextlib import AbstractContextManager

import torch
import torch.distributed as dist

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registries (process-global, keyed by group.group_name)
# ---------------------------------------------------------------------------

# One MemPool per GTP/EGTP group, created once.
_pools: "dict[str, torch.cuda.MemPool]" = {}

# Groups whose pool registration is live. Maps name -> group (not a set) because
# teardown needs the group object to deregister.
_registered: "dict[str, object]" = {}

# Groups whose NCCL communicator has already been warmed up (see _warmup_group_comm).
_warmed_groups: "set[str]" = set()


# ---------------------------------------------------------------------------
# Pool lifecycle: create -> warm -> register -> allocate-into -> query -> deregister
# ---------------------------------------------------------------------------


def get_gtp_symm_pool(group: dist.ProcessGroup) -> torch.cuda.MemPool:
    """Return the per-group ``ncclMemAlloc``-backed MemPool, creating it once."""
    name = group.group_name
    pool = _pools.get(name)
    if pool is None:
        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)
        _pools[name] = pool
    return pool


def _warmup_group_comm(group: dist.ProcessGroup) -> None:
    """Run one tiny all-reduce so ``group``'s (lazily created) NCCL communicator exists
    before a pool is registered on it. Runs once per group."""
    if group.group_name in _warmed_groups:
        return
    warmup = torch.zeros(1, device=torch.cuda.current_device())
    dist.all_reduce(warmup, group=group)
    _warmed_groups.add(group.group_name)


def register_gtp_symm_pool(group: dist.ProcessGroup | None) -> torch.cuda.MemPool | None:
    """Create (if needed) and register the group's pool. Safe to call more than once;
    does nothing for ``None`` or single-rank groups.

    Issues a collective, so call it during model construction — before the first
    forward or any CUDA-graph capture. New segments register automatically afterwards.
    """
    if group is None or group.size() <= 1:
        return None
    pool = get_gtp_symm_pool(group)
    if group.group_name in _registered:
        return pool
    _warmup_group_comm(group)
    nccl_allocator.register_mem_pool(pool, group, symmetric=True)
    _registered[group.group_name] = group
    log_single_rank(
        logger,
        logging.INFO,
        f"[MCORE][GTP] Registered GTP cache pool on group {group.group_name} "
        f"(size={group.size()})",
    )
    return pool


def gtp_symm_pool_ctx(group: dist.ProcessGroup) -> AbstractContextManager[None]:
    """Context manager: allocations inside it come from ``group``'s pool. Collective-free
    (capture-safe); register the pool first or allocations are not window-registered."""
    return torch.cuda.use_mem_pool(get_gtp_symm_pool(group))


def is_gtp_symm_pool_registered(group: dist.ProcessGroup | None) -> bool:
    """True once ``register_gtp_symm_pool`` has registered this group's pool; also False for
    ``None`` and single-rank groups, which are never registered."""
    return group is not None and group.size() > 1 and group.group_name in _registered


def deregister_gtp_symm_pools() -> None:
    """Deregister every GTP-owned pool (a window left registered at process-group
    destruction makes NCCL abort). Call on all ranks before teardown; no-op if nothing
    was registered. Also drops the recycled send buffers living in these pools."""
    for name in sorted(_registered):
        nccl_allocator.deregister_mem_pool(_pools[name], _registered[name])
    _registered.clear()
    _pools.clear()
    symmetric_wgrad_pool.clear()
    _warmed_groups.clear()


# ---------------------------------------------------------------------------
# DDP param-buffer registration: put the GTP all-gather INPUT in the window
# (the pool itself is created by core DDP; see param_and_grad_buffer.py)
# ---------------------------------------------------------------------------


def _ddp_buffers(ddp_module: torch.nn.Module) -> list:
    """All param/grad buffers of a DDP-wrapped module (dense + expert-parallel)."""
    return list(getattr(ddp_module, "buffers", [])) + list(
        getattr(ddp_module, "expert_parallel_buffers", [])
    )


def _buffer_symm_groups(buf) -> list[dist.ProcessGroup]:
    """Return the GTP groups this buffer's pool must be (de)registered on: the buffer has
    a pool and a param_data section, and a param opted in via ``needs_nccl_mem``. Sorted
    by name so all ranks walk groups in the same order (mismatched order can deadlock).
    """
    if getattr(buf, "nccl_mem_pool", None) is None or getattr(buf, "param_data", None) is None:
        return []
    groups = {}
    for param in buf.params:
        if not getattr(param, "needs_nccl_mem", False):
            continue
        group = getattr(param, "group", None)
        if group is not None and group.size() > 1:
            groups.setdefault(group.group_name, group)
    return [group for _, group in sorted(groups.items())]


def register_ddp_buffers_on_gtp_groups(ddp_module: torch.nn.Module) -> None:
    """Register each DDP buffer's pool on its params' GTP groups. This puts the DDP param
    buffer — the GTP all-gather input under the distributed optimizer — in the symmetric
    window.

    Always symmetric: --disable-symmetric-registration scopes to the DP-group
    registration, not the GTP groups, which are opted into by --gtp-nccl-ub/--egtp-nccl-ub.
    """
    for buf in _ddp_buffers(ddp_module):
        for group in _buffer_symm_groups(buf):
            # buf.nccl_mem_pool is non-None here (checked in _buffer_symm_groups).
            _warmup_group_comm(group)
            nccl_allocator.register_mem_pool(buf.nccl_mem_pool, group, symmetric=True)
            log_single_rank(
                logger,
                logging.INFO,
                f"[MCORE][GTP] Registered DDP param/grad pool on GTP group "
                f"{group.group_name} (size={group.size()})",
            )


def deregister_ddp_buffers_from_gtp_groups(ddp_module: torch.nn.Module) -> None:
    """Undo ``register_ddp_buffers_on_gtp_groups`` at shutdown, before the process groups
    are destroyed. (The DP-group registration from --use-nccl-ub is deregistered
    separately by the training loop.)
    """
    for buf in _ddp_buffers(ddp_module):
        for group in _buffer_symm_groups(buf):
            # buf.nccl_mem_pool is non-None here (checked in _buffer_symm_groups).
            nccl_allocator.deregister_mem_pool(buf.nccl_mem_pool, group)
            log_single_rank(
                logger,
                logging.INFO,
                f"[MCORE][GTP] Deregistered DDP param/grad pool from GTP group "
                f"{group.group_name} (size={group.size()})",
            )


# ---------------------------------------------------------------------------
# RS send-buffer LIFO: recycled window-registered scratch for wgrad reduce-scatters
# ---------------------------------------------------------------------------


class RegisteredLifoPool:
    """A recycling cache of window-registered buffers, one free list per group.

    The wgrad reduce-scatter can only use symmetric collectives if its send buffer is
    window-registered, so the wgrad is written into a buffer from this cache. ``alloc``
    pops a free buffer (or allocates a new one through ``gtp_symm_pool_ctx``); ``free``
    returns it once the reduce-scatter has finished reading it. Buffers are shared by
    all weights of the same size, so memory stays at the peak number of in-flight
    reduce-scatters instead of one buffer per weight.

    CUDA graphs: the eager warmup iterations run the same reduce-scatter overlap as
    the captured steps, so by capture time the free lists already hold enough buffers
    and ``alloc`` only ever pops. Allocating a new buffer during capture would be
    illegal, so that case raises a clear error instead.

    Buffers are stored 1-D, so one free list serves every shape with the same element
    count. ``alloc`` returns a view tagged with ``_gtp_symm_group``; ``free`` ignores
    untagged tensors, which lets callers pass mixed buffer lists to both this pool and
    the plain scratch pool and have each take only its own.
    """

    def __init__(self) -> None:
        # (numel, dtype, group_name) -> list of free 1-D buffers.
        self._free: dict[tuple, list] = defaultdict(list)

    def alloc(
        self,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Return a buffer of ``shape`` from ``group``'s free list, allocating one if empty.

        Raises RuntimeError if a fresh allocation would be needed during CUDA-graph capture.
        """
        numel = int(math.prod(shape))
        bucket = self._free[(numel, dtype, group.group_name)]
        if bucket:
            flat = bucket.pop()
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "[GTP] RegisteredLifoPool exhausted during CUDA-graph capture "
                    f"(group={group.group_name}, numel={numel}, dtype={dtype}). The "
                    "eager warmup did not pre-populate enough RS send buffers for "
                    "the reduce-scatter overlap depth -- run more warmup iters, or "
                    "the RS concurrency changed between warmup and capture."
                )
            # Allocate from the group's registered pool when it has one; else plain memory.
            if is_gtp_symm_pool_registered(group):
                with gtp_symm_pool_ctx(group):
                    flat = torch.empty(numel, dtype=dtype, device=device)
            else:
                flat = torch.empty(numel, dtype=dtype, device=device)
        out = flat.view(shape)
        out._gtp_symm_group = group  # marks the buffer as pool-owned; free() keys on this
        return out

    def free(self, buf: torch.Tensor) -> None:
        """Return ``buf`` to its group's free list; no-op for untagged (foreign) buffers."""
        group = getattr(buf, "_gtp_symm_group", None)
        if group is None:
            return
        self._free[(buf.numel(), buf.dtype, group.group_name)].append(buf.reshape(-1))

    def clear(self) -> None:
        """Drop every cached buffer. Called at teardown, before the pools they alias go away."""
        self._free.clear()


# The process-wide send-buffer cache, used by generalized_tensor_parallelism. Lives here
# so deregister_gtp_symm_pools can drop its buffers at teardown.
symmetric_wgrad_pool = RegisteredLifoPool()
