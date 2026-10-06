"""Execution-only resource limits, excluded from numerical graph identity."""
from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os


@dataclass(frozen=True)
class ResourceLimits:
    total_threads: int = 4
    parallel_nodes: int = 1
    memory_target_mib: int = 10240
    lightgbm_threads: int = 2
    histogram_pool_mib: int = 512
    batch_rows: int = 250000
    cache_mib: int = 1024

    def __post_init__(self):
        for name in ("total_threads", "parallel_nodes", "memory_target_mib", "lightgbm_threads",
                     "histogram_pool_mib", "batch_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.cache_mib < 0 or self.cache_mib >= self.memory_target_mib:
            raise ValueError("cache_mib must be nonnegative and below memory_target_mib")

    def effective(self, logical_cpus: int):
        threads = min(self.total_threads, max(1, logical_cpus))
        nodes = min(self.parallel_nodes, threads)
        return replace(self, total_threads=threads, parallel_nodes=nodes,
                       lightgbm_threads=min(self.lightgbm_threads, threads))

    def threads_per_worker(self, workers: int) -> int:
        return max(1, self.total_threads // min(max(1, workers), self.total_threads))

    def for_workers(self, workers: int) -> ResourceLimits:
        """Divide one admission budget among simultaneously running kernels."""
        workers = min(max(1, workers), self.total_threads)
        threads = self.threads_per_worker(workers)
        memory = max(1, self.memory_target_mib // workers)
        return replace(self, total_threads=threads, parallel_nodes=1,
            lightgbm_threads=min(self.lightgbm_threads, threads),
            histogram_pool_mib=max(1, self.histogram_pool_mib // workers),
            memory_target_mib=memory, batch_rows=max(1, self.batch_rows // workers),
            cache_mib=min(self.cache_mib // workers, memory - 1))


def kernel_batch_rows(bytes_per_row: int, *, history_rows: int = 0,
                      maximum_bytes: int = 64 * 1024 * 1024) -> int:
    """Bound temporary numerical arrays by the admitted worker's row/memory budget."""
    limits = active_resource_limits()
    working = min(maximum_bytes, max(1, limits.memory_target_mib * 1024 * 1024 // 8))
    return min(limits.batch_rows, max(1, working // max(1, bytes_per_row) - history_rows))


_active_limits: ContextVar[ResourceLimits | None] = ContextVar("bagelquant_resource_limits", default=None)


def active_resource_limits() -> ResourceLimits:
    selected = _active_limits.get()
    if selected is not None:
        return selected
    raw = os.environ.get("BAGELQUANT_RESOURCE_LIMITS")
    selected = ResourceLimits(**json.loads(raw)) if raw else ResourceLimits()
    return selected


@contextmanager
def resource_limits(limits: ResourceLimits):
    """Provide numerical operators with execution-only limits in this context."""
    token = _active_limits.set(limits)
    try:
        yield limits
    finally:
        _active_limits.reset(token)
