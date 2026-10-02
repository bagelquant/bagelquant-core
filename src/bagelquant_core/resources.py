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

    def under_pressure(self, resident_mib: float):
        """Reduce subsequent batches and suspend new concurrent nodes at the soft limit."""
        if resident_mib < self.memory_target_mib:
            return self
        return replace(self, parallel_nodes=1, batch_rows=max(1024, self.batch_rows // 2), cache_mib=0)


_active_limits: ContextVar[ResourceLimits | None] = ContextVar("bagelquant_resource_limits", default=None)


def active_resource_limits() -> ResourceLimits:
    selected = _active_limits.get()
    if selected is not None:
        return selected
    raw = os.environ.get("BAGELQUANT_RESOURCE_LIMITS")
    selected = ResourceLimits(**json.loads(raw)) if raw else ResourceLimits()
    selected = selected.effective(os.cpu_count() or 1)
    return selected


@contextmanager
def resource_limits(limits: ResourceLimits):
    """Provide numerical operators with execution-only limits in this context."""
    token = _active_limits.set(limits)
    try:
        yield limits
    finally:
        _active_limits.reset(token)
