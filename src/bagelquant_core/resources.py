"""Execution-only resource limits, excluded from numerical graph identity."""
from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import subprocess
import sys


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

    def under_pressure(self, resident_mib: float):
        """Reduce subsequent batches and suspend new concurrent nodes at the soft limit."""
        if resident_mib < self.memory_target_mib:
            return self
        return replace(self, parallel_nodes=1, batch_rows=max(1, self.batch_rows // 2), cache_mib=0)


def current_resident_mib() -> float:
    """Read current process resident memory, rather than a historical high-water mark."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        process = kernel.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not query(process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize / (1024 * 1024)
    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", encoding="ascii") as stream:
            pages = int(stream.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    # macOS ps reports current RSS in KiB; sample only at admission boundaries.
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True, capture_output=True, text=True)
    return float(result.stdout.strip()) / 1024


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
