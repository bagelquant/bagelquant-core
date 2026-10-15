"""Finite, byte-bounded reuse of immutable execution preparation.

This cache owns no completion or currentness authority. Mutable graph guards and
missing optional proofs are never cached. Domain objects remain in the keys so
an opaque caller-provided signature cannot hide different coordinates.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, RLock


class ExecutionPreparation:
    def __init__(self, maximum_bytes: int):
        self.maximum_bytes = maximum_bytes
        self.bytes = 0
        self.peak_bytes = 0
        self._values = OrderedDict()
        self._histories = {}
        self._in_flight = {}
        self.lock = RLock()
        self.counts = Counter()

    def restrict(self, maximum_bytes: int) -> None:
        """A later smaller admission may reduce, never enlarge, this budget."""
        with self.lock:
            self.maximum_bytes = min(self.maximum_bytes, maximum_bytes)
            self._evict(0)

    def register_halo(self, domain, history: int) -> None:
        self._histories.setdefault(domain, set()).add(history)

    def histories(self, domain, history: int) -> tuple[int, ...]:
        return tuple(sorted(self._histories.get(domain, set()) | {history}))

    def _evict(self, required: int) -> None:
        while self._values and self.bytes + required > self.maximum_bytes:
            _, (_, charge) = self._values.popitem(last=False)
            self.bytes -= charge
            self.counts["evictions"] += 1

    def remember(self, kind, key, build, size, *, check_canceled=None):
        """Coalesce immutable preparation, retaining only admitted successes."""
        locator = (kind, key)
        while True:
            with self.lock:
                found = self._values.pop(locator, None)
                if found is not None:
                    self._values[locator] = found
                    self.counts[kind + "_hits"] += 1
                    return found[0]
                pending = self._in_flight.get(locator)
                if pending is None:
                    pending = self._in_flight[locator] = Event()
                    self.counts[kind + "_builds"] += 1
                    break
            while not pending.wait(timeout=0.1):
                if check_canceled is not None:
                    check_canceled()
        try:
            value = build()
            # Optional absent evidence can become available through explicit
            # maintenance. Its absence must never acquire cached authority.
            if value is None:
                return None
            charge = 1024 + size(value)
            with self.lock:
                if charge <= self.maximum_bytes:
                    self._evict(charge)
                    self._values[locator] = (value, charge)
                    self.bytes += charge
                    self.peak_bytes = max(self.peak_bytes, self.bytes)
            return value
        finally:
            with self.lock:
                self._in_flight.pop(locator)
                pending.set()

    def clear(self) -> None:
        with self.lock:
            self._values.clear()
            self.bytes = 0

    def report(self) -> dict:
        with self.lock:
            return {**self.counts, "resident_bytes": self.bytes,
                "peak_bytes": self.peak_bytes, "maximum_bytes": self.maximum_bytes}


_active: ContextVar[ExecutionPreparation | None] = ContextVar("core_execution_preparation", default=None)


def active_preparation() -> ExecutionPreparation | None:
    return _active.get()


@contextmanager
def preparation_scope(preparation: ExecutionPreparation):
    token = _active.set(preparation)
    try:
        yield preparation
    finally:
        _active.reset(token)


def domain_bytes(domain) -> int:
    size = domain.times.estimated_size() + domain.asset_ids.estimated_size()
    if domain.is_dynamic:
        size += domain.grid_lazy().collect().estimated_size()
    return 2 * size
