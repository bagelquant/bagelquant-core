"""Finite, byte-bounded reuse of immutable execution preparation.

This cache owns no completion or currentness authority. Mutable graph guards and
missing optional proofs are never cached. Domain objects remain in the keys so
an opaque caller-provided signature cannot hide different coordinates.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, RLock

from .metadata_memo import metadata_priority


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

    def get(self, kind, key):
        with self.lock:
            locator = (kind, key)
            found = self._values.pop(locator, None)
            if found is None:
                return None
            self._values[locator] = found
            self.counts[kind + "_hits"] += 1
            return found[0]

    @staticmethod
    def _priority(locator):
        kind, key = locator
        if kind == "metadata":
            return metadata_priority(key[1])
        return 2 if kind in {"interval_proof", "domain_signature"} else 0

    def _evict(self, required: int, *, locator=None) -> bool:
        while self._values and self.bytes + required > self.maximum_bytes:
            # Coordinate windows and decoded values must not flush immutable
            # proofs. Both classes still debit this one reserve; metadata can
            # evict metadata when its own working set fills the budget.
            priority = 2 if locator is None else self._priority(locator)
            candidate = None
            for level in range(priority + 1):
                candidate = next((key for key in self._values if self._priority(key) == level), None)
                if candidate is not None:
                    break
            if candidate is None:
                return False
            removed_kind = candidate[0]
            _, charge = self._values.pop(candidate)
            self.bytes -= charge
            self.counts["evictions"] += 1
            self.counts[removed_kind + "_evictions"] += 1
            self.counts[removed_kind + "_evicted_bytes"] += charge
        return self.bytes + required <= self.maximum_bytes

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
                self.counts[kind + "_waits"] += 1
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
                    if self._evict(charge, locator=locator):
                        self._values[locator] = (value, charge)
                        self.bytes += charge
                        self.peak_bytes = max(self.peak_bytes, self.bytes)
                    else:
                        self.counts[kind + "_nonadmissions"] += 1
                else:
                    self.counts[kind + "_oversize"] += 1
                    self.counts[kind + "_oversize_bytes"] += charge
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


class PreparationMetadataMemo:
    """Use the existing preparation reserve for immutable owner metadata.

    This view allocates no second cache. Serialized manifests keep their fresh
    public parsed trees; private indexes and successful source checks share the
    same admission, eviction and operation lifetime as numerical preparation.
    """

    def __init__(self, preparation, namespace):
        self.preparation = preparation
        self.namespace = namespace

    def get(self, key, *, copy_value=True):
        preparation = self.preparation
        locator = ("metadata", (self.namespace, key))
        with preparation.lock:
            found = preparation._values.pop(locator, None)
            if found is None:
                return None
            preparation._values[locator] = found
            preparation.counts["metadata_hits"] += 1
            value = found[0]
        return deepcopy(value) if copy_value else value

    def put(self, key, value, encoded_bytes):
        self.preparation.remember("metadata", (self.namespace, key),
            lambda: deepcopy(value), lambda _: 8 * encoded_bytes)

    def discard_scope(self, scope):
        preparation = self.preparation
        with preparation.lock:
            for locator in list(preparation._values):
                if locator[0] != "metadata":
                    continue
                namespace, key = locator[1]
                if namespace == self.namespace and isinstance(key, tuple) and key[0] is scope:
                    preparation.bytes -= preparation._values.pop(locator)[1]


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
