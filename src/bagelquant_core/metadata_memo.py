"""Bounded immutable metadata reuse within one explicitly admitted operation."""
from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from ._priority_lru import PriorityLRU


def metadata_priority(key):
    """Keep compact authenticated authority ahead of large metadata bodies."""
    while isinstance(key, tuple) and len(key) == 2 and type(key[0]) is object:
        key = key[1]  # fresh transaction/audit scope
    return 2 if isinstance(key, tuple) and key and key[0] in {
        "manifest_authority", "source_bindings", "finite_contract_plan", "finite_contract_authority", "finite_contract_blocks",
        "streak_checkpoint_plan", "streak_checkpoint_authority"
    } else 1


class MetadataMemo:
    def __init__(self, maximum_bytes: int):
        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 0:
            raise ValueError("max_metadata_bytes must be a nonnegative integer")
        self.maximum_bytes = maximum_bytes
        self.bytes = 0
        self.values = PriorityLRU(metadata_priority)
        self.lock = RLock()
        self.closed = False

    def get(self, key, *, copy_value=True):
        with self.lock:
            if self.closed or key not in self.values:
                return None
            value, charge = self.values.pop(key)
            self.values[key] = (value, charge)
            return deepcopy(value) if copy_value else value

    def put(self, key, value, encoded_bytes):
        # Account for Python strings, dicts and cells as well as encoded bytes.
        charge = 1024 + 8 * encoded_bytes
        with self.lock:
            if self.closed or charge > self.maximum_bytes:
                return
            previous = self.values.pop(key, None)
            if previous is not None:
                self.bytes -= previous[1]
            while self.values and self.bytes + charge > self.maximum_bytes:
                candidate = self.values.first_at_most(1)
                if candidate is None:
                    if metadata_priority(key) == 1:
                        return
                    candidate = next(iter(self.values))
                _, removed = self.values.pop(candidate)
                self.bytes -= removed
            self.values[key] = (deepcopy(value), charge)
            self.bytes += charge

    def close(self):
        with self.lock:
            self.closed = True
            self.values.clear()
            self.bytes = 0

    def discard_scope(self, scope):
        with self.lock:
            for key in list(self.values):
                if isinstance(key, tuple) and key[0] is scope:
                    self.bytes -= self.values.pop(key)[1]


class _MetadataScope:
    def __init__(self, memo, scope):
        self.memo, self.scope = memo, scope

    def get(self, key, *, copy_value=True):
        return self.memo.get((self.scope, key), copy_value=copy_value)

    def put(self, key, value, encoded_bytes):
        self.memo.put((self.scope, key), value, encoded_bytes)


@contextmanager
def fresh_metadata_scope(memo):
    """Deduplicate a fresh verification without borrowing earlier successes.

    The view debits the caller's existing reserve and expires before a closing
    transaction ends. A zero/absent reserve retains the uncached verifier.
    """
    scope = object()
    try:
        yield None if memo is None else _MetadataScope(memo, scope)
    finally:
        if memo is not None:
            memo.discard_scope(scope)
