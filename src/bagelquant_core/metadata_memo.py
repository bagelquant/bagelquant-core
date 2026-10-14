"""Bounded immutable metadata reuse within one explicitly admitted operation."""
from collections import OrderedDict
from copy import deepcopy
from threading import RLock


class MetadataMemo:
    def __init__(self, maximum_bytes: int):
        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 0:
            raise ValueError("max_metadata_bytes must be a nonnegative integer")
        self.maximum_bytes = maximum_bytes
        self.bytes = 0
        self.values = OrderedDict()
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
                _, (_, removed) = self.values.popitem(last=False)
                self.bytes -= removed
            self.values[key] = (deepcopy(value), charge)
            self.bytes += charge

    def close(self):
        with self.lock:
            self.closed = True
            self.values.clear()
            self.bytes = 0
