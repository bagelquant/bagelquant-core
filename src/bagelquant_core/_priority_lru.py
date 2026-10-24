"""Operation-local priority indexes over one ordered value authority."""
from collections import OrderedDict


_MISSING = object()


class PriorityLRU(OrderedDict):
    """Keep the oldest key at each priority without scanning retained proofs.

    Callers supply their existing lock and per-entry overhead reservation. The
    indexes retain only references to the same keys, never another value body.
    """

    def __init__(self, priority):
        super().__init__()
        self._priority = priority
        self._keys = {}

    def __setitem__(self, key, value):
        level = self._priority(key)
        existing = key in self
        super().__setitem__(key, value)
        if not existing:
            self._keys.setdefault(level, OrderedDict())[key] = None

    def __delitem__(self, key):
        super().__delitem__(key)
        del self._keys[self._priority(key)][key]

    def pop(self, key, default=_MISSING):
        value = super().pop(key, _MISSING)
        if value is _MISSING:
            if default is _MISSING:
                raise KeyError(key)
            return default
        del self._keys[self._priority(key)][key]
        return value

    def popitem(self, last=True):
        key, value = super().popitem(last=last)
        del self._keys[self._priority(key)][key]
        return key, value

    def move_to_end(self, key, last=True):
        super().move_to_end(key, last=last)
        self._keys[self._priority(key)].move_to_end(key, last=last)

    def clear(self):
        super().clear()
        self._keys.clear()

    def first_at_most(self, maximum):
        """Select the same priority/LRU victim with a fixed number of lookups."""
        for level in range(maximum + 1):
            keys = self._keys.get(level)
            if keys:
                return next(iter(keys))
        return None
