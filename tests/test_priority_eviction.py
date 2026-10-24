"""Priority eviction must preserve LRU policy with work bounded by arrivals."""
from collections import OrderedDict
from copy import deepcopy
from random import Random

import pytest

from bagelquant_core import metadata_memo
from bagelquant_core._priority_lru import PriorityLRU
from bagelquant_core.metadata_memo import MetadataMemo
from bagelquant_core.preparation import ExecutionPreparation, PreparationMetadataMemo


class CountedKey:
    def __init__(self, value, counter):
        self.value, self.counter = value, counter

    def __hash__(self):
        self.counter[0] += 1
        return self.value

    def __eq__(self, other):
        return isinstance(other, CountedKey) and self.value == other.value


def test_ordered_mapping_mutations_keep_priority_victims_synchronized():
    actual = PriorityLRU(lambda key: key[0])
    reference = OrderedDict()

    def check():
        assert list(actual.items()) == list(reference.items())
        for maximum in range(3):
            expected = next((key for priority in range(maximum + 1)
                for key in reference if key[0] == priority), None)
            assert actual.first_at_most(maximum) == expected

    for key, value in [((2, "old"), 1), ((0, "low"), None), ((1, "middle"), 3), ((2, "new"), 4)]:
        actual[key] = reference[key] = value
        check()
    actual[(2, "old")] = reference[(2, "old")] = 10
    check()  # Replacing a value does not make its key newer.
    for key, last in [((2, "new"), False), ((0, "low"), True), ((1, "middle"), False)]:
        actual.move_to_end(key, last=last)
        reference.move_to_end(key, last=last)
        check()
    assert actual.pop((0, "low")) == reference.pop((0, "low")) is None
    check()
    assert actual.pop((0, "absent"), "fallback") == reference.pop((0, "absent"), "fallback")
    with pytest.raises(KeyError):
        actual.pop((0, "absent"))
    with pytest.raises(KeyError):
        actual.move_to_end((0, "absent"))
    check()
    assert actual.popitem(last=False) == reference.popitem(last=False)
    check()
    del actual[(2, "old")]
    del reference[(2, "old")]
    check()
    actual.update([((0, "replacement"), 7), ((2, "another"), 8)])
    reference.update([((0, "replacement"), 7), ((2, "another"), 8)])
    check()
    assert actual.popitem() == reference.popitem()
    check()
    actual.clear()
    reference.clear()
    check()
    assert actual.setdefault((1, "after_clear"), 9) == reference.setdefault((1, "after_clear"), 9)
    check()


@pytest.mark.parametrize("owner", ["preparation", "metadata"])
@pytest.mark.parametrize("action", ["high_churn", "low_rejection"])
def test_full_priority_cache_work_is_linear_in_new_arrivals(monkeypatch, owner, action):
    size = 128
    classified, hashed = [0], [0]
    if owner == "preparation":
        priority = ExecutionPreparation._priority

        def counted(key):
            classified[0] += 1
            return priority(key)

        monkeypatch.setattr(ExecutionPreparation, "_priority", staticmethod(counted))
        cache = ExecutionPreparation(size * 1024)

        def put(index, high):
            cache.remember("interval_proof" if high else "numerical", CountedKey(index, hashed),
                lambda: index, lambda _value: 0)

        resident = cache._values
    else:
        priority = metadata_memo.metadata_priority

        def counted(key):
            classified[0] += 1
            return priority(key)

        monkeypatch.setattr(metadata_memo, "metadata_priority", counted)
        cache = MetadataMemo(size * 1024)

        def put(index, high):
            cache.put(("manifest_authority" if high else "manifest", CountedKey(index, hashed)), index, 0)

        resident = cache.values
    for index in range(size):
        put(index, True)
    classified[0], hashed[0] = 0, 0
    for index in range(size, 2 * size):
        put(index, action == "high_churn")
    # Old admission scanned all resident high-priority keys for every missing
    # lower-priority victim, making both counters quadratic in this fixture.
    assert classified[0] <= 20 * size
    assert hashed[0] <= 50 * size
    assert len(resident) == size and cache.bytes == size * 1024
    expected = range(size, 2 * size) if action == "high_churn" else range(size)
    assert {value for value, _charge in resident.values()} == set(expected)
    if owner == "preparation":
        assert cache.counts["evictions"] == (size if action == "high_churn" else 0)
        assert cache.counts["numerical_nonadmissions"] == (size if action == "low_rejection" else 0)


class Reference:
    """Simple list-based policy oracle; each entry is oldest to newest."""
    def __init__(self, maximum, *, replace):
        self.maximum, self.replace = maximum, replace
        self.entries = []

    @property
    def bytes(self):
        return sum(entry[2] for entry in self.entries)

    def get(self, key):
        for index, entry in enumerate(self.entries):
            if entry[0] == key:
                self.entries.append(self.entries.pop(index))
                return deepcopy(entry[1])
        return None

    def put(self, key, value, charge, priority):
        found = next((index for index, entry in enumerate(self.entries) if entry[0] == key), None)
        if found is not None and not self.replace:
            self.get(key)
            return
        if charge > self.maximum:
            return
        if found is not None:
            self.entries.pop(found)
        while self.entries and self.bytes + charge > self.maximum:
            eligible = [entry[3] for entry in self.entries if entry[3] <= priority]
            if not eligible:
                return
            level = min(eligible)
            index = next(index for index, entry in enumerate(self.entries) if entry[3] == level)
            self.entries.pop(index)
        if self.bytes + charge <= self.maximum:
            self.entries.append((key, deepcopy(value), charge, priority))

    def restrict(self, maximum):
        self.maximum = min(self.maximum, maximum)
        while self.entries and self.bytes > self.maximum:
            level = min(entry[3] for entry in self.entries)
            index = next(index for index, entry in enumerate(self.entries) if entry[3] == level)
            self.entries.pop(index)


@pytest.mark.parametrize("owner", ["preparation", "metadata"])
def test_priority_lru_matches_independent_policy_for_hits_scopes_and_accounting(owner):
    random = Random(771)
    scopes = (object(), object())
    cache = ExecutionPreparation(6 * 1024) if owner == "preparation" else MetadataMemo(6 * 1024)
    view = PreparationMetadataMemo(cache, "fixture") if owner == "preparation" else cache
    reference = Reference(6 * 1024, replace=owner == "metadata")
    descriptors = [(level, scope, index) for level in (1, 2) for scope in (None, 0, 1) for index in range(4)]
    if owner == "preparation":
        descriptors.extend((0, None, index) for index in range(4))

    def key(descriptor):
        level, scope, index = descriptor
        value = ("manifest_authority" if level == 2 else "manifest", index)
        return value if scope is None else (scopes[scope], value)

    def locator(descriptor):
        return (("numerical", descriptor[2]) if descriptor[0] == 0 else
            ("metadata", ("fixture", key(descriptor)))) if owner == "preparation" else key(descriptor)

    for step in range(350):
        descriptor = random.choice(descriptors)
        operation = random.randrange(10)
        if operation < 3:
            expected = reference.get(descriptor)
            actual = (cache.get("numerical", descriptor[2]) if descriptor[0] == 0 else
                view.get(key(descriptor))) if owner == "preparation" else cache.get(key(descriptor))
            assert actual == expected
        elif operation < 8:
            encoded = random.choice((0, 16, 64, 1024))
            value = {"step": step}
            reference.put(descriptor, value, 1024 + 8 * encoded, descriptor[0])
            if owner == "preparation" and descriptor[0] == 0:
                cache.remember("numerical", descriptor[2], lambda: dict(value), lambda _value: 8 * encoded)
            else:
                view.put(key(descriptor), value, encoded)
            value["step"] = -1  # Metadata views must retain an independent tree.
        elif operation == 8:
            scope = random.randrange(2)
            view.discard_scope(scopes[scope])
            reference.entries = [entry for entry in reference.entries if entry[0][1] != scope]
        elif owner == "preparation":
            if step % 2:
                maximum = random.choice((2, 3, 5, 8)) * 1024
                cache.restrict(maximum)
                reference.restrict(maximum)
            else:
                cache.clear()
                reference.entries.clear()
        assert cache.bytes == reference.bytes
        actual = cache._values if owner == "preparation" else cache.values
        assert list(actual.items()) == [(locator(entry[0]), (entry[1], entry[2])) for entry in reference.entries]
        assert cache.bytes <= reference.maximum
    if owner == "metadata":
        cache.close()
        assert cache.bytes == 0 and not cache.values
        cache.put(("manifest_authority", 100), {"closed": True}, 0)
        assert not cache.values
    else:
        cache.clear()
        assert cache.bytes == 0 and not cache._values
