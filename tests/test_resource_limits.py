"""Execution-only cache ceilings fail at the public resource boundary."""
import pytest

from bagelquant_core.resources import ResourceLimits


@pytest.mark.parametrize("cache", [None, True, False, 1.5, float("nan"), float("inf"), -1, 10240])
def test_cache_limit_requires_a_nonnegative_integer_below_memory(cache):
    with pytest.raises(ValueError, match="cache_mib"):
        ResourceLimits(cache_mib=cache)


def test_disabled_cache_remains_valid_when_budget_is_divided():
    limits = ResourceLimits(total_threads=4, cache_mib=0)
    assert limits.for_workers(4).cache_mib == 0
