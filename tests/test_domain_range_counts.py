"""Private scalar counts preserve independent calendar and active-key semantics."""
from collections.abc import Sequence
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from bagelquant_core import Domain
from bagelquant_core.domain import _inclusive_date_count
from bagelquant_core.incremental import _calendar_sessions


def gap_domain():
    days = [date(2020, 1, 2), date(2020, 1, 6)]
    rows = [(date(2020, 1, 1), "off-left", True),
        (date(2020, 1, 3), "gap-only-资产", True),
        (date(2020, 1, 6), "A", True),
        (date(2020, 1, 7), "off-right", True),
        (date(2020, 1, 6), "inactive", False)]
    return Domain(calendar=days, universe=pl.DataFrame(rows, orient="row",
        schema={"time": pl.Date, "asset_id": pl.String, "active": pl.Boolean}))


@pytest.mark.parametrize("first,last,keys,sessions", [
    ("2020-01-02", "2020-01-06", 2, 2),
    ("2020-01-03", "2020-01-03", 1, 0),
    ("2020-01-01", "2020-01-07", 4, 2),
    ("2020-01-06", "2020-01-06", 1, 1),
    ("2019-12-01", "2019-12-31", 0, 0),
    ("2020-02-01", "2020-02-02", 0, 0),
    ("2020-01-07", "2020-01-01", 0, 0),
])
def test_off_calendar_keys_are_not_calendar_rectangles(first, last, keys, sessions):
    domain = gap_domain()
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    old = domain.grid_lazy().filter(pl.col("time").is_between(start, end)).collect()
    assert domain._count_rows(start, end) == old.height == keys
    assert _inclusive_date_count(domain.times, start, end) == sessions
    if sessions:
        window = domain.window(start, end)
        assert _calendar_sessions(domain.times.to_list(), start, end) == len(window.times) == sessions
        assert window.size == keys
    else:
        for operation in (lambda: domain.window(start, end),
                lambda: _calendar_sessions(domain.times.to_list(), start, end)):
            with pytest.raises(ValueError, match="calendar must contain at least one time"):
                operation()


@pytest.mark.parametrize("dynamic", [False, True, "empty"])
@pytest.mark.parametrize("start,end", [
    (date(1969, 12, 31), date(2020, 3, 2)),
    (date(2020, 2, 29), date(2020, 2, 29)),
    (date(1970, 1, 1), date(2020, 2, 28)),
    (date(2020, 3, 1), date(2020, 3, 1)),
])
def test_inclusive_duplicates_static_empty_and_leap_dates(dynamic, start, end):
    days = [date(1969, 12, 31), date(2020, 2, 28), date(2020, 2, 29), date(2020, 3, 2)]
    assets = ["A", "B", "Unicode-资产"]
    if dynamic is False:
        universe = assets
    else:
        universe = pl.DataFrame({"time": days}).join(
            pl.DataFrame({"asset_id": assets}), how="cross").with_columns(
                pl.lit(dynamic != "empty").alias("active"))
    domain = Domain(calendar=days, universe=universe)
    expected = domain.grid_lazy().filter(pl.col("time").is_between(start, end)).collect().height
    assert domain._count_rows(start, end) == expected
    if start == end == date(2020, 2, 29):
        assert expected == (0 if dynamic == "empty" else 3)
    assert _inclusive_date_count(domain.times, days[0], days[-1]) == 4


def test_multichunk_scalar_lookup_needs_no_full_column_operation(monkeypatch):
    days = [date(1969, 12, 31), date(2020, 2, 29), date(2020, 2, 29), date(2020, 3, 2)]
    axis = pl.concat([pl.Series("time", days[:2], dtype=pl.Date),
        pl.Series("time", days[2:], dtype=pl.Date)], rechunk=False)
    assert axis.n_chunks() == 2
    def forbidden(*_args, **_kwargs):
        pytest.fail("scalar count requested a full column operation")
    for name in ("filter", "search_sorted", "to_numpy", "to_list", "rechunk", "sort"):
        monkeypatch.setattr(pl.Series, name, forbidden)
    assert _inclusive_date_count(axis, days[1], days[1]) == 2
    assert axis.n_chunks() == 2


@pytest.mark.parametrize("length", [0, 1, 2, 3, 32, 1001, 1_048_576])
def test_scalar_visits_are_logarithmic_without_an_allocated_axis(length):
    class Dates(Sequence):
        def __init__(self):
            self.visits = 0
        def __len__(self):
            return length
        def __getitem__(self, index):
            assert type(index) is int and 0 <= index < length
            self.visits += 1
            return date(1900, 1, 1) + timedelta(days=index // 3)
        def __iter__(self):
            pytest.fail("count iterated the axis")
    axis = Dates()
    assert _inclusive_date_count(axis, date(1900, 1, 11), date(1900, 1, 21)) == (
        min(length, 63) - min(length, 30))
    assert axis.visits <= 2 * length.bit_length()
    before = axis.visits
    assert _inclusive_date_count(axis, date(1900, 1, 2), date(1900, 1, 1)) == 0
    assert axis.visits == before
    assert vars(axis) == {"visits": axis.visits}


@pytest.mark.parametrize("invalid", [None, "2020-01-02", datetime(2020, 1, 2)])
def test_private_bounds_do_not_coerce_broader_public_inputs(invalid):
    with pytest.raises(TypeError, match="datetime.date bounds"):
        _inclusive_date_count([date(2020, 1, 2)], invalid, date(2020, 1, 6))
    # The original public path explicitly supports datetime comparisons.
    assert gap_domain().window(datetime(2020, 1, 2), datetime(2020, 1, 6)).size == 2


def test_counts_do_not_hash_construct_scopes_or_retain_new_state(monkeypatch):
    domain = gap_domain()
    before = {name: id(value) for name, value in vars(domain).items()}
    signature, original = domain.signature, pl.Series.item
    visits = []
    def item(axis, index=None):
        assert index is not None
        visits.append(index)
        return original(axis, index)
    def forbidden(*_args, **_kwargs):
        pytest.fail("count constructed, hashed or collected a Domain")
    monkeypatch.setattr(pl.Series, "item", item)
    for name in ("window", "grid_lazy", "_set_signature", "__init__"):
        monkeypatch.setattr(Domain, name, forbidden)
    for name in ("filter", "to_list", "to_numpy", "rechunk", "search_sorted"):
        monkeypatch.setattr(pl.Series, name, forbidden)
    for name in ("filter", "select", "lazy", "unique", "sort"):
        monkeypatch.setattr(pl.DataFrame, name, forbidden)
    assert domain._count_rows(date(2020, 1, 2), date(2020, 1, 6)) == 2
    assert len(visits) <= 2 * domain.size.bit_length()
    assert domain.signature == signature
    assert {name: id(value) for name, value in vars(domain).items()} == before


@pytest.mark.parametrize("error", [MemoryError("scalar"), KeyboardInterrupt()])
def test_scalar_errors_propagate_once_and_release_owned_borrow_aliases(monkeypatch, error):
    domain = gap_domain()
    calls = []
    def broken(_axis, _index=None):
        calls.append(1)
        raise error
    monkeypatch.setattr(pl.Series, "item", broken)
    with pytest.raises(type(error)) as captured:
        domain._count_rows(date(2020, 1, 2), date(2020, 1, 6))
    assert captured.value is error and calls == [1]
    trace = captured.value.__traceback__
    locals_by_name = {}
    while trace is not None:
        locals_by_name[trace.tb_frame.f_code.co_name] = trace.tb_frame.f_locals
        trace = trace.tb_next
    assert locals_by_name["_count_rows"]["coordinates"] is None
    assert locals_by_name["_inclusive_date_count"]["item"] is None
    assert "values" not in locals_by_name["_inclusive_date_count"]
    # Foreign caller/getter traceback frames are left intact.
    assert locals_by_name["broken"]["_axis"] is not None


def test_equal_endpoints_counts_or_opaque_identity_never_prove_membership():
    first = gap_domain()
    other = Domain(calendar=first.times, universe=["A"], identity=first.signature)
    assert other.signature == first.signature
    assert first._count_rows(first.start_time, first.end_time) == other.size == 2
    assert not first.grid_lazy().collect().equals(other.grid_lazy().collect())
    assert first.window(first.start_time, first.end_time).signature != other.window(
        other.start_time, other.end_time).signature


def test_inactive_duplicates_are_still_invalid_before_counting():
    day = date(2020, 1, 2)
    members = pl.DataFrame({"time": [day, day], "asset_id": ["A", "A"], "active": [True, False]})
    with pytest.raises(ValueError, match="unique by"):
        Domain(calendar=[day], universe=members)
