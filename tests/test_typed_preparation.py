"""Typed preparation keeps canonical coordinates and trace row alignment."""
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, Node
from bagelquant_core.frame import normalize_date_expression, normalize_panel_frame
from bagelquant_core.incremental import scoped_domain


@pytest.mark.parametrize("dynamic", [False, True])
def test_windows_match_public_construction_with_opaque_source_identity(dynamic):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    universe = ["A", "B", "C"]
    if dynamic:
        universe = pl.DataFrame({"time": [days[0], days[0], days[2], days[5]],
            "asset_id": ["A", "B", "C", "A"], "active": [True] * 4})
    domain = Domain(calendar=days, universe=universe, identity="opaque-source")
    for first, last in ((days[0], days[-1]), (days[2], days[4]), (days[3], days[4]),
                        (days[5], days[5]), (days[0] - timedelta(days=1), days[0])):
        dates = domain.times.filter(domain.times.is_between(first, last))
        expected_universe = (domain.grid_lazy().filter(pl.col("time").is_between(first, last))
            .collect().with_columns(pl.lit(True).alias("active"))) if dynamic else domain.asset_ids
        expected = Domain(calendar=dates, universe=expected_universe)
        actual = domain.window(first, last)
        assert actual.signature == expected.signature
        assert actual.signature != domain.signature
        assert actual.times.equals(expected.times)
        assert actual.asset_ids.equals(expected.asset_ids)
        assert actual.size == expected.size
        assert (actual.start_time, actual.end_time) == (expected.start_time, expected.end_time)
        assert_frame_equal(actual.grid_lazy().collect(), expected.grid_lazy().collect(), check_exact=True)
    with pytest.raises(ValueError, match="calendar must contain at least one time"):
        domain.window(days[-1] + timedelta(days=1), days[-1] + timedelta(days=2))


@pytest.mark.parametrize("dynamic", [False, True])
def test_finite_scope_selects_validated_coordinates_without_reconstructing(monkeypatch, dynamic):
    days = [date(2024, 1, 1), date(2024, 1, 2)]
    universe = (pl.DataFrame({"time": days, "asset_id": ["A", "B"], "active": [True, True]})
        if dynamic else ["A", "B"])
    domain = Domain(calendar=days, universe=universe)
    expected = domain.window(days[0], days[0])
    def forbidden(*_args, **_kwargs):
        pytest.fail("finite execution reconstructed already validated Domain coordinates")
    with monkeypatch.context() as patch:
        patch.setattr(Domain, "__init__", forbidden)
        actual = scoped_domain(domain, days[0], days[0])
    assert actual.signature == expected.signature
    assert_frame_equal(actual.grid_lazy().collect(), expected.grid_lazy().collect(), check_exact=True)
    explicit = scoped_domain(domain, days[0], days[0], identity="explicit")
    assert explicit.signature == "explicit"


def _original_trace_normalization(frame, *, numeric):
    traces = ("available_date", "observed_at")
    base = normalize_panel_frame(frame.select("time", "asset_id", "value"), numeric=numeric)
    evidence = frame.select("time", "asset_id", *traces).with_columns(
        normalize_date_expression("time", frame.schema["time"]), pl.col("asset_id").cast(pl.String))
    return base.join(evidence, on=["time", "asset_id"], how="left").sort("time", "asset_id")


@pytest.mark.parametrize("time_kind", ["date", "string", "datetime"])
@pytest.mark.parametrize("category", [False, True])
def test_trace_validation_preserves_original_row_alignment_without_key_joins(monkeypatch, time_kind, category):
    days = [date(2024, 1, 1), date(2024, 1, 2)]
    times = [days[1], days[0], days[1]]
    if time_kind == "string":
        times = [str(day) for day in times]
    elif time_kind == "datetime":
        times = [datetime.combine(day, datetime.min.time()) for day in times]
    values = ["late", None, "early"] if category else [float("nan"), float("inf"), -0.0]
    frame = pl.DataFrame({"time": times, "asset_id": [2, 2, 1], "value": values,
        "available_date": [days[1], None, days[0]], "observed_at": [None, days[0], days[1]]})
    expected = _original_trace_normalization(frame, numeric=not category)
    def forbidden(*_args, **_kwargs):
        pytest.fail("row-aligned traces were separated and joined again")
    monkeypatch.setattr(pl.DataFrame, "join", forbidden)
    panel = Node.from_domain(frame, Domain(calendar=days, universe=[1, 2]),
        value_type="category" if category else "numeric", identity="original",
        trace_columns=("available_date", "observed_at"))
    assert_frame_equal(panel.collect(dense=False, include_traces=True), expected, check_exact=True)


@pytest.mark.parametrize("invalid", ["null", "duplicate", "coerced_duplicate"])
def test_trace_normalization_keeps_key_validation(invalid):
    days = [date(2024, 1, 1), date(2024, 1, 2)]
    keys = [str(day) for day in days]
    if invalid == "null":
        keys[0] = "not-a-date"
    elif invalid == "duplicate":
        keys[1] = keys[0]
    else:
        keys = [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2)]
    frame = pl.DataFrame({"time": keys, "asset_id": ["A", "A"], "value": [1., 2.],
        "available_date": days})
    with pytest.raises(ValueError, match="keys must be valid|unique"):
        Node.from_domain(frame, Domain(calendar=days, universe=["A"]), trace_columns=("available_date",))
