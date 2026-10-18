"""Narrow numerical reads preserve canonical proofs and exact saved outputs."""
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, Node
import bagelquant_core.incremental as incremental
from bagelquant_core.operator import pct_change, rolling_mean, rolling_std, rolling_sum, smooth
from bagelquant_core.resources import ResourceLimits
from test_global_graph import finish, global_graph


def source(count, *, case="retired"):
    dates = [date(2024, 1, 1) + timedelta(days=i + (i // 5 * 2 if case == "irregular" else 0))
        for i in range(count)]
    members = []
    for index, day in enumerate(dates):
        if case != "empty" or index < 510:
            members.append((day, "A"))
        if index < 4:
            members.append((day, "RETIRED"))
        if case in ("reentry", "mixed") and (index < 15 or index >= 510):
            members.append((day, "REENTRY"))
        if case in ("listing", "mixed") and index >= 510:
            members.append((day, "NEW"))
        if case in ("partial_exit", "mixed") and (index < 10 or 480 <= index < 510):
            members.append((day, "EXIT"))
    membership = pl.DataFrame(members, schema={"time": pl.Date, "asset_id": pl.String}, orient="row")
    domain = Domain(calendar=dates, universe=membership.with_columns(pl.lit(True).alias("active")))
    frame = membership.with_columns(
        pl.when(pl.col("time").dt.ordinal_day() % 17 == 0).then(None)
        .when(pl.col("time").dt.ordinal_day() % 23 == 0).then(float("nan"))
        .when(pl.col("time").dt.ordinal_day() % 29 == 0).then(0.0)
        .otherwise((pl.col("time").cast(pl.Int32) % 37).cast(pl.Float64) / 3 + 2).alias("value"),
        (pl.col("time") + timedelta(days=1)).alias("available_date"))
    return Node.from_domain(frame, domain, name="x", source_key="x", trace_columns=("available_date",))


def canonical_read_start(domain, first, _last, history):
    origin = (domain.times.to_list().index(first) // incremental.CANONICAL_SESSIONS
        * incremental.CANONICAL_SESSIONS)
    return incremental._canonical_halos(domain, history)[origin]


@pytest.mark.parametrize("operation", ["pct_change", "smooth"])
@pytest.mark.parametrize("case", ["retired", "reentry", "listing", "empty", "partial_exit", "irregular"])
def test_append_keeps_old_blocks_and_exact_canonical_outputs(tmp_path, monkeypatch, operation, case):
    before, current = source(510, case=case), source(512, case=case)
    factory = (lambda value: pct_change(value, periods=240)) if operation == "pct_change" else smooth
    graph, store, root = global_graph(tmp_path / "append", factory(before))
    with monkeypatch.context() as legacy:
        legacy.setattr(incremental, "_numerical_read_start", canonical_read_start)
        saved = finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max()),
            limits=ResourceLimits(cache_mib=0))
    previous_id = saved["results"]["main"][root]
    previous_manifest = store.manifest(previous_id)
    previous = store.read(previous_id).panel.collect(dense=False, include_traces=True)
    windows = []

    def loader(_context, _parent, first, last):
        windows.append((first, last))
        frame = current.lazy(include_traces=True).filter(pl.col("time").is_between(first, last)).collect()
        return Node.from_domain(frame, current.domain.window(first, last), identity=current.identity,
            value_type=current.value_type, trace_columns=current.trace_columns, trace_identity=current.trace_identity)

    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(), input_window_loader=loader)
    published = finish(plan)
    actual = store.read(published["results"]["main"][root]).panel.collect(dense=False, include_traces=True)
    reference, cold, cold_root = global_graph(tmp_path / "reference", factory(current))
    with monkeypatch.context() as legacy:
        legacy.setattr(incremental, "_numerical_read_start", canonical_read_start)
        completed = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max()),
            limits=ResourceLimits(cache_mib=0))
    expected = cold.read(completed["results"]["main"][cold_root]).panel.collect(dense=False, include_traces=True)
    assert_frame_equal(actual, expected, check_exact=True)
    assert_frame_equal(actual.filter(pl.col("time") <= before.domain.times.max()), previous, check_exact=True)
    assert store.manifest(previous_id) == previous_manifest
    assert published["results"]["main"][root] == completed["results"]["main"][cold_root]
    usage = plan.resource_usage["main"][root]
    assert usage["reused_blocks"] == 15 and usage["computed_blocks"] == 1
    assert usage["computed_output_sessions"] == 2
    expected_sessions = (512 if operation == "pct_change" else 506) if case == "reentry" else (
        242 if operation == "pct_change" else 11)
    assert usage["kernel_sessions"] == usage["input_sessions"] == expected_sessions
    assert windows[-1] == (current.domain.times[-expected_sessions], current.domain.times.max())
    assert incremental._canonical_halos(current.domain, 240 if operation == "pct_change" else 9)[480] == current.domain.times.min()


@pytest.mark.parametrize("operation", ["pct_change", "smooth"])
def test_initialization_matches_original_canonical_values_and_traces(tmp_path, monkeypatch, operation):
    current = source(512, case="mixed")
    factory = (lambda value: pct_change(value, periods=240)) if operation == "pct_change" else smooth
    reference, old_store, old_root = global_graph(tmp_path / "reference", factory(current))
    with monkeypatch.context() as legacy:
        legacy.setattr(incremental, "_numerical_read_start", canonical_read_start)
        old = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max()),
            limits=ResourceLimits(cache_mib=0))
    graph, store, root = global_graph(tmp_path / "new", factory(current))
    result = finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    assert result["results"]["main"][root] == old["results"]["main"][old_root]
    assert_frame_equal(store.read(result["results"]["main"][root]).panel.collect(dense=False, include_traces=True),
        old_store.read(old["results"]["main"][old_root]).panel.collect(dense=False, include_traces=True), check_exact=True)


@pytest.mark.parametrize("operation", [rolling_mean, rolling_std, rolling_sum])
def test_generic_rolling_keeps_original_rounding_state(tmp_path, operation):
    before, current = source(510), source(512)
    graph, _store, root = global_graph(tmp_path, operation(before, window=10))
    finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max()))
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max())
    finish(plan)
    assert plan.resource_usage["main"][root]["kernel_sessions"] == 512
    assert plan.resource_usage["main"][root]["computed_output_sessions"] == 2


@pytest.mark.parametrize("history", [0, 1, 9, 240])
@pytest.mark.parametrize("position", [0, 31, 32, 480, 510])
@pytest.mark.parametrize("dynamic", [False, True])
def test_read_start_matches_admitted_coordinates(history, position, dynamic):
    value = source(512, case="mixed")
    domain = value.domain if dynamic else Domain(calendar=value.domain.times, universe=["A", "B"])
    dates = domain.times.to_list()
    first, last = dates[position], dates[min(position + 1, len(dates) - 1)]
    expected = first if history == 0 else dates[max(0, position - history)]
    if history and dynamic:
        membership = domain.grid_lazy().collect()
        needed = set(membership.filter(pl.col("time").is_between(first, last))["asset_id"])
        for asset in needed:
            preceding = sorted(membership.filter((pl.col("asset_id") == asset) & (pl.col("time") < first))["time"])
            if preceding:
                expected = min(expected, preceding[-history:][0])
    assert incremental._numerical_read_start(domain, first, last, history) == expected


def test_empty_dynamic_membership_and_exact_builtin_gate():
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(512)]
    domain = Domain(calendar=dates, universe=pl.DataFrame(schema={"time": pl.Date, "asset_id": pl.String, "active": pl.Boolean}))
    assert incremental._numerical_read_start(domain, dates[510], dates[511], 9) == dates[501]
    assert incremental._EXACT_WINDOW_OPERATORS == {
        "bagelquant_core.operator.basic.pct_change", "bagelquant_core.operator.rolling_stats.smooth"}
