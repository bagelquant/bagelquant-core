"""Narrow numerical reads preserve canonical proofs and exact saved outputs."""
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, FiniteKernelResult, FiniteKernelWork, Node, compute_finite_work
import bagelquant_core.incremental as incremental
from bagelquant_core.operator import lag, pct_change, rankpct, rolling_mean, rolling_std, rolling_sum, smooth, sub
from bagelquant_core.resources import ResourceLimits
from test_global_graph import finish, global_graph


def source(count, *, case="retired"):
    dates = [date(2024, 1, 1) + timedelta(days=i + (i // 5 * 2 if case == "irregular" else 0))
        for i in range(count)]
    members = []
    for index, day in enumerate(dates):
        if (case != "empty" or index < 510) and (case != "sparse" or index % 5 == 0):
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


def expression(operation, value):
    if operation == "smooth":
        return smooth(value)
    return {"pct_change": pct_change, "lag": lag}[operation](value, periods=240)


@pytest.mark.parametrize("operation", ["pct_change", "smooth", "lag"])
@pytest.mark.parametrize("case", ["retired", "reentry", "listing", "empty", "partial_exit", "irregular", "sparse"])
def test_append_keeps_old_blocks_and_exact_canonical_outputs(tmp_path, monkeypatch, operation, case):
    before, current = source(510, case=case), source(512, case=case)
    def factory(value):
        return expression(operation, value)
    graph, store, root = global_graph(tmp_path / "append", factory(before))
    with monkeypatch.context() as legacy:
        legacy.setattr(incremental, "_numerical_read_start", canonical_read_start)
        legacy.setattr(incremental, "_EXACT_WINDOW_OPERATORS",
            incremental._EXACT_WINDOW_OPERATORS - {"bagelquant_core.operator.temporal.lag"})
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
        legacy.setattr(incremental, "_EXACT_WINDOW_OPERATORS",
            incremental._EXACT_WINDOW_OPERATORS - {"bagelquant_core.operator.temporal.lag"})
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
    expected_sessions = (506 if operation == "smooth" else 512) if case == "reentry" and operation != "lag" else (
        11 if operation == "smooth" else 242)
    if case == "sparse" and operation != "lag":
        expected_sessions = 47 if operation == "smooth" else 512
    assert usage["kernel_sessions"] == usage["input_sessions"] == expected_sessions
    assert windows[-1] == (current.domain.times[-expected_sessions], current.domain.times.max())
    assert incremental._canonical_halos(current.domain, 9 if operation == "smooth" else 240)[480] == current.domain.times.min()


@pytest.mark.parametrize("operation", ["pct_change", "smooth", "lag"])
def test_initialization_matches_original_canonical_values_and_traces(tmp_path, monkeypatch, operation):
    current = source(512, case="mixed")
    def factory(value):
        return expression(operation, value)
    reference, old_store, old_root = global_graph(tmp_path / "reference", factory(current))
    with monkeypatch.context() as legacy:
        legacy.setattr(incremental, "_numerical_read_start", canonical_read_start)
        legacy.setattr(incremental, "_EXACT_WINDOW_OPERATORS",
            incremental._EXACT_WINDOW_OPERATORS - {"bagelquant_core.operator.temporal.lag"})
        old = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max()),
            limits=ResourceLimits(cache_mib=0))
    graph, store, root = global_graph(tmp_path / "new", factory(current))
    result = finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    assert result["results"]["main"][root] == old["results"]["main"][old_root]
    assert_frame_equal(store.read(result["results"]["main"][root]).panel.collect(dense=False, include_traces=True),
        old_store.read(old["results"]["main"][old_root]).panel.collect(dense=False, include_traces=True), check_exact=True)


def test_calendar_lag_on_sparse_membership_bounds_every_cold_kernel(tmp_path, monkeypatch):
    current = source(1024, case="sparse")
    with incremental._finite_time_index_reservation(current.domain, 240,
            "bagelquant_core.operator.temporal.lag", ResourceLimits()) as lease:
        assert not lease.admitted and lease.reserved_bytes == 0 and lease.index is None
    results = []
    for name, original in (("original", True), ("calendar", False)):
        graph, store, root = global_graph(tmp_path / name, lag(current, periods=240))
        plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max())
        with monkeypatch.context() as selected:
            if original:
                selected.setattr(incremental, "_EXACT_WINDOW_OPERATORS",
                    incremental._EXACT_WINDOW_OPERATORS - {"bagelquant_core.operator.temporal.lag"})
            saved = finish(plan)
        identity = saved["results"]["main"][root]
        frame = store.read(identity).panel.collect(dense=False, include_traces=True)
        assert frame.height > 100
        results.append((identity, frame, plan.resource_usage["main"][root]))
    assert results[0][0] == results[1][0]
    assert_frame_equal(results[0][1], results[1][1], check_exact=True)
    assert results[0][1]["value"].drop_nulls().to_numpy().tobytes() == results[1][1]["value"].drop_nulls().to_numpy().tobytes()
    assert results[1][2]["computed_output_sessions"] == 1024
    assert results[1][2]["kernel_sessions"] <= 32 * (32 + 240)
    assert results[1][2]["kernel_sessions"] < results[0][2]["kernel_sessions"] // 2


@pytest.mark.parametrize("value_type", ["numeric", "category", "weights", "prediction"])
def test_calendar_lag_typed_staged_work_matches_original_sparse_keys(tmp_path, monkeypatch, value_type):
    current = source(70, case="sparse")
    frame = current.collect(dense=False, include_traces=True)
    if value_type == "category":
        frame = frame.with_columns(pl.col("value").cast(pl.String))
    current = Node.from_domain(frame, current.domain, source_key="x",
        value_type=value_type, trace_columns=current.trace_columns)
    outputs, stages = [], []

    def dispatch(work):
        restored = FiniteKernelWork.from_dict(work.to_dict())
        assert restored == work
        assert all(len(item.calendar) <= incremental.CANONICAL_SESSIONS + 5 for item in work.inputs)
        stages.append(work.staging_root)
        return FiniteKernelResult.from_dict(compute_finite_work(restored).to_dict())

    for name in ("original", "inline", "staged"):
        graph, store, root = global_graph(tmp_path / name, lag(current, periods=5))
        with monkeypatch.context() as selected:
            if name == "original":
                selected.setattr(incremental, "_EXACT_WINDOW_OPERATORS",
                    incremental._EXACT_WINDOW_OPERATORS - {"bagelquant_core.operator.temporal.lag"})
            plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
                kernel_dispatch=dispatch if name == "staged" else None)
            saved = finish(plan)
        identity = saved["results"]["main"][root]
        outputs.append((identity, store.read_values(identity, include_traces=True)))
    assert stages and all(not Path(stage).exists() for stage in stages)
    for identity, frame in outputs[1:]:
        assert identity == outputs[0][0]
        assert_frame_equal(frame, outputs[0][1], check_exact=True)
    assert outputs[0][1].height > 0


def test_nested_lag_rank_preserves_bits_repairs_and_historical_references(tmp_path, monkeypatch):
    def observed(count, revision=False):
        value = source(count, case="mixed")
        frame = value.collect(dense=False, include_traces=True).with_columns(
            pl.when(pl.col("time").dt.ordinal_day() % 19 == 0)
            .then(-0.0).otherwise(pl.col("value")).alias("value"),
            pl.col("time").alias("observed_at"))
        frame = pl.concat((frame, frame.filter(pl.col("asset_id") == "A").with_columns(
            pl.lit("B").alias("asset_id"), (pl.col("value") * 1.3 + 0.75).alias("value"))))
        membership = value.domain.grid_lazy().collect()
        domain = Domain(calendar=value.domain.times, universe=pl.concat((membership,
            membership.filter(pl.col("asset_id") == "A").with_columns(pl.lit("B").alias("asset_id"))))
            .with_columns(pl.lit(True).alias("active")))
        # Implicit holes differ from explicit null observations.
        frame = frame.filter(pl.col("time").dt.ordinal_day() % 13 != 0)
        if revision:
            # Both direct240 and composed240+60 effects fit in the saved range.
            frame = frame.with_columns(pl.when((pl.col("time") == value.domain.times[200])
                & (pl.col("asset_id") == "A"))
                .then(8.25).otherwise(pl.col("value")).alias("value"))
        return Node.from_domain(frame, domain, source_key="x",
            trace_columns=("available_date", "observed_at"))

    def formula(value):
        growth = sub(value, lag(value, periods=240))
        acceleration = sub(growth, lag(growth, periods=60))
        return rankpct(acceleration)

    snapshots = (observed(510), observed(514), observed(514, True))
    outputs, work, identities = [], [], []
    for label in ("canonical", "narrow"):
        with monkeypatch.context() as patch:
            if label == "canonical":
                patch.setattr(incremental, "_EXACT_WINDOW_OPERATORS",
                    incremental._EXACT_WINDOW_OPERATORS - {"bagelquant_core.operator.temporal.lag"})
            graph, store, root = global_graph(tmp_path / label, formula(snapshots[0]))
            frames, usages, saved = [], [], []
            for value in snapshots:
                plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
                result = finish(plan, limits=ResourceLimits(cache_mib=16))
                identity = result["results"]["main"][root]
                frames.append({node: store.read_values(saved_id, include_traces=True)
                    for node, saved_id in result["results"]["main"].items()})
                usages.append(plan.resource_usage["main"])
                saved.append((identity, store.manifest(identity)))
            assert all(store.manifest(identity) == manifest for identity, manifest in saved)
            repeated = graph.plan_update({"main": {"x": snapshots[-1]}}, through=snapshots[-1].domain.times.max())
            assert repeated.complete
            assert finish(repeated)["results"] == result["results"]
            outputs.append(frames)
            work.append(usages)
            identities.append([identity for identity, _ in saved])
    assert identities[0] == identities[1]
    for actual_nodes, expected_nodes in zip(outputs[1], outputs[0], strict=True):
        assert actual_nodes.keys() == expected_nodes.keys()
        for node, actual in actual_nodes.items():
            expected = expected_nodes[node]
            assert_frame_equal(actual, expected, check_exact=True)
            assert actual["value"].drop_nulls().to_numpy().tobytes() == expected["value"].drop_nulls().to_numpy().tobytes()
    for current, original in zip(work[1], work[0], strict=True):
        assert current.keys() == original.keys()
        for node, usage in current.items():
            for name in ("computed_blocks", "reused_blocks", "computed_output_sessions"):
                assert usage[name] == original[node][name]
            assert usage["kernel_sessions"] <= original[node]["kernel_sessions"]
    assert any(item["reused_blocks"] for item in work[1][-1].values())


def test_interrupted_lag_reuses_published_blocks_on_retry(tmp_path, monkeypatch):
    from contextlib import contextmanager

    value = source(130, case="irregular")
    graph, store, root = global_graph(tmp_path / "retry", lag(value, periods=5))
    kernel = incremental._finite_kernel
    calls = []

    @contextmanager
    def interrupted(*args, **kwargs):
        calls.append(args[6:8])
        if len(calls) == 3:
            raise RuntimeError("interrupted lag")
        with kernel(*args, **kwargs) as result:
            yield result

    with monkeypatch.context() as patch:
        patch.setattr(incremental, "_finite_kernel", interrupted)
        plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
        with pytest.raises(RuntimeError, match="interrupted lag"):
            finish(plan)
    assert not plan._in_flight
    retry = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    result = finish(retry)
    assert retry.resource_usage["main"][root]["reused_blocks"] == 2
    assert retry.resource_usage["main"][root]["computed_output_sessions"] == 66
    reference, old_store, old_root = global_graph(tmp_path / "reference", lag(value, periods=5))
    with monkeypatch.context() as patch:
        patch.setattr(incremental, "_EXACT_WINDOW_OPERATORS",
            incremental._EXACT_WINDOW_OPERATORS - {"bagelquant_core.operator.temporal.lag"})
        expected = finish(reference.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    assert result["results"] == expected["results"]
    actual = store.read_values(result["results"]["main"][root], include_traces=True)
    original = old_store.read_values(expected["results"]["main"][old_root], include_traces=True)
    assert_frame_equal(actual, original, check_exact=True)
    assert actual["value"].drop_nulls().to_numpy().tobytes() == original["value"].drop_nulls().to_numpy().tobytes()


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
        "bagelquant_core.operator.basic.pct_change", "bagelquant_core.operator.rolling_stats.smooth",
        "bagelquant_core.operator.temporal.lag"}
