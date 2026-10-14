"""Independent Prediction fits reuse original evidence under bounded updates."""
from datetime import date, timedelta
from dataclasses import replace

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import (
    Domain, Node, PredictionTrainingContext, ICWeightedPredictionOperator,
    ICWeightedDecayPredictionOperator, QuantileICWeightedPredictionOperator,
    OLSPredictionOperator, GLSPredictionOperator, RevisionConflict,
)
from bagelquant_core.operator import prediction as prediction_module
from bagelquant_core.operator import OPERATOR_REGISTRY
from test_global_graph import global_graph, finish


FACTORIES = [ICWeightedPredictionOperator, ICWeightedDecayPredictionOperator,
             QuantileICWeightedPredictionOperator, OLSPredictionOperator, GLSPredictionOperator]


def _inputs(count, *, change=None, delay=1, empty=(), removed_date=None, receipt=None):
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(count)]
    dates = [day for index, day in enumerate(dates) if index != removed_date]
    members = [(day, str(asset)) for day in dates if (day - date(2024, 1, 1)).days not in empty
               for asset in range(8)
               if change != "membership" or day != date(2024, 1, 16) or asset != 2]
    membership = pl.DataFrame(members, schema={"time": pl.Date, "asset_id": pl.String}, orient="row")
    domain = Domain(calendar=dates, universe=membership.with_columns(pl.lit(True).alias("active")))
    frames = {key: [] for key in ("x", "z", "targets", "availability")}
    for day, asset in members:
        t, a = (day - date(2024, 1, 1)).days, int(asset)
        x = a + (t % 7) / 11
        z = (a * a % 11) / 5 + (t % 5) / 9
        target = 0.7 * x + 0.2 * z + ((a * 13 + t * 7) % 11) / 17
        available = day + timedelta(days=delay)
        if t == 15:
            if change == "target":
                target += (a % 3) / 2
            if change == "feature":
                x += (a % 3) / 3
            if change == "availability":
                available += timedelta(days=10)
        for key, value in {"x": x, "z": z, "targets": target,
                           "availability": float(available.toordinal())}.items():
            trace = day + timedelta(days=2) if change == "trace" and key == "x" and t == 15 else day
            if change == "null" and key == "x" and t == 15 and a == 2:
                value = None
            frames[key].append((day, asset, value, trace))
    return {key: Node.from_domain(pl.DataFrame(rows,
        schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64, "available_date": pl.Date},
        orient="row"), domain, source_key=key, identity=None if receipt is None else receipt + key,
        trace_columns=("available_date",)) for key, rows in frames.items()}


def _expression(factory, inputs):
    kwargs = {"window": 3}
    if factory is ICWeightedDecayPredictionOperator:
        kwargs["half_life"] = 2
    if factory is QuantileICWeightedPredictionOperator:
        kwargs["quantiles"] = 3
    return factory(**kwargs)(inputs["x"], inputs["z"],
        training=PredictionTrainingContext(inputs["targets"], inputs["availability"]))


def _finish(graph, values, *, windows=None, input_proof=None, **execute):
    def whole(*_):
        pytest.fail("Prediction update read full historical inputs")

    def window(_context, key, start, end):
        if windows is not None:
            windows.append((key, start, end))
        value = values[key]
        return Node.from_domain(value.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
            value.domain.window(start, end), identity=value.identity, value_type=value.value_type,
            trace_columns=value.trace_columns, trace_identity=value.trace_identity)

    plan = graph.plan_update({"main": values}, through=values["x"].domain.times.max(),
        input_loader=whole, input_window_loader=window, input_proof=input_proof)
    return plan, finish(plan, **execute)


def _fit_spy(monkeypatch, factory):
    periods = []
    name = "_cross_sectional_regression_records" if factory in {OLSPredictionOperator, GLSPredictionOperator} else "_rank_records"
    original = getattr(prediction_module, name)

    def record(data, *args, **kwargs):
        periods.extend(data["time"].unique().sort().to_list())
        return original(data, *args, **kwargs)

    monkeypatch.setattr(prediction_module, name, record)
    return periods


def _assert_reference(tmp_path, factory, current, store, root, updated):
    graph, full, reference_root = global_graph(tmp_path / "reference", _expression(factory, current))
    _, expected = _finish(graph, current)
    identity, reference = updated["results"]["main"][root], expected["results"]["main"][reference_root]
    assert identity == reference
    assert_frame_equal(store.read_values(identity, include_traces=True),
                       full.read_values(reference, include_traces=True), check_exact=True)
    assert store.read_checkpoint(identity) == full.read_checkpoint(reference)
    assert not store.evidence(identity, channels=("training_audits",))["training_audits"]
    return identity


@pytest.mark.parametrize("factory", FACTORIES)
def test_append_reuses_all_old_fits_and_files(tmp_path, monkeypatch, factory):
    first, current = _inputs(40), _inputs(41)
    graph, store, root = global_graph(tmp_path / "incremental", _expression(factory, first))
    _, old = _finish(graph, first)
    original = store.manifest(old["results"]["main"][root])
    files = {part["$file"]: store._file(part["$file"]).read_bytes() for part in original["partitions"]}
    fits = _fit_spy(monkeypatch, factory)
    windows = []
    plan, updated = _finish(graph, current, windows=windows)
    assert fits == [date(2024, 2, 9)]  # Only the formerly pending final period matured.
    assert {day for _, start, end in windows for day in (start, end)} == {date(2024, 2, 9), date(2024, 2, 10)}
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 1
    assert plan.resource_usage["main"][root]["checkpoint_through"] == "2024-02-09"
    identity = _assert_reference(tmp_path, factory, current, store, root, updated)
    assert files.keys() <= {part["$file"] for part in store.manifest(identity)["partitions"]}
    assert all(store._file(path).read_bytes() == content for path, content in files.items())
    fits.clear()
    windows.clear()
    _, repeated = _finish(graph, current, windows=windows)
    assert repeated["results"] == updated["results"] and not fits and not windows


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("change", ["target", "feature", "availability", "membership", "null", "trace"])
def test_revision_repairs_only_one_period_fit_and_dependent_predictions(tmp_path, monkeypatch, factory, change):
    first, current = _inputs(40), _inputs(40, change=change)
    graph, store, root = global_graph(tmp_path / "incremental", _expression(factory, first))
    _, old = _finish(graph, first)
    original_id = old["results"]["main"][root]
    original = store.manifest(original_id)
    fits = _fit_spy(monkeypatch, factory)
    windows = []
    plan, updated = _finish(graph, current, windows=windows)
    assert fits == [date(2024, 1, 16)]
    assert all(date(2024, 1, 16) <= start <= end <= date(2024, 1, 28) for _, start, end in windows)
    if change == "target":
        assert plan.resource_usage["main"][root]["output_periods"] == ["2024-01-17", "2024-01-18", "2024-01-19"]
    identity = _assert_reference(tmp_path, factory, current, store, root, updated)
    assert store.manifest(original_id) == original
    assert set(part["$file"] for part in original["partitions"]) <= set(part["$file"] for part in store.manifest(identity)["partitions"])


@pytest.mark.parametrize("factory", FACTORIES)
def test_arbitrary_delays_keep_pending_samples_and_load_only_newly_mature_periods(tmp_path, monkeypatch, factory):
    first, current = _inputs(70, delay=50), _inputs(74, delay=50)
    graph, store, root = global_graph(tmp_path / "incremental", _expression(factory, first))
    _, before = _finish(graph, first)
    fits = _fit_spy(monkeypatch, factory)
    windows = []
    plan, updated = _finish(graph, current, windows=windows)
    assert fits == [date(2024, 1, 21) + timedelta(days=index) for index in range(4)]
    expected = set(fits) | set(current["x"].domain.times.to_list()[-4:])
    assert set(plan.resource_usage["main"][root]["input_periods"]) == {str(day) for day in expected}
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 4
    assert all(start in expected and end in expected for _, start, end in windows)
    _assert_reference(tmp_path, factory, current, store, root, updated)
    assert store.read_checkpoint(before["results"]["main"][root])["state"]["records"]["2024-01-21"]["status"] == "pending"


@pytest.mark.parametrize("empty", [(0, 15, 39, 40), tuple(range(41))])
def test_empty_universe_days_keep_complete_period_axis(tmp_path, empty):
    factory = OLSPredictionOperator
    first, current = _inputs(40, empty=empty), _inputs(41, empty=empty)
    graph, store, root = global_graph(tmp_path / "incremental", _expression(factory, first))
    _finish(graph, first)
    plan, updated = _finish(graph, current)
    identity = _assert_reference(tmp_path, factory, current, store, root, updated)
    assert store.read_checkpoint(identity)["state"]["calendar"] == [str(day) for day in current["x"].domain.times]
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 1


def test_complete_calendar_revision_repairs_windows_without_old_fits(tmp_path, monkeypatch):
    factory = OLSPredictionOperator
    first, current = _inputs(40), _inputs(40, removed_date=15)
    graph, store, root = global_graph(tmp_path / "incremental", _expression(factory, first))
    _finish(graph, first)
    fits = _fit_spy(monkeypatch, factory)
    plan, updated = _finish(graph, current)
    assert fits == []
    assert plan.resource_usage["main"][root]["output_periods"] == ["2024-01-17", "2024-01-18", "2024-01-19"]
    _assert_reference(tmp_path, factory, current, store, root, updated)


def test_identical_values_with_new_enclosing_receipts_are_metadata_only(tmp_path, monkeypatch):
    first, current = _inputs(40), _inputs(40, receipt="new-receipt-")
    graph, store, root = global_graph(tmp_path, _expression(OLSPredictionOperator, first))
    def proof(_context, key, start, end):
        return first[key].interval_identity(start, end)
    _finish(graph, first, input_proof=proof)
    for key in first:
        assert_frame_equal(first[key].collect(include_traces=True), current[key].collect(include_traces=True))
    monkeypatch.setattr(prediction_module, "_cross_sectional_regression_records", lambda *_a, **_k: pytest.fail("noop fit old periods"))
    windows = []
    plan, updated = _finish(graph, current, windows=windows, input_proof=proof)
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 0
    assert not windows
    assert store.read_checkpoint(updated["results"]["main"][root])["state"]["through"] == "2024-02-09"


def test_unknown_proof_and_missing_original_state_fail_before_reads_or_fits(tmp_path, monkeypatch):
    first, current = _inputs(40), _inputs(41)
    graph, store, root = global_graph(tmp_path, _expression(OLSPredictionOperator, first))
    _, old = _finish(graph, first)
    monkeypatch.setattr(prediction_module, "_cross_sectional_regression_records", lambda *_a, **_k: pytest.fail("unknown evidence refit"))
    plan = graph.plan_update({"main": current}, through=current["x"].domain.times.max(),
        input_proof=lambda *_: None, input_window_loader=lambda *_: pytest.fail("unknown evidence read values"))
    with pytest.raises(ValueError, match="interval-proof maintenance"):
        finish(plan)
    assert store.graph_state("global")["current_update"] == old["identity"]
    original = store.read_checkpoint
    monkeypatch.setattr(store, "read_checkpoint", lambda identity: {**original(identity), "state": {"through": "2024-02-09"}})
    with pytest.raises(ValueError, match="original per-period statistical evidence"):
        _finish(graph, current)
    assert store.graph_state("global")["current_update"] == old["identity"]


def test_canceled_after_statistics_reuses_fits_on_retry_and_preserves_completion(tmp_path, monkeypatch):
    first, current = _inputs(40), _inputs(41)
    graph, store, root = global_graph(tmp_path, _expression(OLSPredictionOperator, first))
    _, old = _finish(graph, first)
    fits = _fit_spy(monkeypatch, OLSPredictionOperator)
    publish = store.publish

    def interrupt(value, **kwargs):
        publish(value, **kwargs)
        if kwargs.get("execution_policy", {}).get("role") == "statistics":
            raise RuntimeError("interrupted after committed statistics")

    with monkeypatch.context() as patch:
        patch.setattr(store, "publish", interrupt)
        with pytest.raises(RuntimeError, match="committed statistics"):
            _finish(graph, current)
    assert fits == [date(2024, 2, 9)]
    assert store.graph_state("global")["current_update"] == old["identity"]
    fits.clear()
    _finish(graph, current)
    assert fits == []


def test_publication_conflict_keeps_numerical_cache_and_original_evidence(tmp_path, monkeypatch):
    first, current = _inputs(40), _inputs(41)
    graph, store, root = global_graph(tmp_path, _expression(OLSPredictionOperator, first))
    _finish(graph, first)
    one = graph.plan_update({"main": current}, through=current["x"].domain.times.max())
    two = graph.plan_update({"main": current}, through=current["x"].domain.times.max())
    saved = finish(one)
    fits = _fit_spy(monkeypatch, OLSPredictionOperator)
    with pytest.raises(RevisionConflict):
        finish(two)
    assert fits == []
    assert store.graph_state("global")["current_update"] == saved["identity"]


def test_proof_window_plan_declares_actual_period_scopes_without_callbacks(tmp_path):
    values = _inputs(8, empty=(3,))
    graph, _store, _root = global_graph(tmp_path, _expression(OLSPredictionOperator, values))
    plan = graph.plan_update({"main": values}, through=values["x"].domain.times.max(),
        input_proof=lambda *_: pytest.fail("planning invoked a source proof"),
        input_loader=lambda *_: pytest.fail("planning loaded values"))
    windows = plan.input_proof_windows("main")
    assert windows.height == 8 * 4
    assert windows["strategy"].unique().to_list() == ["checkpoint_periods"]
    assert windows["status"].unique().to_list() == ["required"]
    assert windows["start"].equals(windows["end"])
    assert windows["consumption_cutoff"].null_count() == windows.height


def test_legacy_output_without_original_fits_requires_explicit_evidence(tmp_path, monkeypatch):
    first, current = _inputs(40), _inputs(41)
    graph, store, _root = global_graph(tmp_path, _expression(OLSPredictionOperator, first))
    operation = OPERATOR_REGISTRY.get("prediction:ols")
    with monkeypatch.context() as patch:
        patch.setattr(operation, "contract", replace(operation.contract, checkpoint_replay=False, checkpoint_periods=None))
        old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    monkeypatch.setattr(prediction_module, "_cross_sectional_regression_records", lambda *_a, **_k: pytest.fail("legacy maintenance refit old periods"))
    with pytest.raises(ValueError, match="original per-period statistical evidence"):
        _finish(graph, current)
    assert store.graph_state("global")["current_update"] == old["identity"]


def test_all_history_proof_change_never_refits_completed_history(tmp_path, monkeypatch):
    first = _inputs(40)
    graph, store, _root = global_graph(tmp_path, _expression(OLSPredictionOperator, first))
    _, old = _finish(graph, first)
    current = {}
    for key, value in first.items():
        frame = value.collect(include_traces=True).with_columns((pl.col("value") + 1).alias("value"))
        current[key] = Node.from_domain(frame, value.domain, source_key=key, trace_columns=value.trace_columns)
    monkeypatch.setattr(prediction_module, "_cross_sectional_regression_records", lambda *_a, **_k: pytest.fail("all-history change refit periods"))
    with pytest.raises(ValueError, match="all completed prediction history"):
        _finish(graph, current)
    assert store.graph_state("global")["current_update"] == old["identity"]
