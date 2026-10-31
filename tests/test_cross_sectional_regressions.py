"""Stable residuals, lossless dependency dates and saved-result isolation."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import CoreStore, Domain, ExecutionRuntime, Node
from bagelquant_core.operator import (
    cross_sectional_residual,
    cross_sectional_weighted_mean,
    mad_zscore,
    orthogonalize,
)
from test_global_graph import finish, global_graph


OPERATIONS = (orthogonalize, mad_zscore, cross_sectional_residual, cross_sectional_weighted_mean)
ROBUST_OPERATIONS = OPERATIONS[1:]


def _source(values, *, domain=None, source_key="target", traces=None):
    day = date(2024, 1, 1)
    domain = domain or Domain(calendar=[day], universe=[str(i) for i in range(len(values))])
    frame = pl.DataFrame({"time": [day] * len(values),
        "asset_id": [str(i) for i in range(len(values))], "value": values},
        schema_overrides={"value": pl.Float64})
    if traces:
        frame = frame.with_columns(pl.Series(name, values) for name, values in traces.items())
    return Node.from_domain(frame, domain, source_key=source_key, trace_columns=tuple(traces or ()))


def _expression(operation, target, factor):
    if operation is orthogonalize:
        return operation(target, factors=(factor,), fit_intercept=True)
    if operation is mad_zscore:
        return operation(target)
    if operation is cross_sectional_residual:
        return operation(target, controls=(factor,))
    return operation(target, weights=factor)


@pytest.mark.parametrize("offset", [0.0, 1e8, 1e12])
def test_single_factor_residual_is_translation_invariant(offset):
    target = _source([1.0, 3.0, 5.0, 7.0])
    factor = _source([offset + i for i in range(4)], domain=target.domain, source_key="factor")
    # y = 2 * (x - offset) + 1 has exactly zero residuals.
    result = orthogonalize(target, factors=(factor,), fit_intercept=True).compute().collect()
    np.testing.assert_allclose(result["value"].to_numpy(), np.zeros(4), atol=1e-12, rtol=0)


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("invalid_parent", ["target", "factor"])
def test_single_factor_residual_excludes_nonfinite_peers(fit_intercept, invalid_parent):
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    y = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    selected = y if invalid_parent == "target" else x
    selected[4:] = [float("inf"), -float("inf"), float("nan"), None]
    target = _source(y)
    factor = _source(x, domain=target.domain, source_key="factor")
    result = orthogonalize(target, factors=(factor,), fit_intercept=fit_intercept).compute().collect()
    design = np.arange(4.0).reshape(-1, 1)
    if fit_intercept:
        design = np.column_stack([np.ones(4), design])
    finite_y = np.array([1.0, 3.0, 5.0, 7.0])
    expected = finite_y - design @ np.linalg.lstsq(design, finite_y, rcond=None)[0]
    np.testing.assert_allclose(result["value"].head(4).to_numpy(), expected, atol=1e-12, rtol=0)
    assert result["value"].tail(4).to_list() == [None] * 4


@pytest.mark.parametrize("x,y,fit_intercept,expected", [
    ([2.0, 2.0, 2.0], [1.0, 3.0, 5.0], True, [-2.0, 0.0, 2.0]),
    ([0.0, 0.0, 0.0], [1.0, 3.0, 5.0], False, [1.0, 3.0, 5.0]),
    ([1.0, None], [2.0, None], True, [None, None]),
    ([None, None], [2.0, None], False, [None, None]),
])
def test_single_factor_residual_retains_degenerate_sample_policy(x, y, fit_intercept, expected):
    target = _source(y)
    factor = _source(x, domain=target.domain, source_key="factor")
    actual = orthogonalize(target, factors=(factor,), fit_intercept=fit_intercept).compute().collect()
    assert actual["value"].to_list() == expected


@pytest.mark.parametrize("operation", [cross_sectional_residual, cross_sectional_weighted_mean])
def test_robust_traces_promote_dates_and_preserve_nanoseconds(operation):
    day = date(2024, 1, 1)
    midnight = 1_704_067_200_000_000_000
    precise = pl.Series([midnight + i for i in [1, 2, 3, 987]], dtype=pl.Int64).cast(pl.Datetime("ns"))
    coarse = pl.Series([midnight // 1_000_000] * 4, dtype=pl.Int64).cast(pl.Datetime("ms"))
    target = _source([1.0, 3.0, 4.0, 8.0], traces={"available": [day] * 4,
        "observed": precise, "target_only": [day] * 4})
    factor = _source([0.0, 1.0, 3.0, 4.0], domain=target.domain, source_key="factor",
        traces={"available": precise, "observed": coarse, "auxiliary_only": precise})
    output = _expression(operation, target, factor).compute().collect(include_traces=True)
    for channel in ("available", "observed", "auxiliary_only"):
        assert output.schema[channel] == pl.Datetime("ns")
        assert output[channel].cast(pl.Int64).to_list() == [midnight + 987] * 4
    assert output["target_only"].to_list() == [day] * 4


@pytest.mark.parametrize("operation", ROBUST_OPERATIONS)
def test_robust_traces_include_missing_peers_and_null_channels(operation):
    day = date(2024, 1, 1)
    later = day + timedelta(days=10)
    target = _source([1.0, 3.0, 4.0, None], traces={"available": [day, day, day, later],
        "unknown": [None] * 4})
    factor = _source([0.0, 1.0, 3.0, 4.0], domain=target.domain, source_key="factor")
    output = _expression(operation, target, factor).compute().collect(include_traces=True)
    assert output["available"].to_list() == [later] * 4
    assert output["unknown"].to_list() == [None] * 4


@pytest.mark.parametrize("operation", [cross_sectional_residual, cross_sectional_weighted_mean])
def test_robust_traces_reject_different_timezones(operation):
    epoch = pl.Series([1_704_067_200_000_000] * 4, dtype=pl.Int64)
    target = _source([1.0, 3.0, 4.0, 8.0],
        traces={"available": epoch.cast(pl.Datetime("us", "UTC"))})
    factor = _source([0.0, 1.0, 3.0, 4.0], domain=target.domain, source_key="factor",
        traces={"available": epoch.cast(pl.Datetime("us", "Asia/Shanghai"))})
    with pytest.raises(ValueError, match="matching time zones"):
        _expression(operation, target, factor).compute().collect(include_traces=True)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_corrected_operator_version_isolates_original_receipts(tmp_path, monkeypatch, operation):
    assert operation.version == "2"
    target = _source([1.0, 3.0, 4.0, 8.0])
    factor = _source([0.0, 1.0, 3.0, 4.0], domain=target.domain, source_key="factor")
    expression = _expression(operation, target, factor)
    logical_id = expression.logical_id
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "values")
    # Simulate an original-version receipt without maintaining another formula.
    with monkeypatch.context() as patch:
        patch.setattr(operation, "version", "1")
        original_runtime = ExecutionRuntime(materialization_store=store)
        original_values = expression.compute(runtime=original_runtime).collect()
        original = original_runtime.node_materializations[logical_id]
    runtime = ExecutionRuntime(materialization_store=store)
    expression.compute(runtime=runtime)
    corrected = runtime.node_materializations[logical_id]
    assert expression.logical_id == logical_id
    assert corrected.key.implementation_id != original.key.implementation_id
    assert_frame_equal(store.read_values(original.key.identity), original_values)
    repeat = ExecutionRuntime(materialization_store=store)
    expression.compute(runtime=repeat)
    assert repeat.persistent_hits >= 1
    assert repeat.eager_barriers == 0


def _history(count, *, revision=None):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]
    domain = Domain(calendar=days, universe=["a", "b", "c", "d"])
    frame = pl.DataFrame({"time": [day for day in days for _ in range(4)],
        "asset_id": ["a", "b", "c", "d"] * count,
        "value": [float(value) for _ in days for value in [1, 3, 4, 8]]})
    if revision is not None:
        frame = frame.with_columns(pl.when((pl.col("time") == days[revision]) & (pl.col("asset_id") == "a"))
            .then(2.0).otherwise(pl.col("value")).alias("value"))
    target = Node.from_domain(frame.with_columns(pl.col("time").alias("available")), domain,
        source_key="target", trace_columns=("available",))
    factor = Node.from_domain(frame.with_columns(pl.col("asset_id").replace_strict(
        {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}).alias("value")),
        domain, source_key="factor")
    return target, factor


@pytest.mark.parametrize("operation", OPERATIONS)
def test_corrected_cross_sections_preserve_append_repair_and_noop(tmp_path, monkeypatch, operation):
    target, factor = _history(64)
    graph, store, root = global_graph(tmp_path / "incremental", _expression(operation, target, factor))
    inputs = {"target": target, **({"factor": factor} if operation is not mad_zscore else {})}
    original = finish(graph.plan_update({"main": inputs}, through=target.domain.end_time))
    original_identity = original["results"]["main"][root]
    for index, revision in enumerate((None, 40)):
        target, factor = _history(65, revision=revision)
        inputs = {"target": target, **({"factor": factor} if operation is not mad_zscore else {})}
        starts = []
        run = ExecutionRuntime.run

        def observe(runtime, local, **kwargs):
            starts.append(local._single_output().domain.start_time)
            return run(runtime, local, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(ExecutionRuntime, "run", observe)
            plan = graph.plan_update({"main": inputs}, through=target.domain.end_time)
            updated = finish(plan)
        assert starts and min(starts) >= date(2024, 1, 1) + timedelta(days=64 if revision is None else 32)
        assert plan.resource_usage["main"][root]["reused_blocks"] >= 1
        assert plan.resource_usage["main"][root]["computed_output_sessions"] == (1 if revision is None else 32)
        actual = store.read_values(updated["results"]["main"][root], include_traces=True)
        reference_graph, reference_store, reference_root = global_graph(tmp_path / f"reference-{index}",
            _expression(operation, target, factor))
        reference = finish(reference_graph.plan_update({"main": inputs}, through=target.domain.end_time))
        assert_frame_equal(actual, reference_store.read_values(reference["results"]["main"][reference_root], include_traces=True), check_exact=True)
    assert store.describe(original_identity) is not None
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("no-op recalculated historical values"))
        patch.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("no-op read numerical values"))
        lazy_inputs = {key: Node.from_domain(node.lazy(include_traces=True).map_batches(
            lambda _frame: pytest.fail("no-op consumed source values")), node.domain,
            identity=node.identity, source_key=node.source_key, trace_columns=node.trace_columns,
            trace_identity=node.trace_identity) for key, node in inputs.items()}
        finish(graph.plan_update({"main": lazy_inputs}, through=target.domain.end_time))
