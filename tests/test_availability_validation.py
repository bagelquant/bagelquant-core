"""Availability follows numerical dependencies; lazy inputs fail at consumption."""
from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import CoreStore, Domain, ExecutionRuntime, Graph, Node, OPERATOR_REGISTRY
from bagelquant_core.operation_contract import TraceRule
from bagelquant_core.operator import (
    anscombe, demean, equal_weight, group_mean, group_rank, inv_log_sqrt_rank,
    kelly_nonan_standardize, kelly_rank_boxcox, log_rank, mask, min_max_scale,
    net_scale, normalize, nrank, operator, orthogonalize, rank, rankpct, regularized_weights, top_n,
    translate_to_pos, trim_quantile, winsorize, zscore,
)
from test_global_graph import finish, global_graph


def source(count=1, *, lazy=False, revision=None):
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(count)]
    rows = [(day, asset, float(index + 1), day + timedelta(days=index),
             day + timedelta(days=2-index))
            for day in days for index, asset in enumerate(("A", "B", "C"))]
    frame = pl.DataFrame(rows, orient="row", schema={"time": pl.Date,
        "asset_id": pl.String, "value": pl.Float64, "available": pl.Date,
        "observed": pl.Date})
    if revision is not None:
        frame = frame.with_columns(pl.when(pl.col("time") == days[revision])
            .then(pl.col("available") + pl.duration(days=5))
            .otherwise(pl.col("available")).alias("available"))
    return Node.from_domain(frame.lazy() if lazy else frame,
        Domain(calendar=days, universe=["A", "B", "C"]), source_key="x",
        trace_columns=("available", "observed"))


@pytest.mark.parametrize("operation", [rank, rankpct, nrank, zscore, demean,
    min_max_scale, normalize, net_scale, winsorize, trim_quantile, translate_to_pos,
    log_rank, inv_log_sqrt_rank, anscombe])
def test_cross_section_traces_include_later_peer_and_every_channel(operation):
    node = source()
    output = operation(node).compute().collect(include_traces=True)
    assert output["available"].to_list() == [date(2024, 1, 3)] * 3
    assert output["observed"].to_list() == [date(2024, 1, 3)] * 3


def test_grouped_traces_scope_values_to_group_and_include_assignment_evidence():
    node = source()
    groups = Node.from_domain(pl.DataFrame({"time": [date(2024, 1, 1)] * 3,
        "asset_id": ["A", "B", "C"], "value": ["one", "one", "two"],
        "classification": [date(2024, 1, 1)] * 3}), node.domain,
        value_type="category", trace_columns=("classification",))
    for operation in (group_mean, group_rank):
        output = operation(node, group=groups).compute().collect(include_traces=True)
        assert output["available"].to_list() == [date(2024, 1, 2)] * 2 + [date(2024, 1, 3)]
        assert output["classification"].to_list() == [date(2024, 1, 1)] * 3
    revised = groups.collect(include_traces=True).with_columns(
        pl.when(pl.col("asset_id") == "C").then(date(2024, 1, 7))
        .otherwise(pl.col("classification")).alias("classification"))
    classified = Node.from_domain(revised, node.domain, value_type="category",
        trace_columns=("classification",))
    output = group_mean(node, group=classified).compute().collect(include_traces=True)
    assert output["classification"].to_list() == [date(2024, 1, 7)] * 3


def test_cross_section_auxiliary_traces_preserve_disjoint_datetime_channels():
    from datetime import datetime
    node = source()
    original = node.collect().with_columns(pl.lit(datetime(2024, 1, 10, 12)).alias("known"))
    factor = node.collect().with_columns(pl.lit(datetime(2024, 1, 20, 8)).alias("observed"))
    left = Node.from_domain(original, node.domain, trace_columns=("known",))
    right = Node.from_domain(factor, node.domain, trace_columns=("observed",))
    actual = orthogonalize(left, factors=(right,)).compute().collect(include_traces=True)
    assert actual["known"].to_list() == [datetime(2024, 1, 10, 12)] * 3
    assert actual["observed"].to_list() == [datetime(2024, 1, 20, 8)] * 3
    assert actual.schema["known"] == actual.schema["observed"] == pl.Datetime("us")


@pytest.mark.parametrize("grouped", [False, True])
def test_shared_trace_channels_keep_finest_datetime_unit_before_aggregation(grouped):
    node = source()
    nano = pl.Datetime("ns", "UTC")
    micro = pl.Datetime("us", "UTC")
    stamp = 1735689600000000033
    left = Node.from_domain(node.collect().with_columns(
        pl.Series("known", [stamp] * 3, dtype=pl.Int64).cast(nano)), node.domain,
        trace_columns=("known",))
    frame = node.collect().with_columns(pl.Series("known", [stamp // 1000] * 3, dtype=pl.Int64).cast(micro))
    if grouped:
        frame = frame.with_columns(pl.lit("same").alias("value"))
    right = Node.from_domain(frame, node.domain, value_type="category" if grouped else "numeric", trace_columns=("known",))
    expression = group_mean(left, group=right) if grouped else orthogonalize(left, factors=(right,))
    actual = expression.compute().collect(include_traces=True)
    assert actual.schema["known"] == nano
    assert actual["known"].cast(pl.Int64).to_list() == [stamp] * 3


@pytest.mark.parametrize("grouped", [False, True])
def test_shared_trace_channels_require_explicit_timezone_alignment(grouped):
    node = source()
    left = Node.from_domain(node.collect().with_columns(
        pl.Series("known", [1735689600000000] * 3, dtype=pl.Int64).cast(pl.Datetime("us", "UTC"))),
        node.domain, trace_columns=("known",))
    frame = node.collect().with_columns(
        pl.Series("known", [1735689600000000] * 3, dtype=pl.Int64).cast(pl.Datetime("us", "Asia/Shanghai")))
    if grouped:
        frame = frame.with_columns(pl.lit("same").alias("value"))
    right = Node.from_domain(frame, node.domain, value_type="category" if grouped else "numeric", trace_columns=("known",))
    expression = group_mean(left, group=right) if grouped else orthogonalize(left, factors=(right,))
    with pytest.raises(ValueError, match="matching time zones"):
        expression.compute()


@pytest.mark.parametrize("operation", [kelly_nonan_standardize, kelly_rank_boxcox])
def test_composed_cross_section_and_rolling_traces_include_both_scopes(operation):
    node = source(3)
    output = operation(node, window=2).compute().collect(include_traces=True)
    expected = [day + timedelta(days=2) for day in node.domain.times for _ in range(3)]
    assert output["available"].to_list() == expected


@pytest.mark.parametrize("operation", [kelly_nonan_standardize, kelly_rank_boxcox])
def test_cross_section_rolling_traces_preserve_sparse_reentry_window(operation):
    rows = [(date(2024, 1, 1), "A", 1., date(2024, 1, 30)),
            (date(2024, 1, 1), "B", 2., date(2024, 1, 1)),
            (date(2024, 1, 2), "B", 3., date(2024, 1, 20)),
            (date(2024, 1, 3), "A", 2., date(2024, 1, 3)),
            (date(2024, 1, 3), "B", 3., date(2024, 1, 3))]
    frame = pl.DataFrame(rows, orient="row", schema={"time": pl.Date,
        "asset_id": pl.String, "value": pl.Float64, "available": pl.Date})
    domain = Domain(calendar=[date(2024, 1, day) for day in (1, 2, 3)],
        universe=frame.select("time", "asset_id").with_columns(pl.lit(True).alias("active")))
    node = Node.from_domain(frame, domain, trace_columns=("available",))
    output = operation(node, window=2).compute().collect(include_traces=True)
    assert output.filter(pl.col("time") == date(2024, 1, 3))["available"].to_list() == [date(2024, 1, 30), date(2024, 1, 20)]


@pytest.mark.parametrize("operation", [kelly_nonan_standardize, kelly_rank_boxcox])
@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_composite_rolling_traces_preserve_datetime_dtype_and_precision(operation, unit):
    from datetime import datetime
    node = source(3)
    dtype = pl.Datetime(unit)
    base = pl.Series([datetime(2025, 1, 1)]).cast(dtype).cast(pl.Int64).item()
    raw = [base + 30 - day * 10 + asset for day in range(3) for asset in range(3)]
    frame = node.collect(include_traces=True).with_columns(pl.Series("available", raw).cast(dtype))
    typed = Node.from_domain(frame, node.domain, trace_columns=node.trace_columns)
    output = operation(typed, window=2).compute().collect(include_traces=True)
    assert output.schema["available"] == dtype
    assert output["available"].cast(pl.Int64).to_list() == [base + 32] * 6 + [base + 22] * 3


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_saved_cumulative_trace_checkpoint_preserves_datetime_precision(tmp_path, unit):
    from datetime import datetime
    node = source(3)
    dtype = pl.Datetime(unit)
    base = pl.Series([datetime(2025, 1, 1)]).cast(dtype).cast(pl.Int64).item()
    frame = node.collect(include_traces=True).with_columns(
        pl.Series("available", [base + 17, base + 18, base + 19] + [None] * 6, dtype=pl.Int64).cast(dtype))
    current = Node.from_domain(frame, node.domain, source_key="x", trace_columns=node.trace_columns)
    first = Node.from_domain(frame.filter(pl.col("time") <= date(2024, 1, 2)),
        node.domain.window(date(2024, 1, 1), date(2024, 1, 2)), source_key="x", trace_columns=node.trace_columns)
    graph, store, root = global_graph(tmp_path / "incremental", regularized_weights(first, max_weight=0.6))
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.end_time))
    updated = finish(graph.plan_update({"main": {"x": current}}, through=current.domain.end_time))
    actual = store.read_values(updated["results"]["main"][root], include_traces=True)
    fresh, fresh_store, fresh_root = global_graph(tmp_path / "fresh", regularized_weights(current, max_weight=0.6))
    reference = finish(fresh.plan_update({"main": {"x": current}}, through=current.domain.end_time))
    assert_frame_equal(actual, fresh_store.read_values(reference["results"]["main"][fresh_root], include_traces=True))
    assert actual.schema["available"] == dtype
    assert actual["available"].cast(pl.Int64).to_list() == [base + 19] * 9


def test_cumulative_trace_preserves_an_entirely_unknown_channel():
    node = source(2)
    frame = node.collect(include_traces=True).with_columns(pl.lit(None).alias("available"))
    unknown = Node.from_domain(frame, node.domain, trace_columns=node.trace_columns)
    actual = regularized_weights(unknown, max_weight=0.6).compute().collect(include_traces=True)
    assert actual.schema["available"] == pl.Null
    assert actual["available"].null_count() == 6


def test_portfolio_selection_and_equal_weight_traces_include_peers():
    node = source()
    for output in (top_n(node, count=1), equal_weight(top_n(node, count=1))):
        actual = output.compute().collect(include_traces=True)
        assert actual["available"].to_list() == [date(2024, 1, 3)] * 3


@pytest.mark.parametrize("category", [False, True])
def test_default_mask_has_portable_null_semantics_in_python_dsl_and_storage(tmp_path, category):
    node = source()
    if category:
        node = Node.from_domain(node.collect(include_traces=True).with_columns(
            pl.Series("value", ["one", "two", "three"])), node.domain,
            source_key="x", value_type="category", trace_columns=node.trace_columns)
    selector = Node.from_domain(node.collect().with_columns(pl.Series("value", [1., 0., None])),
        node.domain, source_key="selector")
    python = mask(node, mask_frame=selector)
    dsl = Graph.from_dsl("output = mask(x, mask_frame=selector)", inputs={"x": node, "selector": selector})
    restored = Graph.from_logical_spec(dsl.logical_spec(), inputs={"x": node, "selector": selector})
    expected = ["one" if category else 1., None, None]
    assert python.logical_id == mask(node, mask_frame=selector, replace_value=None).logical_id
    for expression in (python, dsl, restored):
        result = expression.compute().collect(include_traces=True)
        assert result["value"].to_list() == expected
    graph, store, root = global_graph(tmp_path, python)
    receipt = finish(graph.plan_update({"main": {"x": node, "selector": selector}},
        through=node.domain.end_time))
    assert store.read_values(receipt["results"]["main"][root])["value"].to_list() == expected


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("invalid", ["date", "asset"])
@pytest.mark.parametrize("dense", [False, True])
def test_lazy_invalid_keys_survive_membership_and_dense_alignment(dynamic, invalid, dense):
    day = date(2024, 1, 1)
    membership = pl.DataFrame({"time": [day, day], "asset_id": ["A", "B"], "active": [True, True]})
    domain = Domain(calendar=[day], universe=membership if dynamic else ["A", "B"])
    frame = pl.DataFrame({"time": ["2024-01-01", "broken" if invalid == "date" else "2024-01-01"],
        "asset_id": ["A", None if invalid == "asset" else "B"], "value": [1., 2.]})
    with pytest.raises(ValueError, match="keys must be valid"):
        Node.from_domain(frame.lazy(), domain).collect(dense=dense)
    node = Node.from_domain(frame.lazy(), domain)
    with pytest.raises(ValueError, match="keys must be valid"):
        rank(node).compute()


@pytest.mark.parametrize("collision", [False, True])
def test_custom_reduction_cannot_hide_duplicate_consumed_lazy_keys(collision, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(OPERATOR_REGISTRY, "_items", dict(OPERATOR_REGISTRY._items))

    @operator
    def collapsed(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.group_by("time", "asset_id").agg(pl.col("value").mean())

    day = date(2024, 1, 1)
    times = ([datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2)] if collision else [day, day])
    frame = pl.DataFrame({"time": times, "asset_id": ["A", "A"], "value": [1., 3.]})
    domain = Domain(calendar=[day], universe=["A"])
    with pytest.raises(ValueError, match="unique by"):
        Node.from_domain(frame, domain)
    with pytest.raises(ValueError, match="unique by"):
        collapsed(Node.from_domain(frame.lazy(), domain)).compute()


@pytest.mark.parametrize("dynamic", [False, True])
def test_lazy_parquet_validation_keeps_bounded_scan(tmp_path, dynamic):
    node = source(96)
    path = tmp_path / "source.parquet"
    node.collect(include_traces=True).write_parquet(path, row_group_size=3)
    domain = node.domain.window(date(2024, 2, 2), date(2024, 2, 4))
    if dynamic:
        domain = Domain(calendar=domain.times, universe=domain.grid_lazy().with_columns(pl.lit(True).alias("active")).collect())
    consumed = []
    def observe(frame):
        consumed.append(frame.height)
        return frame
    bounded = pl.scan_parquet(path).map_batches(observe, predicate_pushdown=True, streamable=True)
    selected = Node.from_domain(bounded, domain,
        identity="saved", trace_columns=node.trace_columns)
    assert "SELECTION:" in selected.lazy(include_traces=True).explain()
    assert selected.collect(dense=False).height == 9
    assert rank(selected).compute().collect().height == 9
    assert consumed and max(consumed) <= 9


def test_corrected_trace_policy_does_not_reuse_or_rewrite_original_receipts(tmp_path, monkeypatch):
    node = source()
    expression = rank(node)
    logical_id = expression.logical_id
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "values")
    with monkeypatch.context() as patch:
        patch.setattr(rank, "contract", replace(rank.contract, trace_rule=TraceRule.PASSTHROUGH))
        old_runtime = ExecutionRuntime(materialization_store=store)
        expression.compute(runtime=old_runtime)
        old = old_runtime.node_materializations[logical_id]
    current_runtime = ExecutionRuntime(materialization_store=store)
    corrected = expression.compute(runtime=current_runtime).collect(include_traces=True)
    new = current_runtime.node_materializations[logical_id]
    assert expression.logical_id == logical_id
    assert new.key.implementation_id != old.key.implementation_id
    assert corrected["available"].to_list() == [date(2024, 1, 3)] * 3
    assert store.read_values(old.key.identity, include_traces=True)["available"].to_list() == [date(2024, 1, day) for day in (1, 2, 3)]
    assert store.describe(old.key.identity) is not None


def test_cross_section_saved_append_repair_and_metadata_only_noop(tmp_path, monkeypatch):
    first = source(65)
    graph, store, root = global_graph(tmp_path / "incremental", rank(first))
    old = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.end_time))
    old_identity = old["results"]["main"][root]
    for index, current in enumerate((source(75), source(75, revision=40))):
        original_run = ExecutionRuntime.run
        spans = []
        def counted(runtime, local, **kwargs):
            start = local._single_output().domain.start_time
            assert start >= (date(2024, 3, 6) if index == 0 else date(2024, 2, 2))
            spans.append(start)
            return original_run(runtime, local, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(ExecutionRuntime, "run", counted)
            plan = graph.plan_update({"main": {"x": current}}, through=current.domain.end_time)
            updated = finish(plan)
        actual = store.read_values(updated["results"]["main"][root], include_traces=True)
        reference_graph, reference_store, reference_root = global_graph(tmp_path / f"full-{index}", rank(current))
        reference = finish(reference_graph.plan_update({"main": {"x": current}}, through=current.domain.end_time))
        assert_frame_equal(actual, reference_store.read_values(reference["results"]["main"][reference_root], include_traces=True))
        assert spans and plan.resource_usage["main"][root]["reused_blocks"] >= 1
    assert store.describe(old_identity) is not None
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("no-op ran historical kernels"))
        patch.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("no-op read values"))
        lazy = current.lazy(include_traces=True).map_batches(
            lambda _frame: pytest.fail("no-op consumed lazy source values"))
        unchanged = Node.from_domain(lazy, current.domain, identity=current.identity,
            source_key="x", trace_columns=current.trace_columns, trace_identity=current.trace_identity)
        finish(graph.plan_update({"main": {"x": unchanged}}, through=current.domain.end_time))


def test_stateful_portfolio_append_carries_global_dependency_trace(tmp_path):
    node = source(35)
    frame = node.collect(include_traces=True).with_columns(
        pl.when(pl.col("time") == date(2024, 1, 1)).then(date(2025, 1, 1))
        .when(pl.col("time") == date(2024, 1, 2)).then(None)
        .otherwise(pl.col("available")).alias("available"))
    first = Node.from_domain(frame, node.domain, source_key="x", trace_columns=node.trace_columns)
    graph, store, root = global_graph(tmp_path / "incremental", regularized_weights(first, max_weight=0.6))
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.end_time))
    extended = source(40)
    current = Node.from_domain(pl.concat([frame, extended.collect(include_traces=True).filter(pl.col("time") > first.domain.end_time)]),
        extended.domain, source_key="x", trace_columns=node.trace_columns)
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.end_time)
    receipt = finish(plan)
    actual = store.read_values(receipt["results"]["main"][root], include_traces=True)
    assert actual["available"].to_list() == [date(2025, 1, 1)] * 120
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 5
    fresh_graph, fresh_store, fresh_root = global_graph(tmp_path / "fresh", regularized_weights(current, max_weight=0.6))
    reference = finish(fresh_graph.plan_update({"main": {"x": current}}, through=current.domain.end_time))
    assert_frame_equal(actual, fresh_store.read_values(reference["results"]["main"][fresh_root], include_traces=True))
