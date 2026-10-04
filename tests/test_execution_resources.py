from datetime import date, timedelta
from threading import Barrier, Lock

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import (
    CategoryPanel, Domain, ExecutionMode, ExecutionRuntime, Graph, MaterializationLookup,
    MaterializationStatus, OPERATOR_REGISTRY, OperationContract, Panel, TraceRule,
    capture_operator_checkpoints, capture_training_audits, causal_history_requirements,
    project_domain, materialization_trace_identity,
)
from bagelquant_core.operator_state import save_operator_artifact, save_operator_state, save_training_audit
from bagelquant_core.prediction import IdentityPredictionComposer
from bagelquant_core.resources import ResourceLimits, active_resource_limits, current_resident_mib, resource_limits
from bagelquant_core.transformer import identity, negate, pct_change, rolling_rank, rolling_ols, smooth, zscore
from bagelquant_core.transformer.core import transformer


@pytest.fixture(autouse=True)
def isolate_extensions(monkeypatch):
    monkeypatch.setattr(OPERATOR_REGISTRY, "_items", dict(OPERATOR_REGISTRY._items))


def source():
    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(12)]
    domain = Domain(calendar=days, universe=["A", "B"])
    frame = domain.grid_lazy().collect().with_columns(
        pl.Series("value", [float(index + 1) for index in range(domain.size)]),
        pl.col("time").alias("available_date"))
    return Panel.from_domain(frame, domain, source_key="quotes", identity="quotes.v1",
        trace_columns=("available_date",), trace_identity="timing.v1")


class MemoryStore:
    def __init__(self):
        self.records = {}

    def query(self, key):
        value = self.records.get(key.identity)
        return MaterializationLookup(MaterializationStatus.MISS if value is None else MaterializationStatus.HIT, value)

    def publish(self, value):
        self.records[value.key.identity] = value


def test_frontier_runs_independent_kernels_and_merges_evidence_with_one_budget():
    barrier, lock = Barrier(2), Lock()
    budgets = []

    @transformer(contract=OperationContract(execution=ExecutionMode.EAGER_BARRIER, trace_rule=TraceRule.PASSTHROUGH))
    def kernel(frame: pl.DataFrame, *, scale: int):
        limits = active_resource_limits()
        with lock:
            budgets.append(limits)
        barrier.wait(timeout=5)
        save_operator_artifact("scale", scale)
        save_training_audit({"scale": scale})
        save_operator_state({"through": "2024-01-13", "scale": scale})
        return frame.with_columns((pl.col("value") * scale).alias("value"))

    input_ = source()
    graph = Graph(outputs=[kernel(input_, scale=2, name="two"), kernel(input_, scale=3, name="three")])
    runtime = ExecutionRuntime()
    with resource_limits(ResourceLimits(total_threads=4, parallel_nodes=2, lightgbm_threads=4,
            histogram_pool_mib=128)), capture_operator_checkpoints(calendar=input_.domain.times) as states, \
            capture_training_audits() as audits:
        result = graph.compute(runtime=runtime)
    assert len(budgets) == 2
    assert sum(item.total_threads for item in budgets) == 4
    assert all(item.lightgbm_threads == 2 and item.histogram_pool_mib == 64 for item in budgets)
    assert runtime.resource_usage["max_parallel_nodes"] == 2
    assert len(states.captured) == len(audits) == len(runtime.node_artifacts) == 2
    assert [item["scale"] for item in audits] == [2, 3]
    for name, scale in [("two", 2), ("three", 3)]:
        assert_frame_equal(result[name].collect(), input_.collect().with_columns((pl.col("value") * scale).alias("value")))


def test_parallel_runtime_matches_serial_and_exact_materialization_keys():
    input_ = source()
    graph = Graph(outputs=[rolling_rank(input_, window=3, name="first"),
        rolling_rank(input_, window=5, name="second"), negate(rolling_rank(input_, window=3), name="downstream")])
    full = graph.compute(runtime=ExecutionRuntime())
    store = MemoryStore()
    runtime = ExecutionRuntime(materialization_store=store)
    expected = runtime.plan_materialization_keys(graph)
    with resource_limits(ResourceLimits(total_threads=4, parallel_nodes=2)):
        actual = graph.compute(runtime=runtime)
    assert expected == {node_id: record.key for node_id, record in runtime.node_materializations.items()}
    for name in full:
        assert_frame_equal(actual[name].collect(include_traces=True), full[name].collect(include_traces=True))
    assert runtime.resource_usage["max_parallel_nodes"] == 2
    assert all(record.panel.trace_identity == materialization_trace_identity(record.key, record.panel.trace_columns)
        for record in runtime.node_materializations.values())


def test_identity_planner_never_executes_and_preserves_typed_auxiliary_domains(monkeypatch):
    input_ = source()
    category = CategoryPanel.from_domain(input_.collect(include_traces=True).with_columns(pl.lit("group").alias("value")),
        input_.domain, source_key="groups", identity="groups.v1", trace_columns=("available_date",))
    members = Panel.from_domain(input_.collect().filter(pl.col("asset_id") == "A"),
        Domain(calendar=input_.domain.times, universe=["A"]), source_key="members", identity="members.v1")
    forecast = IdentityPredictionComposer().compose(input_)
    graph = Graph(outputs=[identity(category, name="groups"), project_domain(zscore(forecast), membership=members, name="forecast")])
    runtime = ExecutionRuntime(materialization_store=MemoryStore())
    with monkeypatch.context() as patch:
        patch.setattr(pl.LazyFrame, "collect", lambda *args, **kwargs: pytest.fail("planner collected payload"))
        expected = runtime.plan_materialization_keys(graph)
        domains = dict(runtime.node_domains)
    graph.compute(runtime=runtime, dense_output=False)
    assert expected == {node_id: record.key for node_id, record in runtime.node_materializations.items()}
    assert all(domains[node_id].equivalent_to(record.panel.domain) for node_id, record in runtime.node_materializations.items())


def test_memory_pressure_reduces_new_admission_and_releases_output_cache(monkeypatch):
    monkeypatch.setattr("bagelquant_core.execution.current_resident_mib", lambda: 200.0)
    input_ = source()
    graph = Graph(outputs=[rolling_rank(input_, window=3, name="a"), rolling_rank(input_, window=5, name="b")])
    runtime = ExecutionRuntime()
    with resource_limits(ResourceLimits(total_threads=4, parallel_nodes=2, memory_target_mib=100, cache_mib=10, batch_rows=64)):
        graph.compute(runtime=runtime)
    assert runtime.resource_usage["pressure_events"] >= 1
    assert runtime._effective_limits.parallel_nodes == 1
    assert runtime._effective_limits.batch_rows <= 32
    assert runtime.cache == {}
    assert runtime.resource_usage["peak_resident_mib"] == 200.0


def test_finite_history_composes_halos_and_unknown_operations_stay_conservative():
    input_ = source()
    graph = smooth(zscore(pct_change(input_, periods=240)))
    spec = graph.logical_spec()
    histories = causal_history_requirements(spec)
    assert histories[next(iter(spec.outputs.values()))] == 249

    @transformer
    def unknown(frame: pl.DataFrame):
        return frame

    spec = zscore(unknown(input_)).logical_spec()
    assert causal_history_requirements(spec)[next(iter(spec.outputs.values()))] is None


def test_resident_memory_is_available_and_worker_budgets_do_not_expand_small_batches():
    assert current_resident_mib() > 0
    limits = ResourceLimits(total_threads=8, parallel_nodes=4, memory_target_mib=100,
        cache_mib=10, histogram_pool_mib=24, batch_rows=1, lightgbm_threads=8)
    worker = limits.for_workers(4)
    assert worker.total_threads == worker.lightgbm_threads == 2
    assert worker.histogram_pool_mib == 6
    assert worker.batch_rows == worker.under_pressure(100).batch_rows == 1


def test_proven_node_binding_cuts_upstream_execution_and_keeps_logical_definition():
    calls = []

    @transformer(contract=OperationContract(execution=ExecutionMode.EAGER_BARRIER, trace_rule=TraceRule.PASSTHROUGH))
    def counted(frame: pl.DataFrame):
        calls.append(1)
        return frame

    input_ = source()
    original = negate(counted(input_))
    spec = original.logical_spec()
    upstream = next(node for node in spec.nodes if node.operator == counted.registry_name)
    bound = Graph.from_logical_spec(spec, inputs={}, node_bindings={upstream.node_id: input_})
    actual = bound.compute()
    assert calls == []
    assert bound.logical_spec().to_dict() == spec.to_dict()
    assert len(bound.nodes) == 2
    assert_frame_equal(actual.collect(), input_.collect().with_columns((-pl.col("value")).alias("value")))
    with pytest.raises(ValueError, match="operator nodes"):
        Graph.from_logical_spec(spec, inputs={}, node_bindings={"unknown": input_})
    category = CategoryPanel.from_domain(input_.collect().with_columns(pl.lit("x").alias("value")), input_.domain)
    with pytest.raises(ValueError, match="requires panel"):
        Graph.from_logical_spec(spec, inputs={}, node_bindings={upstream.node_id: category})


def test_small_temporary_batches_preserve_rank_and_regression_values():
    base = source()
    random = np.random.default_rng(1729)
    frame = base.collect(include_traces=True)
    input_ = Panel.from_domain(frame.with_columns(pl.Series("value", random.normal(size=len(frame)))),
        base.domain, source_key="feature", identity="feature.v1", trace_columns=base.trace_columns)
    target = Panel.from_domain(frame.with_columns(pl.Series("value", random.normal(size=len(frame)))),
        base.domain, source_key="target", identity="target.v1", trace_columns=base.trace_columns)
    graph = Graph(outputs=[rolling_rank(input_, window=4, name="rank"),
        rolling_ols(target, factors=(input_,), window=4, name="ols")])
    reference = graph.compute(runtime=ExecutionRuntime())
    with resource_limits(ResourceLimits(total_threads=4, parallel_nodes=2, batch_rows=1)):
        actual = graph.compute(runtime=ExecutionRuntime())
    for name in reference:
        assert_frame_equal(actual[name].collect(), reference[name].collect())


def test_node_contexts_isolate_finite_ancestors_and_propagate_their_exact_parent_keys():
    @transformer(contract=OperationContract(execution=ExecutionMode.EAGER_BARRIER, trace_rule=TraceRule.PASSTHROUGH))
    def unknown(frame: pl.DataFrame):
        return frame.with_columns((pl.col("value") + 1).alias("value"))

    input_ = source()
    graph = negate(unknown(rolling_rank(input_, window=3)))
    specification = graph.logical_spec()
    finite = next(node for node in specification.nodes if node.operator == rolling_rank.registry_name)
    unknown_node = next(node for node in specification.nodes if node.operator == unknown.registry_name)
    plain = ExecutionRuntime(context_identity="default").plan_materialization_keys(graph)
    contexts = {finite.node_id: "finite_causal_blocks.v1"}
    store = MemoryStore()
    runtime = ExecutionRuntime(materialization_store=store, context_identity="default", node_contexts=contexts)
    contexts[finite.node_id] = "mutated-after-construction"
    planned = runtime.plan_materialization_keys(graph)
    graph.compute(runtime=runtime, dense_output=False)
    assert planned == {node_id: record.key for node_id, record in runtime.node_materializations.items()}
    assert planned[finite.node_id].context_identity != plain[finite.node_id].context_identity
    assert planned[unknown_node.node_id].context_identity == plain[unknown_node.node_id].context_identity
    assert planned[unknown_node.node_id].inputs == (planned[finite.node_id].identity,)
    assert planned[unknown_node.node_id].identity != plain[unknown_node.node_id].identity
    root = next(iter(specification.outputs.values()))
    assert planned[root].inputs == (planned[unknown_node.node_id].identity,)
    assert planned[root].identity != plain[root].identity
    repeated = ExecutionRuntime(materialization_store=store, context_identity="default",
        node_contexts={finite.node_id: "finite_causal_blocks.v1"})
    graph.compute(runtime=repeated, dense_output=False)
    assert repeated.persistent_misses == 0
    assert repeated.persistent_hits == len(planned)
