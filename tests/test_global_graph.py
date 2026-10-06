from datetime import date, timedelta
from dataclasses import replace

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import CoreStore, Domain, ExecutionRuntime, Graph, Node, NodeMaterialization, RevisionConflict
from bagelquant_core.operator import add, negate, rolling_mean, zscore
from bagelquant_core.resources import ResourceLimits


def source(count=70, *, revision=None, dynamic=False, name="x"):
    dates = [date(2024, 1, 1)+timedelta(days=i) for i in range(count)]
    frame = pl.DataFrame({"time": [t for t in dates for _ in range(2)], "asset_id": ["a", "b"]*count,
        "value": [float(i % 17) for i in range(count*2)]}).with_columns(pl.col("time").alias("available_date"))
    if revision is not None:
        frame = frame.with_columns(pl.when(pl.col("time") == dates[revision]).then(20.0).otherwise(pl.col("value")).alias("value"))
    membership = frame.select("time", "asset_id").filter(~((pl.col("asset_id") == "b") & (pl.col("time") < dates[min(count-1, 20)])))
    domain = Domain(calendar=dates, universe=membership.with_columns(pl.lit(True).alias("active")) if dynamic else ["a", "b"])
    return Node.from_domain(frame, domain, source_key=name, name=name, trace_columns=("available_date",))


def global_graph(tmp_path, expression):
    store = CoreStore(tmp_path/"core.sqlite", tmp_path/"values")
    graph = Graph(store=store, graph_id="global")
    merged = graph.merge(expression)
    graph.register_context("main", anchor="2024-01-01")
    return graph, store, next(iter(merged.outputs.values()))


def finish(plan, **kwargs):
    while not plan.complete:
        progressed = False
        for context in plan.results:
            for node_id in plan.ready(context):
                plan.execute(context, node_id, **kwargs)
                progressed = True
        assert progressed
    return plan.publish()


def test_standalone_dsl_merge_restart_exact_hit_and_branch(tmp_path, monkeypatch):
    x = source()
    local = Graph.from_dsl("a = rolling_mean(x, window=3)\noutput = zscore(a)", inputs={"x": x})
    graph, store, root = global_graph(tmp_path, local)
    receipt = finish(graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    count = len(store.inventory())
    restarted = Graph(store=CoreStore(store.meta_path, store.artifact_path), graph_id="global")
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("exact hit executed a kernel"))
        result = finish(restarted.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    assert result["results"] == receipt["results"]
    assert len(store.inventory()) == count
    extra = restarted.merge(negate(local._single_output()))
    plan = restarted.plan_update({"main": {"x": x}}, through=x.domain.times.max(), roots=list(extra.outputs.values()))
    branch = finish(plan)
    assert branch["mode"] == "branch"
    assert store.graph_state("global")["current_update"] == result["identity"]
    assert store.verify_update(receipt["identity"])["results"]["main"][root]


@pytest.mark.parametrize("dynamic", [False, True])
def test_append_and_revision_equal_canonical_full_build(tmp_path, dynamic):
    first = source(65, dynamic=dynamic)
    graph, store, root = global_graph(tmp_path/"incremental", zscore(rolling_mean(first, window=4)))
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    for index, value in enumerate((source(75, dynamic=dynamic), source(75, revision=40, dynamic=dynamic))):
        plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
        updated = finish(plan, limits=ResourceLimits(batch_rows=3))
        assert any(usage["reused_blocks"] for usage in plan.resource_usage["main"].values())
        full, fresh, full_root = global_graph(tmp_path/f"full-{index}", zscore(rolling_mean(value, window=4)))
        reference = finish(full.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
        actual = store.read(updated["results"]["main"][root]).panel.collect(include_traces=True)
        expected = fresh.read(reference["results"]["main"][full_root]).panel.collect(include_traces=True)
        assert_frame_equal(actual, expected, check_exact=True)


def test_sleep_cascades_across_contexts_and_wake_is_atomic(tmp_path):
    x = source(3)
    a = negate(x)
    b = add(a, x)
    graph, _, root = global_graph(tmp_path, b)
    graph.register_context("other", anchor="2024-01-01")
    bound = graph.bind({"x": x})
    plan = graph.plan_update({"main": {"x": x}, "other": {"x": x}}, through=x.domain.times.max())
    graph.sleep(a)
    assert graph.status(root) == "sleeping"
    for operation in (lambda: bound.compute(), lambda: bound._single_output().compute(),
                      lambda: graph.bind({"x": x}).compute(), lambda: graph.merge(negate(b)),
                      lambda: graph.resolve_local([root]).bind({"x": x}).compute(),
                      lambda: plan.ready("main")):
        with pytest.raises((ValueError, RevisionConflict)):
            operation()
    with pytest.raises(ValueError, match="upstream"):
        graph.wake([root])
    assert graph.status(root) == "sleeping"
    graph.wake([root, a.logical_id])
    assert graph.status(root) == graph.status(a) == "active"


def test_publication_all_contexts_cancellation_and_conflict(tmp_path):
    x = source(3)
    graph, store, _ = global_graph(tmp_path, negate(x))
    graph.register_context("second", anchor="2024-01-01")
    bindings = {"main": {"x": x}, "second": {"x": x}}
    one = graph.plan_update(bindings, through=x.domain.times.max())
    two = graph.plan_update(bindings, through=x.domain.times.max())
    one.execute("main", one.ready("main")[0])
    with pytest.raises(ValueError, match="incomplete"):
        one.publish()
    assert store.graph_state("global")["current_update"] is None
    def cancel():
        raise RuntimeError("canceled")
    with pytest.raises(RuntimeError, match="canceled"):
        one.execute("second", one.ready("second")[0], check_canceled=cancel)
    assert store.graph_state("global")["current_update"] is None
    receipt = finish(one)
    with pytest.raises(RevisionConflict):
        finish(two)
    assert one.publish() == receipt


def test_sources_require_content_evidence_and_through_coverage(tmp_path):
    x = source(3)
    graph, store, _ = global_graph(tmp_path, negate(x))
    lazy = Node.from_domain(x.lazy(), x.domain, source_key="x")
    with pytest.raises(ValueError, match="evidence"):
        graph.plan_update({"main": {"x": lazy}}, through=x.domain.times.max())
    with pytest.raises(ValueError, match="evidence"):
        negate(lazy).compute(runtime=ExecutionRuntime(materialization_store=store))
    with pytest.raises(ValueError, match="final session"):
        graph.plan_update({"main": {"x": x}}, through=date(2030, 1, 1))
    assert source(3, revision=1).identity != x.identity


def test_publish_checks_values_and_all_evidence(tmp_path):
    x = source(3)
    graph, store, root = global_graph(tmp_path, negate(x))
    receipt = finish(graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    record = store.read(receipt["results"]["main"][root])
    manifest = store.manifest(record.key.identity)
    store.publish(record, execution_policy=manifest["execution_policy"])
    changed = Node.from_domain(record.panel.collect().with_columns(pl.lit(99.0).alias("value")), record.panel.domain)
    for candidate in (replace(record, panel=changed), replace(record, checkpoint={"execution_policy": "changed"})):
        with pytest.raises(ValueError, match="conflict"):
            store.publish(candidate, execution_policy=manifest["execution_policy"])
    assert isinstance(record, NodeMaterialization)


@pytest.mark.parametrize("operation", ["ewm_mean", "ewm_var", "ewm_std", "regularized_weights"])
def test_checkpoint_append_and_revised_prefix_match_cold(tmp_path, operation):
    from bagelquant_core import OPERATOR_REGISTRY
    registered = next(OPERATOR_REGISTRY.get(name) for name in OPERATOR_REGISTRY.names() if name.endswith("." + operation))
    def expression(x):
        if operation.startswith("ewm_"):
            return registered(x, alpha=0.4)
        from bagelquant_core.operator import prediction_signal, top_n, equal_weight
        if operation == "regularized_weights":
            return registered(prediction_signal(x), max_weight=0.6)
        return registered(equal_weight(top_n(prediction_signal(x), count=2)), every=3, anchor="2024-01-01")
    first = source(35, dynamic=True)
    graph, store, root = global_graph(tmp_path/"incremental", expression(first))
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    for index, x in enumerate((source(45, dynamic=True), source(50, revision=10, dynamic=True))):
        plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max())
        updated = finish(plan)
        reference_graph, reference_store, reference_root = global_graph(tmp_path/f"reference-{index}", expression(x))
        reference = finish(reference_graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
        actual = store.read(updated["results"]["main"][root])
        expected = reference_store.read(reference["results"]["main"][reference_root])
        assert_frame_equal(actual.panel.collect(include_traces=True), expected.panel.collect(include_traces=True), check_exact=True)
        assert actual.checkpoint == expected.checkpoint
        assert plan.resource_usage["main"][root]["checkpoint_through"] == (str(first.domain.times.max()) if index == 0 else None)
