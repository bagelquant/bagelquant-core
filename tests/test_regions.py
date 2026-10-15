"""Durable pointwise regions retain exact interval evidence and retry units."""
import json
import sqlite3
from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import CoreStore, Domain, Graph, Node, compute_finite_work
from bagelquant_core.operator import canonicalize_values, ceil, constant, div, log, negate, project_domain, rolling_mean
from test_execution_preparation import retired_source
from test_global_graph import finish, global_graph
from bagelquant_core.resources import ResourceLimits


def expression(value):
    return canonicalize_values(negate(ceil(div(value, constant(value, value=10)))))


def memory_source(count, *, revision=None, assets=8):
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(count)]
    domain = Domain(calendar=days, universe=[f"A{index}" for index in range(assets)])
    frame = domain.grid_lazy().with_columns(pl.lit(1.).alias("value"), pl.col("time").alias("available"))
    if revision is not None:
        frame = frame.with_columns(pl.when(pl.col("time") == days[revision]).then(2.).otherwise(pl.col("value")).alias("value"))
    return Node.from_domain(frame.collect(), domain, source_key="x", trace_columns=("available",))


def long_expression(value):
    for _ in range(24):
        value = negate(value)
    return value


@pytest.mark.parametrize("windowed", [False, True])
def test_region_memory_admission_precedes_source_reads_and_kernel(tmp_path, monkeypatch, windowed):
    value = memory_source(66)
    graph, store, _root = global_graph(tmp_path, long_expression(value))
    def forbidden(*_args, **_kwargs):
        pytest.fail("oversized region consumed numerical inputs or launched a kernel")
    monkeypatch.setattr("bagelquant_core.regions._parent_window", forbidden)
    monkeypatch.setattr("bagelquant_core.regions._run_finite_outputs", forbidden)
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(),
        input_loader=forbidden, input_window_loader=forbidden if windowed else None, kernel_dispatch=forbidden)
    with pytest.raises(MemoryError, match="retained outputs.*persist_nodes"):
        finish(plan, limits=ResourceLimits(memory_target_mib=1, cache_mib=0))
    assert not store.inventory()
    assert store.graph_state("global")["current_update"] is None


def test_reduced_region_memory_reuses_completed_history_and_admits_only_small_append(tmp_path, monkeypatch):
    before, current = memory_source(66), memory_source(67)
    graph, store, root = global_graph(tmp_path, long_expression(before))
    old = finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max()))
    low = ResourceLimits(memory_target_mib=1, cache_mib=0)
    def forbidden(*_args, **_kwargs):
        pytest.fail("retained or oversized region consumed numerical inputs")
    with monkeypatch.context() as unchanged:
        unchanged.setattr("bagelquant_core.regions._parent_window", forbidden)
        repeated = graph.plan_update({"main": {"x": before}}, through=before.domain.times.max(), input_loader=forbidden)
        assert repeated.complete
        assert finish(repeated, limits=low)["results"] == old["results"]
    reads, estimates = [], []
    def window(_context, _key, start, end):
        reads.append((start, end))
        return Node.from_domain(current.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
            current.domain.window(start, end), identity=current.identity,
            trace_columns=current.trace_columns, trace_identity=current.trace_identity)
    class Inline:
        def should_dispatch(self, estimate):
            estimates.append(estimate)
            return False
        def __call__(self, _work):
            pytest.fail("inline predicate launched a child")
    appended = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
        input_loader=forbidden, input_window_loader=window, kernel_dispatch=Inline())
    saved = finish(appended, limits=low)
    assert reads == [(current.domain.times[-1], current.domain.times[-1])]
    assert all(usage["computed_output_sessions"] == 1 for usage in appended.resource_usage["main"].values())
    assert len(estimates) == 1 and estimates[0] > current.domain.window(*reads[0]).size * 48 * 24
    reference, reference_store, reference_root = global_graph(tmp_path / "reference", long_expression(current))
    expected = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    assert saved["results"]["main"][root] == expected["results"]["main"][reference_root]
    assert_frame_equal(store.read_values(saved["results"]["main"][root]),
        reference_store.read_values(expected["results"]["main"][reference_root]), check_exact=True)
    repair = memory_source(67, revision=40)
    plan = graph.plan_update({"main": {"x": repair}}, through=repair.domain.times.max(),
        input_loader=forbidden, input_window_loader=forbidden)
    with pytest.raises(MemoryError, match="admitted memory"):
        finish(plan, limits=low)
    assert store.graph_state("global")["current_update"] == saved["identity"]
    assert store.describe(saved["results"]["main"][root]) is not None


@pytest.mark.parametrize("region", [False, True])
def test_region_worker_checks_its_smaller_budget_before_staged_reads(tmp_path, monkeypatch, region):
    value = memory_source(32, assets=8 if region else 128)
    graph, store, _root = global_graph(tmp_path, long_expression(value) if region else negate(value))
    def forbidden(*_args, **_kwargs):
        pytest.fail("oversized child consumed staged inputs")
    def dispatch(work):
        assert work.estimated_payload_bytes > 1024 * 1024 // 8
        with monkeypatch.context() as child:
            child.setattr(pl, "read_ipc", forbidden)
            return compute_finite_work(replace(work, limits=ResourceLimits(memory_target_mib=1, cache_mib=0)))
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), kernel_dispatch=dispatch)
    with pytest.raises(MemoryError, match="admitted memory"):
        finish(plan)
    assert not store.inventory()
    assert store.graph_state("global")["current_update"] is None


@pytest.mark.parametrize("windowed", [False, True])
def test_single_finite_kernel_has_the_same_pre_read_memory_guard(tmp_path, monkeypatch, windowed):
    value = memory_source(32, assets=128)
    graph, store, _root = global_graph(tmp_path, negate(value))
    def forbidden(*_args, **_kwargs):
        pytest.fail("oversized finite kernel consumed inputs")
    monkeypatch.setattr("bagelquant_core.incremental._parent_window", forbidden)
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), input_loader=forbidden,
        input_window_loader=forbidden if windowed else None)
    assert not plan.regions
    with pytest.raises(MemoryError, match="admitted memory"):
        finish(plan, limits=ResourceLimits(memory_target_mib=1, cache_mib=0))
    assert not store.inventory()


def initialize(path, value, *, persist=False, dispatch=None):
    graph, store, root = global_graph(path, expression(value))
    nodes = tuple(node.node_id for node in expression(value).logical_spec().nodes if node.node_type == "operator")
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(),
        persist_nodes=nodes if persist else (), kernel_dispatch=dispatch)
    return graph, store, root, plan, finish(plan)


@pytest.mark.parametrize("dispatch", [None, compute_finite_work])
def test_fused_region_keeps_keys_and_exact_values_with_fewer_publications(tmp_path, dispatch):
    value = retired_source(70)
    _, baseline, root, original, old = initialize(tmp_path / "baseline", value, persist=True)
    graph, fused, current_root, plan, saved = initialize(tmp_path / "fused", value, dispatch=dispatch)
    identity = saved["results"]["main"][current_root]
    assert root == current_root and identity == old["results"]["main"][root]
    assert len(plan.regions) == 1 and len(plan.regions[0].nodes) == 5
    assert set(saved["results"]["main"]) == {root}
    assert set(saved["coverage"]["main"]) == set(plan.regions[0].nodes) - {root}
    assert len(baseline.inventory()) == 20
    assert len(fused.inventory()) == 4
    assert_frame_equal(fused.read_values(identity, include_traces=True), baseline.read_values(identity, include_traces=True), check_exact=True)
    assert {item["identity"] for item in baseline.inventory(node_id=root)} == {
        item["identity"] for item in fused.inventory(node_id=root)}
    assert plan.keys == original.keys
    assert fused.describe(identity)["execution_policy"] == baseline.describe(identity)["execution_policy"]
    assert fused.update_covers(saved["identity"], "main", node_ids=plan.regions[0].nodes)
    assert not fused.update_covers(saved["identity"], "main", node_ids=("unknown",))
    assert fused.verify_update(saved["identity"]) == saved
    repeated = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    assert repeated.complete


def test_projection_region_preserves_sparse_membership_nulls_and_traces(tmp_path):
    source = retired_source(70)
    frame = source.collect(include_traces=True).filter(~((pl.col("time") == source.domain.times[33]) & (pl.col("asset_id") == "A")))
    frame = frame.with_columns(pl.when(pl.col("time") == source.domain.times[50]).then(None).otherwise(pl.col("value")).alias("value"))
    value = Node.from_domain(frame, source.domain, source_key="x", trace_columns=source.trace_columns)
    keys = frame.select("time", "asset_id").filter(pl.col("time") != source.domain.times[34])
    membership_domain = Domain(calendar=source.domain.times, universe=keys.with_columns(pl.lit(True).alias("active")))
    membership = Node.from_domain(keys.with_columns(pl.lit(1.).alias("value")), membership_domain, source_key="member")
    expression_ = canonicalize_values(negate(log(project_domain(value, membership=membership))))
    outputs = []
    for name, persist in (("full", True), ("fused", False)):
        graph, store, root = global_graph(tmp_path / name, expression_)
        operators = [node.node_id for node in expression_.logical_spec().nodes if node.node_type == "operator"]
        plan = graph.plan_update({"main": {"x": value, "member": membership}}, through=value.domain.times.max(),
            persist_nodes=operators if persist else ())
        saved = finish(plan)
        outputs.append((saved["results"]["main"][root], store.read_values(saved["results"]["main"][root], include_traces=True)))
    assert outputs[0][0] == outputs[1][0]
    assert_frame_equal(outputs[0][1], outputs[1][1], check_exact=True)


def test_shared_parent_is_durable_once_for_union_roots(tmp_path):
    value = retired_source(70)
    shared = negate(value)
    union = Graph(outputs=[canonicalize_values(ceil(shared), name="first"), canonicalize_values(shared, name="second")])
    graph, store, _root = global_graph(tmp_path, union)
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    assert shared.logical_id not in plan._transient_owners
    saved = finish(plan)
    assert shared.logical_id in saved["results"]["main"]
    assert plan.resource_usage["main"][shared.logical_id]["computed_blocks"] == 3
    assert len(store.inventory(node_id=shared.logical_id)) == 4


@pytest.mark.parametrize("dispatch", [None, compute_finite_work])
def test_region_keeps_durable_ancestor_auxiliary_evidence(tmp_path, dispatch, monkeypatch):
    from test_training_checkpoint_evidence import expression as trained, sources
    inputs = sources(18)
    value = inputs["x"]
    frame = value.collect(include_traces=True)
    model = trained(inputs)
    root_expression = canonicalize_values(negate(model))
    def content(value):
        if isinstance(value, dict):
            return {key: content(item) for key, item in value.items() if key != "$file"}
        if isinstance(value, list):
            return [content(item) for item in value]
        return value
    expected = None
    for name, persist in (("durable", True), ("region", False)):
        store = CoreStore(tmp_path / name / "core.sqlite", tmp_path / name / "values")
        publish = store.publish
        def publish_model(record, **kwargs):
            if record.key.node_id == model.logical_id:
                record = replace(record, artifacts={**record.artifacts, "diagnostic": frame},
                    checkpoint={**record.checkpoint, "fixture_panel": frame})
            return publish(record, **kwargs)
        monkeypatch.setattr(store, "publish", publish_model)
        graph = Graph(store=store, graph_id="evidence")
        root = next(iter(graph.merge(root_expression).outputs.values()))
        graph.register_context("main", anchor=value.domain.times.min())
        operators = [node.node_id for node in root_expression.logical_spec().nodes if node.node_type == "operator"]
        plan = graph.plan_update({"main": inputs}, through=value.domain.times.max(),
            persist_nodes=operators if persist else (), kernel_dispatch=dispatch)
        saved = finish(plan)
        identity = saved["results"]["main"][root]
        with monkeypatch.context() as metadata:
            metadata.setattr(store, "_decode", lambda *_: pytest.fail("evidence description decoded auxiliary bytes"))
            description = store.describe_evidence(identity)
            artifact = store.describe_artifact(identity, "diagnostic")
        assert description["artifacts"] == ("diagnostic",)
        assert description["checkpoint_nodes"] == (model.logical_id,)
        assert description["training_audit_count"] > 0
        model_identity = saved["results"]["main"][model.logical_id]
        assert artifact["identity"] == model_identity
        evidence = store.evidence(identity)
        assert_frame_equal(evidence["artifacts"]["diagnostic"], frame, check_exact=True)
        assert_frame_equal(store.read_artifact(identity, "diagnostic"), frame, check_exact=True)
        assert_frame_equal(evidence["checkpoints"][model.logical_id]["fixture_panel"], frame, check_exact=True)
        checkpoint = content(store.manifest(model_identity)["checkpoint"])
        if expected is None:
            expected = (identity, description, checkpoint, evidence["training_audits"])
        else:
            assert (identity, description, checkpoint, evidence["training_audits"]) == expected
            assert len(plan.regions) == 1
            assert all(store.describe(plan.keys["main"][node].identity) is None
                for node in plan.logical_coverage["main"])


def test_append_and_local_revision_reuse_old_region_work_and_references(tmp_path):
    before, current = retired_source(66), retired_source(70)
    graph, store, root, _plan, old = initialize(tmp_path / "region", before)
    old_identity = old["results"]["main"][root]
    old_manifest = store.manifest(old_identity)
    appended = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max())
    saved = finish(appended)
    _, reference, _, _full_plan, full = initialize(tmp_path / "full", current, persist=True)
    identity = saved["results"]["main"][root]
    assert identity == full["results"]["main"][root]
    assert_frame_equal(store.read_values(identity, include_traces=True), reference.read_values(identity, include_traces=True), check_exact=True)
    assert store.manifest(old_identity) == old_manifest
    assert all(usage["computed_output_sessions"] == 4 and usage["reused_blocks"] == 2
        for usage in appended.resource_usage["main"].values())
    changed_frame = current.collect(include_traces=True).with_columns(pl.when(pl.col("time") == current.domain.times[50])
        .then(99.).otherwise(pl.col("value")).alias("value"))
    changed = Node.from_domain(changed_frame, current.domain, source_key="x", trace_columns=current.trace_columns)
    repair = graph.plan_update({"main": {"x": changed}}, through=changed.domain.times.max())
    repaired = finish(repair)
    _, revised_reference, _, _revised_plan, revised = initialize(tmp_path / "revised", changed, persist=True)
    repaired_id = repaired["results"]["main"][root]
    assert repaired_id == revised["results"]["main"][root]
    assert_frame_equal(store.read_values(repaired_id, include_traces=True), revised_reference.read_values(repaired_id, include_traces=True), check_exact=True)
    assert all(usage["computed_output_sessions"] == 32 and usage["reused_blocks"] == 2
        for usage in repair.resource_usage["main"].values())


def test_metadata_repeat_skips_values_and_transient_kernels(tmp_path, monkeypatch):
    value = retired_source(70)
    graph, store, _root, _plan, old = initialize(tmp_path, value)
    def forbidden(*args, **kwargs):
        pytest.fail("unchanged region consumed numerical data")
    monkeypatch.setattr(store, "lookup", forbidden)
    monkeypatch.setattr(store, "read_values", forbidden)
    monkeypatch.setattr(Node, "collect", forbidden)
    with monkeypatch.context() as metadata:
        metadata.setattr(store, "manifest", forbidden)
        plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(),
            input_loader=forbidden, input_proof=forbidden, kernel_dispatch=forbidden)
        assert plan.complete
        assert plan.input_proof_windows("main").is_empty()
    saved = finish(plan)
    assert saved["results"] == old["results"] and saved["coverage"] == old["coverage"]


def test_retry_retains_atomic_region_blocks(tmp_path, monkeypatch):
    value = retired_source(70)
    graph, store, root = global_graph(tmp_path, expression(value))
    original = store.publish
    canceled = []
    def publish(record, **kwargs):
        original(record, **kwargs)
        if "block" in kwargs.get("region_evidence", {}):
            canceled.append(True)
    monkeypatch.setattr(store, "publish", publish)
    def check():
        if canceled:
            raise InterruptedError("canceled after durable unit")
    first = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    with pytest.raises(InterruptedError):
        first.execute("main", root, check_canceled=check)
    assert store.node_completions("global", "main") == {}
    monkeypatch.setattr(store, "publish", original)
    retry = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    finish(retry)
    assert all(usage["reused_blocks"] == 1 and usage["computed_output_sessions"] == 38
        for usage in retry.resource_usage["main"].values())


def test_explicit_internal_output_is_durable_and_old_roots_stay_current(tmp_path):
    value = retired_source(3)
    graph, store, root, plan, saved = initialize(tmp_path, value)
    internal = plan.regions[0].nodes[1]
    promoted = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), persist_nodes=(internal,))
    assert root in promoted.results["main"]
    finish(promoted)
    assert internal in promoted.results["main"]
    assert promoted.results["main"][root] == saved["results"]["main"][root]
    assert store.describe(promoted.results["main"][internal])["node_id"] == internal


def test_switching_persistence_does_not_replay_completed_history(tmp_path):
    before, current = retired_source(66), retired_source(70)
    graph, _store, _root, _plan, _old = initialize(tmp_path, before, persist=True)
    appended = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max())
    assert not appended.regions  # preserve original durable authority
    finish(appended)
    assert all(usage["computed_output_sessions"] == 4 for usage in appended.resource_usage["main"].values())


def test_relevant_transient_implementation_change_creates_new_authority(tmp_path, monkeypatch):
    value = retired_source(70)
    graph, store, root, _plan, old = initialize(tmp_path / "original", value)
    previous = old["results"]["main"][root]
    retained = store.manifest(previous)
    monkeypatch.setattr(negate, "version", "new-implementation")
    changed = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    saved = finish(changed)
    _, reference, _, _full, expected = initialize(tmp_path / "new-reference", value, persist=True)
    identity = saved["results"]["main"][root]
    assert identity != previous and identity == expected["results"]["main"][root]
    assert_frame_equal(store.read_values(identity, include_traces=True), reference.read_values(identity, include_traces=True), check_exact=True)
    assert store.manifest(previous) == retained
    assert all(usage["computed_output_sessions"] == 70 for usage in changed.resource_usage["main"].values())


def test_rolling_boundaries_stay_durable_and_bad_coverage_is_rejected(tmp_path):
    value = retired_source(70)
    graph, store, root = global_graph(tmp_path, negate(rolling_mean(expression(value), window=10)))
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    rolling = next(node.node_id for node in plan.spec.nodes if node.operator == rolling_mean.registry_name)
    assert all(rolling not in region.nodes for region in plan.regions)
    saved = finish(plan)
    assert rolling in saved["results"]["main"]
    bad = {"main": {rolling: root}}
    with pytest.raises(ValueError, match="transient|seal"):
        store.commit_update("global", revision=saved["revision"], state_revision=saved["state_revision"],
            expected_previous=saved["identity"], results=saved["results"], coverage=bad, through=saved["through"])


def test_corrupt_region_certificate_blocks_reuse_before_kernel(tmp_path, monkeypatch):
    value = retired_source(70)
    graph, store, _root, _plan, _old = initialize(tmp_path, value)
    with sqlite3.connect(store.meta_path) as db:
        row = db.execute("SELECT identity,payload_json FROM region_evidence WHERE kind='block' LIMIT 1").fetchone()
        payload = json.loads(row[1])
        payload["region_id"] = "changed"
        db.execute("UPDATE region_evidence SET payload_json=? WHERE identity=? AND kind='block'", (json.dumps(payload), row[0]))
    changed = retired_source(71)
    plan = graph.plan_update({"main": {"x": changed}}, through=changed.domain.times.max())
    monkeypatch.setattr("bagelquant_core.regions._run_finite_outputs", lambda *_a, **_k: pytest.fail("corrupt proof ran math"))
    with pytest.raises(ValueError, match="region evidence"):
        finish(plan)


def test_unknown_completed_region_proof_blocks_before_input_or_kernel(tmp_path, monkeypatch):
    before, current = retired_source(70), retired_source(71)
    graph, store, _root, _plan, old = initialize(tmp_path, before)
    lazy = Node.from_domain(current.lazy(include_traces=True), current.domain,
        source_key="x", identity="new-envelope", trace_columns=current.trace_columns,
        trace_identity=current.trace_identity)
    def forbidden(*args, **kwargs):
        pytest.fail("unknown retained proof executed numerical work")
    monkeypatch.setattr("bagelquant_core.regions._run_finite_outputs", forbidden)
    plan = graph.plan_update({"main": {"x": lazy}}, through=lazy.domain.times.max(),
        input_proof=lambda *_: None, input_window_loader=forbidden, kernel_dispatch=forbidden)
    with pytest.raises(ValueError, match="interval proofs"):
        finish(plan)
    assert store.graph_state("global")["current_update"] == old["identity"]
