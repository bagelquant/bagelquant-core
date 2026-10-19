"""Explicit cutoff normalization preserves a generic graph and region history."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json

import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Graph, Node, ExecutionRuntime
from bagelquant_core.hashing import hash_mapping
from test_global_graph import source, global_graph, finish
from test_proof_maintenance import projection


def prepared(tmp_path, *, count=5, dynamic=False):
    before = source(count, dynamic=dynamic)
    expression = Graph.from_dsl("p = pct_change_from_last_change(x)\nf = ffill(p, limit=240)\noutput = canonicalize_values(rankpct(f))", inputs={"x": before})
    graph, store, root = global_graph(tmp_path, expression)
    end = before.domain.times.max()
    graph.register_context("main", anchor="2024-01-01", information_cutoff=end + timedelta(days=2))
    old_context = graph.snapshot()["contexts"]["main"]
    update = finish(graph.plan_update({"main": {"x": before}}, through=end))
    after = Node.from_domain(before.lazy(include_traces=True), before.domain,
        identity="replacement-source", trace_columns=before.trace_columns, trace_identity="replacement-traces")
    new_context = {**old_context, "information_cutoff": str(end)}
    kwargs = dict(mode="cutoff_normalization.v1", original_update=update["identity"], context_id="main",
        original_context=old_context, replacement_context=new_context,
        original_inputs={"x": before}, replacement_inputs={"x": after},
        original_projection=projection, replacement_projection=projection,
        report_id="exact-source-proof", report_digest="a" * 64)
    return before, after, graph, store, root, update, kwargs


@pytest.mark.parametrize("count,dynamic", [(5, False), (65, True)])
def test_cutoff_normalization_references_generic_and_region_history(tmp_path, monkeypatch, count, dynamic):
    before, after, graph, store, root, update, kwargs = prepared(tmp_path, count=count, dynamic=dynamic)
    end = before.domain.times.max()
    assert update["coverage"]["main"]
    originals = {identity: store.manifest(identity) for identity in update["results"]["main"].values()}
    files = {part["$file"]: store._file(part["$file"]).read_bytes()
        for manifest in originals.values() for part in manifest["partitions"]}
    spec = graph.snapshot()["spec"]
    plan = store.plan_input_proof_rebind(spec, update["results"]["main"][root], **kwargs)
    def forbidden(*_a, **_k):
        pytest.fail("normalization replayed completed numerical history")
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", forbidden)
        patch.setattr("bagelquant_core.incremental.execute_finite_node", forbidden)
        patch.setattr("bagelquant_core.update.execute_finite_node", forbidden)
        patch.setattr("bagelquant_core.update.execute_checkpoint_node", forbidden)
        patch.setattr("bagelquant_core.regions.execute_region", forbidden)
        result = store.apply_input_proof_rebind(plan)
        assert result["computed_sessions"] == 0
        assert store.apply_input_proof_rebind(plan) == result
        graph.register_context("main", **kwargs["replacement_context"])
        for _ in range(2):
            current = finish(graph.plan_update({"main": {"x": after}}, through=end, input_loader=forbidden))
            assert current["results"]["main"] == result["results"]
            assert current["coverage"] == update["coverage"]
    assert all(store.manifest(key) == value for key, value in originals.items())
    assert all(store._file(path).read_bytes() == value for path, value in files.items())
    assert {row["checked_through"] for row in store.node_completions("global", "main").values()} == {kwargs["original_context"]["information_cutoff"]}
    assert_frame_equal(store.read_values(current["results"]["main"][root], include_traces=True),
        store.read_values(update["results"]["main"][root], include_traces=True), check_exact=True)
    store.verify_update(current["identity"])
    reference, fresh, fresh_root = global_graph(tmp_path / "reference",
        Graph.from_logical_spec(spec, inputs={"x": after}))
    reference.register_context("main", **kwargs["replacement_context"])
    full = finish(reference.plan_update({"main": {"x": after}}, through=end))
    assert_frame_equal(store.read_values(current["results"]["main"][root], include_traces=True),
        fresh.read_values(full["results"]["main"][fresh_root], include_traces=True), check_exact=True)


@pytest.mark.parametrize("change", ["value", "unknown", "trace", "anchor", "parameters", "cutoff", "original_key"])
def test_cutoff_normalization_rejects_changed_or_unknown_evidence(tmp_path, change):
    before, after, graph, store, root, update, kwargs = prepared(tmp_path)
    if change == "value":
        kwargs["replacement_inputs"] = {"x": source(5, revision=2)}
    elif change == "unknown":
        kwargs["replacement_projection"] = lambda *_: None
    elif change == "trace":
        kwargs["replacement_projection"] = lambda *args: replace(projection(*args), trace_columns=())
    elif change in {"anchor", "parameters", "cutoff"}:
        field, value = {"anchor": ("anchor", "2024-01-02"), "parameters": ("parameters", {"x": 2}),
            "cutoff": ("information_cutoff", "2024-01-04")}[change]
        kwargs["replacement_context"] = {**kwargs["replacement_context"], field: value}
    else:
        kwargs["original_inputs"] = {"x": after}
    initial = len(store.inventory())
    with pytest.raises(ValueError):
        store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root], **kwargs)
    assert len(store.inventory()) == initial


def test_unadmitted_generic_history_guard_and_forged_admission_remain_closed(tmp_path):
    _, after, graph, store, root, update, kwargs = prepared(tmp_path)
    plan = store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root], **kwargs)
    forged = deepcopy(plan)
    forged["replacement_context"]["information_cutoff"] = "2024-01-04"
    forged["identity"] = hash_mapping({key: value for key, value in forged.items() if key != "identity"})
    with pytest.raises(ValueError, match="admission"):
        store.apply_input_proof_rebind(forged)
    graph.register_context("main", **kwargs["replacement_context"])
    with pytest.raises(ValueError, match="no declared finite or checkpoint"):
        finish(graph.plan_update({"main": {"x": after}}, through="2024-01-05"))


@pytest.mark.parametrize("problem", ["seal_checksum", "seal_content", "block_checksum", "block_index", "update", "admission"])
@pytest.mark.parametrize("published", [False, True])
def test_original_authority_is_rechecked_at_apply_read_and_audit(tmp_path, problem, published):
    _, _, graph, store, root, update, kwargs = prepared(tmp_path)
    plan = store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root], **kwargs)
    result = store.apply_input_proof_rebind(plan) if published else None
    region = next(item for item in plan["nodes"].values() if item.get("original_seal"))
    block = region["original_seal"]["blocks"][0]
    with store.transaction() as connection:
        if problem.startswith("seal"):
            column = "payload_identity" if problem == "seal_checksum" else "content_identity"
            connection.execute(f"UPDATE region_evidence SET {column}=? WHERE identity=? AND kind='seal'",
                ("invalid", region["original_identity"]))
        elif problem == "block_checksum":
            connection.execute("UPDATE region_evidence SET payload_identity=? WHERE identity=? AND kind='block'", ("invalid", block))
        elif problem == "block_index":
            connection.execute("DELETE FROM computation_index WHERE identity=?", (block,))
        elif problem == "update":
            connection.execute("UPDATE updates SET through='2024-01-04' WHERE identity=?", (update["identity"],))
        else:
            connection.execute("DELETE FROM input_proof_rebinding_plans WHERE identity=?", (plan["identity"],))
    with pytest.raises((ValueError, KeyError)):
        store.apply_input_proof_rebind(plan)
    if published:
        for operation in (store.describe, store.read, store.verify_identity):
            with pytest.raises((ValueError, KeyError)):
                operation(result["results"][root])


def test_region_coordinate_tampering_is_not_a_cutoff_equivalence(tmp_path):
    _, _, graph, store, root, update, kwargs = prepared(tmp_path)
    seal = store.region_evidence(update["results"]["main"][root], kind="seal")[0]
    block = seal["blocks"][0]
    certificate = deepcopy(store.region_evidence(block, kind="block")[0])
    transient = next(node for node in certificate["nodes"] if node != root)
    certificate["nodes"][transient]["index"]["calendar"].pop()
    with store.transaction() as connection:
        connection.execute("UPDATE region_evidence SET payload_json=?,payload_identity=? WHERE identity=? AND kind='block'",
            (json.dumps(certificate), hash_mapping(certificate), block))
    with pytest.raises(ValueError, match="coordinates|region evidence"):
        store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root], **kwargs)


def test_interrupted_normalization_resumes_without_kernels(tmp_path, monkeypatch):
    _, _, graph, store, root, update, kwargs = prepared(tmp_path)
    plan = store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root], **kwargs)
    initial = len(store.inventory())
    def cancel_after_first():
        if len(store.inventory()) > initial:
            raise RuntimeError("canceled")
    def forbidden(*_a, **_k):
        pytest.fail("normalization ran a numerical kernel")
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    with pytest.raises(RuntimeError, match="canceled"):
        store.apply_input_proof_rebind(plan, check_canceled=cancel_after_first)
    assert len(store.inventory()) == initial + 1
    assert store.apply_input_proof_rebind(plan)["computed_sessions"] == 0


def test_ordinary_key_planning_does_not_retain_context_recipes(tmp_path):
    value = source(5)
    runtime = ExecutionRuntime()
    runtime.plan_materialization_keys(Graph.from_dsl("output = rankpct(x)", inputs={"x": value}))
    assert runtime.materialization_contexts is None


def test_transitive_normalization_retains_original_admission_authority(tmp_path):
    _, after, graph, store, root, update, kwargs = prepared(tmp_path)
    final_context = kwargs["replacement_context"]
    middle_context = {**final_context, "information_cutoff": "2024-01-06"}
    plan = store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root],
        **{**kwargs, "replacement_context": middle_context})
    store.apply_input_proof_rebind(plan)
    graph.register_context("main", **middle_context)
    middle = finish(graph.plan_update({"main": {"x": after}}, through="2024-01-05"))
    second = store.plan_input_proof_rebind(graph.snapshot()["spec"], middle["results"]["main"][root],
        **{**kwargs, "original_update": middle["identity"], "original_context": middle_context,
            "original_inputs": {"x": after}})
    result = store.apply_input_proof_rebind(second)
    store.verify_identity(result["results"][root])
    with store.transaction() as connection:
        connection.execute("DELETE FROM input_proof_rebinding_plans WHERE identity=?", (plan["identity"],))
    with pytest.raises(ValueError, match="admission"):
        store.describe(result["results"][root])
