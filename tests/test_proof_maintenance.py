"""Explicit proof-format maintenance reuses original numerical evidence."""
from dataclasses import replace
from copy import deepcopy

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import ConsumedInputEvidence, ExecutionRuntime, Node, NodeMaterialization, materialization_trace_identity
from bagelquant_core.operator import negate
from bagelquant_core.hashing import hash_mapping, hash_dataframe
from bagelquant_core.incremental import scoped_domain
from test_global_graph import source, global_graph, finish


def source_proof(namespace, node, start, end):
    return f"{namespace}:{node.interval_identity(start, end)}"


def bound_proof(namespace):
    def proof(_parent, node, start, end):
        return node.source_interval_identity(source_proof(namespace, node, start, end), start, end)
    return proof


def projection(_parent, node, start, end):
    return ConsumedInputEvidence(f"original-source:{len(node.domain.times)}", "d" * 64,
        hash_dataframe(node.collect(include_traces=True).filter(pl.col("time").is_between(start, end))),
        start, end, scoped_domain(node.domain, start, end).signature, node.value_type.value, node.trace_columns)


def prepare(store, graph, record, before, after, *, replacement_projection=projection):
    spec = graph._state()["spec"]
    parent = next(node.node_id for node in spec.nodes if node.node_type == "input")
    return store.plan_input_proof_rebind(spec, record["identity"], domain=after.domain,
        original_parents={parent: before}, replacement_parents={parent: after},
        original_input_proof=bound_proof("original"), replacement_input_proof=bound_proof("scalar"),
        original_projection=projection, replacement_projection=replacement_projection,
        parameters={}, report_id="source-owner-report", report_digest="e" * 64)


def initial(tmp_path):
    before = source(63)
    graph, store, root = global_graph(tmp_path, negate(before))
    finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max(),
        input_proof=lambda _ctx, _key, start, end: source_proof("original", before, start, end)))
    blocks = [store.manifest(row["identity"]) for row in store.inventory(node_id=root)
        if store.manifest(row["identity"])["coverage_start"] != "2024-01-01"
        or store.manifest(row["identity"])["through"] == "2024-02-01"]
    assert len(blocks) == 2
    return before, graph, store, root, blocks


def test_explicit_format_maintenance_keeps_original_files_and_only_computes_new_day(tmp_path, monkeypatch):
    before, graph, store, root, blocks = initial(tmp_path / "incremental")
    after = source(64)
    plans = [prepare(store, graph, record, before, after) for record in blocks]
    original = {record["identity"]: store.manifest(record["identity"]) for record in blocks}
    files = {part["$file"]: store._file(part["$file"]).read_bytes() for record in blocks for part in record["partitions"]}
    def forbidden(*_args, **_kwargs):
        pytest.fail("proof maintenance ran a kernel or audited original values")
    import bagelquant_core.store as store_module
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", forbidden)
        patch.setattr(store_module, "_checksum", forbidden)
        for plan in plans:
            assert store.apply_input_proof_rebind(plan)["outcome"] == "rebound"
            assert store.apply_input_proof_rebind(plan)["outcome"] == "up_to_date"
    assert all(store.manifest(key) == record for key, record in original.items())
    assert all(store._file(path).read_bytes() == payload for path, payload in files.items())
    plan = graph.plan_update({"main": {"x": after}}, through=after.domain.times.max(),
        input_proof=lambda _ctx, _key, start, end: source_proof("scalar", after, start, end))
    updated = finish(plan)
    assert plan.resource_usage["main"][root]["computed_output_sessions"] == 1
    assert plan.resource_usage["main"][root]["kernel_sessions"] == 1
    reference, fresh, reference_root = global_graph(tmp_path / "reference", negate(after))
    full = finish(reference.plan_update({"main": {"x": after}}, through=after.domain.times.max(),
        input_proof=lambda _ctx, _key, start, end: source_proof("scalar", after, start, end)))
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True),
        fresh.read_values(full["results"]["main"][reference_root], include_traces=True), check_exact=True)
    store.verify_update(updated["identity"])


@pytest.mark.parametrize("problem", ["value", "membership", "missing_traces", "missing_source", "wrong_parameters"])
def test_maintenance_rejects_changed_or_incomplete_original_input_evidence(tmp_path, problem):
    before, graph, store, _root, blocks = initial(tmp_path)
    after = source(64, revision=5) if problem == "value" else source(64, dynamic=problem == "membership")
    def evidence(parent, node, start, end):
        result = projection(parent, node, start, end)
        if problem == "missing_traces":
            return replace(result, trace_columns=())
        if problem == "missing_source":
            return None
        return result
    if problem == "wrong_parameters":
        manifest = store.manifest(blocks[0]["identity"])
        modified = {**manifest["execution_policy"], "parameter_bindings": {"$unknown": 1}}
        with store.transaction() as connection:
            import json
            from bagelquant_core.store import _manifest_content
            manifest["execution_policy"] = modified
            manifest["content_identity"] = hash_mapping(_manifest_content(manifest))
            connection.execute("update materializations set manifest_json=? where identity=?",
                (json.dumps(manifest), manifest["identity"]))
    with pytest.raises(ValueError, match="changed|authority|parameter"):
        prepare(store, graph, blocks[0], before, after, replacement_projection=evidence)


def test_admission_rejects_forged_new_key_and_original_loss(tmp_path):
    before, graph, store, _root, blocks = initial(tmp_path)
    plan = prepare(store, graph, blocks[0], before, source(64))
    forged = {**plan, "replacement_key": {**plan["replacement_key"], "node_id": "unrelated"}}
    forged["identity"] = hash_mapping({key: value for key, value in forged.items() if key != "identity"})
    with pytest.raises(ValueError, match="numerical contract|immutable source admission"):
        store.apply_input_proof_rebind(forged)
    maintained = store.apply_input_proof_rebind(plan)
    with store.transaction() as connection:
        connection.execute("delete from materializations where identity=?", (blocks[0]["identity"],))
    with pytest.raises(ValueError, match="authority is missing"):
        store.describe(maintained["identity"])


def test_legacy_parameter_proof_is_admitted_only_against_its_original_context(tmp_path):
    from bagelquant_core.materialization import MaterializationKey
    before, graph, store, _root, blocks = initial(tmp_path)
    original = blocks[0]
    record = store.read(original["identity"])
    policy = {name: value for name, value in original["execution_policy"].items() if name != "parameter_bindings"}
    parameters = {"$unused": 1}
    key = replace(record.key, context_identity=hash_mapping({"block": policy,
        "start": original["coverage_start"], "end": original["through"], "parameters": parameters}))
    value = Node.from_domain(record.panel.lazy(include_traces=True), record.panel.domain,
        value_type=record.panel.value_type, identity=key.identity, trace_columns=record.panel.trace_columns,
        trace_identity=materialization_trace_identity(key, record.panel.trace_columns))
    store.publish(NodeMaterialization(key, value), execution_policy=policy, value_references=(record.key.identity,))
    spec = graph._state()["spec"]
    parent = next(node.node_id for node in spec.nodes if node.node_type == "input")
    kwargs = dict(domain=source(64).domain, original_parents={parent: before}, replacement_parents={parent: source(64)},
        original_input_proof=bound_proof("original"), replacement_input_proof=bound_proof("scalar"),
        original_projection=projection, replacement_projection=projection,
        report_id="original-context", report_digest="e" * 64)
    with pytest.raises(ValueError, match="completed context"):
        store.plan_input_proof_rebind(spec, key.identity, parameters={"$unused": 2}, **kwargs)
    plan = store.plan_input_proof_rebind(spec, key.identity, parameters=parameters, **kwargs)
    result = store.apply_input_proof_rebind(plan)
    maintained = store.manifest(result["identity"])
    assert maintained["execution_policy"]["parameter_bindings"] == {}
    assert MaterializationKey.from_dict(maintained["key"]).context_identity == key.context_identity
    assert "parameter_bindings" not in store.manifest(key.identity)["execution_policy"]
    forged = deepcopy(plan)
    forged["parameters"]["$unused"] = 2
    forged["identity"] = hash_mapping({name: value for name, value in forged.items() if name != "identity"})
    with pytest.raises(ValueError, match="immutable source admission"):
        store.apply_input_proof_rebind(forged)


@pytest.mark.parametrize("problem", ["proof", "missing_source", "reversed_scope", "type", "traces", "report"])
def test_apply_and_cache_hit_reject_resigned_plan_mutations(tmp_path, problem):
    before, graph, store, _root, blocks = initial(tmp_path)
    plan = prepare(store, graph, blocks[0], before, source(64))
    store.apply_input_proof_rebind(plan)
    forged = deepcopy(plan)
    entry = forged["evidence"][forged["parents"][0]]
    if problem == "proof":
        revised = bound_proof("scalar")(forged["parents"][0], source(64, revision=5),
            before.domain.times.min(), before.domain.times[31])
        entry["replacement_proof"] = hash_mapping({"version": "finite_causal_blocks.v3", "content": revised,
            "domain": entry["replacement"]["domain_signature"], "type": before.value_type.value,
            "traces": before.trace_columns})
        forged["replacement_key"]["inputs"] = [entry["replacement_proof"]]
    elif problem == "report":
        forged["report_id"], forged["report_digest"] = "", "not-sha256"
    else:
        for label in ("original", "replacement"):
            evidence = entry[label]
            if problem == "missing_source":
                evidence["source_receipt"], evidence["source_digest"] = "", ""
            elif problem == "reversed_scope":
                evidence["start"], evidence["end"] = "2099-01-01", "1900-01-01"
            elif problem == "type":
                evidence["value_type"] = "not-a-type"
            else:
                evidence["trace_columns"] = []
    forged["identity"] = hash_mapping({name: value for name, value in forged.items() if name != "identity"})
    with pytest.raises(ValueError, match="immutable source admission"):
        store.apply_input_proof_rebind(forged)


def test_transitive_rebinding_requires_native_source_and_admission_records(tmp_path):
    before, graph, store, _root, blocks = initial(tmp_path)
    after = source(64)
    plan = prepare(store, graph, blocks[0], before, after)
    first = store.apply_input_proof_rebind(plan)
    parent = plan["parents"][0]
    second_plan = store.plan_input_proof_rebind(graph._state()["spec"], first["identity"], domain=after.domain,
        original_parents={parent: after}, replacement_parents={parent: after},
        original_input_proof=bound_proof("scalar"), replacement_input_proof=bound_proof("next"),
        original_projection=projection, replacement_projection=projection,
        parameters={}, report_id="next-source-report", report_digest="f" * 64)
    second = store.apply_input_proof_rebind(second_plan)
    captured = store.manifest(second["identity"])
    with store.transaction() as connection:
        connection.execute("DELETE FROM input_proof_rebinding_plans WHERE identity=?", (plan["identity"],))
    with pytest.raises(ValueError, match="immutable source admission"):
        store.describe(second["identity"])
    with pytest.raises(ValueError, match="immutable source admission"):
        store.verify(captured)
    with store.transaction() as connection:
        import json
        connection.execute("INSERT INTO input_proof_rebinding_plans VALUES(?,?)", (plan["identity"], json.dumps(plan)))
        connection.execute("DELETE FROM materializations WHERE identity=?", (blocks[0]["identity"],))
    with pytest.raises(ValueError, match="authority is missing"):
        store.describe(second["identity"])
    with pytest.raises(ValueError, match="authority is missing"):
        store.verify(captured)


def test_cancelled_admission_publishes_no_positive_authority(tmp_path):
    before, graph, store, _root, blocks = initial(tmp_path)
    after = source(64)
    parent = next(node.node_id for node in graph._state()["spec"].nodes if node.node_type == "input")
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        store.plan_input_proof_rebind(graph._state()["spec"], blocks[0]["identity"], domain=after.domain,
            original_parents={parent: before}, replacement_parents={parent: after},
            original_input_proof=bound_proof("original"), replacement_input_proof=bound_proof("scalar"),
            original_projection=projection, replacement_projection=projection, parameters={},
            report_id="cancelled-report", report_digest="e" * 64, check_canceled=cancel)
    with store.read_transaction() as connection:
        assert connection.execute("SELECT 1 FROM sqlite_master WHERE name='input_proof_rebinding_plans'").fetchone() is None
    assert len(store.inventory()) == 3
