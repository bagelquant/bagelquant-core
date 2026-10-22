"""Retained generic financial-change formulas gain finite proofs, without replay."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import ConsumedInputEvidence, Domain, ExecutionRuntime, Graph, Node, MaterializationKey
from bagelquant_core.hashing import hash_mapping
from bagelquant_core.operator import diff_from_last_change, pct_change_from_last_change, ffill, bfill
from test_global_graph import global_graph, finish, source


@pytest.mark.parametrize("operation", [diff_from_last_change, pct_change_from_last_change])
def test_adjacent_change_declares_one_observation(operation):
    assert operation.history({}) == 1


@pytest.mark.parametrize("limit,history", [(None, None), (0, 0), (1, 1), (240, 240), (True, None), (-1, None)])
def test_only_bounded_forward_fill_has_finite_history(limit, history):
    assert ffill.history({"limit": limit}) == history
    assert bfill.history({"limit": limit}) is None


def financial_source(count, *, dynamic=False, revision=None):
    value = source(count)
    days = value.domain.times.to_list()
    frame = value.collect(include_traces=True).with_columns(
        pl.when(pl.col("time").is_in([days[i] for i in range(8, count, 19)]))
        .then(None).otherwise((pl.col("time").cast(pl.Int32) // 7 % 5).cast(pl.Float64)).alias("value"))
    if revision is not None:
        frame = frame.with_columns(pl.when(pl.col("time") == days[revision]).then(12.0)
            .otherwise(pl.col("value")).alias("value"))
    domain = value.domain
    if dynamic:
        membership = frame.select("time", "asset_id").filter(~((pl.col("asset_id") == "b")
            & pl.col("time").is_between(days[4], days[min(55, count - 1)])))
        domain = Domain(calendar=days, universe=membership.with_columns(pl.lit(True).alias("active")))
    return Node.from_domain(frame, domain, source_key="x", trace_columns=value.trace_columns)


def formula(value, operation="pct_change_from_last_change", limit=5):
    return Graph.from_dsl(f"p = {operation}(x)\nf = ffill(p, limit={limit})\noutput = rankpct(f)", inputs={"x": value})


def projection(_parent, value, first, last):
    return ConsumedInputEvidence("original-source", "a" * 64, value.interval_identity(first, last),
        first, last, value.domain.window(first, last).signature, value.value_type.value, value.trace_columns)


def initial(tmp_path, monkeypatch, *, dynamic=False, operation="pct_change_from_last_change", limit=5, cutoff=False):
    before = financial_source(63, dynamic=dynamic)
    graph, store, root = global_graph(tmp_path, formula(before, operation, limit))
    if cutoff:
        graph.register_context("main", anchor=str(before.domain.times.min()),
            information_cutoff=str(before.domain.times.max()))
    with monkeypatch.context() as old:
        for operator in (pct_change_from_last_change, diff_from_last_change, ffill):
            old.setattr(operator, "contract", replace(operator.contract, history=lambda _: None))
        receipt = finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max()))
    return before, graph, store, root, receipt


def prepare_all(before, graph, store, receipt, *, callback=projection):
    state = graph.snapshot()
    saved = receipt["results"]["main"]
    plans = []
    for node in state["spec"].nodes:
        if node.node_type != "operator" or not node.operator.endswith(("last_change", ".ffill")):
            continue
        originals = {parent: before if parent not in saved else store.lookup(
            MaterializationKey.from_dict(store.describe(saved[parent])["key"]), evidence=False).panel
            for parent in node.inputs}
        plans.append(store.plan_input_proof_rebind(state["spec"], saved[node.node_id], mode="finite_contract.v1",
            original_update=receipt["identity"], context_id="main", original_context=state["contexts"]["main"],
            original_parents=originals, original_projection=callback,
            report_id="finite-contract-test", report_digest="b" * 64))
    return plans


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("operation,limit", [("pct_change_from_last_change", 5), ("diff_from_last_change", 0),
                                           ("pct_change_from_last_change", 240)])
def test_maintenance_preserves_originals_then_appends_only_missing_sessions(tmp_path, monkeypatch, dynamic, operation, limit):
    before, graph, store, root, old = initial(tmp_path / "saved", monkeypatch,
        dynamic=dynamic, operation=operation, limit=limit)
    originals = {identity: store.manifest(identity) for identity in old["results"]["main"].values()}
    files = {part["$file"]: store._file(part["$file"]).read_bytes()
        for item in originals.values() for part in item["partitions"]}
    current = financial_source(65, dynamic=dynamic)
    def proof(_context, _key, first, last):
        return current.interval_identity(first, last)
    with pytest.raises(ValueError, match="policy-proof maintenance"):
        finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(), input_proof=proof))
    plans = prepare_all(before, graph, store, old)
    def forbidden(*_a, **_k):
        pytest.fail("maintenance read numerical values or ran a kernel")
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", forbidden)
        patch.setattr(Node, "collect", forbidden)
        patch.setattr(type(store), "read_values", forbidden)
        patch.setattr("bagelquant_core.store._checksum", forbidden)
        for plan in plans:
            assert store.apply_input_proof_rebind(plan)["computed_sessions"] == 0
            assert store.apply_input_proof_rebind(plan)["computed_sessions"] == 0
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(), input_proof=proof)
    updated = finish(plan)
    assert all(usage["computed_output_sessions"] == 2 for usage in plan.resource_usage["main"].values())
    assert all(store.manifest(identity) == item for identity, item in originals.items())
    assert all(store._file(path).read_bytes() == payload for path, payload in files.items())
    reference, fresh, full_root = global_graph(tmp_path / "full", formula(current, operation, limit))
    full = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max(), input_proof=proof))
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True),
        fresh.read_values(full["results"]["main"][full_root], include_traces=True), check_exact=True)
    # The ordinary full-domain formula is also an independent numerical reference.
    expected = formula(current, operation, limit).compute().collect(include_traces=True)
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True), expected, check_exact=True)
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", forbidden)
        repeated = finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(), input_proof=proof))
        assert repeated["results"] == updated["results"]
    store.verify_update(updated["identity"])


def test_local_revision_reuses_unaffected_blocks(tmp_path):
    first = financial_source(130)
    graph, store, root = global_graph(tmp_path / "saved", formula(first))
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    current = financial_source(130, revision=70)
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max())
    updated = finish(plan)
    assert all(usage["computed_output_sessions"] <= 64 for usage in plan.resource_usage["main"].values())
    assert all(usage["reused_blocks"] >= 3 for usage in plan.resource_usage["main"].values())
    expected = formula(current).compute().collect(include_traces=True)
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True), expected, check_exact=True)


def test_certified_history_survives_a_later_information_cutoff(tmp_path, monkeypatch):
    before, graph, store, root, old = initial(tmp_path, monkeypatch, cutoff=True)
    for admission in prepare_all(before, graph, store, old):
        store.apply_input_proof_rebind(admission)
    current = financial_source(65)
    graph.register_context("main", anchor=str(before.domain.times.min()),
        information_cutoff=str(current.domain.times.max()))
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
        input_proof=lambda _context, _key, first, last: current.interval_identity(first, last))
    updated = finish(plan)
    assert all(usage["computed_output_sessions"] == 2 for usage in plan.resource_usage["main"].values())
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True),
        formula(current).compute().collect(include_traces=True), check_exact=True)


@pytest.mark.parametrize("problem", ["source", "unknown", "traces", "context", "update", "domain"])
def test_admission_rejects_unproven_originals(tmp_path, monkeypatch, problem):
    before, graph, store, _root, old = initial(tmp_path, monkeypatch)
    state = graph.snapshot()
    node = next(n for n in state["spec"].nodes if n.operator and n.operator.endswith("last_change"))
    context = state["contexts"]["main"]
    if problem == "context":
        context = {**context, "information_cutoff": str(before.domain.times.max() + timedelta(days=1))}
    if problem in {"source", "domain"}:
        before = financial_source(63, dynamic=problem == "domain", revision=2 if problem == "source" else None)
    callback = (lambda *_: None) if problem == "unknown" else projection
    if problem == "traces":
        def callback(*args):
            return replace(projection(*args), trace_columns=())
    count = len(store.inventory())
    with pytest.raises((ValueError, KeyError)):
        store.plan_input_proof_rebind(state["spec"], old["results"]["main"][node.node_id], mode="finite_contract.v1",
            original_update="missing" if problem == "update" else old["identity"], context_id="main",
            original_context=context, original_parents={node.inputs[0]: before}, original_projection=callback,
            report_id="test", report_digest="b" * 64)
    assert len(store.inventory()) == count


def test_cancellation_tampering_and_original_loss_never_authorize_replay(tmp_path, monkeypatch):
    before, graph, store, _root, old = initial(tmp_path, monkeypatch)
    plans = prepare_all(before, graph, store, old)
    plan = plans[0]
    forged = deepcopy(plan)
    forged["blocks"][0]["key"]["inputs"] = ["forged"]
    forged["identity"] = hash_mapping({k: v for k, v in forged.items() if k != "identity"})
    with pytest.raises(ValueError, match="admission"):
        store.apply_input_proof_rebind(forged)
    count = len(store.inventory())
    def cancel():
        if len(store.inventory()) > count:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        store.apply_input_proof_rebind(plan, check_canceled=cancel)
    current = financial_source(65)
    with pytest.raises(ValueError, match="policy-proof maintenance"):
        finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    result = store.apply_input_proof_rebind(plan)
    with store.transaction() as connection:
        connection.execute("DELETE FROM input_proof_rebinding_plans WHERE identity=?", (plan["identity"],))
    with pytest.raises(ValueError, match="admission"):
        store.describe(result["identity"])


@pytest.mark.parametrize("problem", ["original_index", "parent_index", "block_index", "block", "original_corrupt", "update"])
def test_missing_retained_authority_blocks_before_any_kernel(tmp_path, monkeypatch, problem):
    before, graph, store, _root, old = initial(tmp_path, monkeypatch)
    plans = prepare_all(before, graph, store, old)
    for plan in plans:
        store.apply_input_proof_rebind(plan)
    plan = plans[-1]
    block = MaterializationKey.from_dict(plan["blocks"][0]["key"]).identity
    with store.transaction() as connection:
        if problem.endswith("index"):
            identity = {"original_index": plan["original_identity"],
                "parent_index": next(iter(plan["parents"].values()))["identity"], "block_index": block}[problem]
            connection.execute("DELETE FROM computation_index WHERE identity=?", (identity,))
        elif problem == "update":
            connection.execute("DELETE FROM update_results WHERE update_id=?", (old["identity"],))
        elif problem == "original_corrupt":
            connection.execute("UPDATE materializations SET manifest_json='{}' WHERE identity=?", (plan["original_identity"],))
        else:
            identity = block if problem == "block" else plan["original_identity"]
            connection.execute("DELETE FROM materialization_headers WHERE identity=?", (identity,))
            connection.execute("DELETE FROM materializations WHERE identity=?", (identity,))
    def forbidden(*_args, **_kwargs):
        pytest.fail("missing maintenance authority allowed a numerical kernel")
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    with pytest.raises((ValueError, KeyError)):
        store.describe(MaterializationKey.from_dict(plan["replacement_key"]).identity)
    current = financial_source(65)
    with pytest.raises((ValueError, KeyError)):
        finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max()))


def test_maintenance_retains_finite_parent_block_authority(tmp_path, monkeypatch):
    before = financial_source(63)
    expression = Graph.from_dsl("a = x + x\noutput = pct_change_from_last_change(a)", inputs={"x": before})
    graph, store, _root = global_graph(tmp_path, expression)
    with monkeypatch.context() as old:
        old.setattr(pct_change_from_last_change, "contract",
            replace(pct_change_from_last_change.contract, history=lambda _: None))
        receipt = finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max()))
    plan, = prepare_all(before, graph, store, receipt)
    maintained = store.apply_input_proof_rebind(plan)
    parent = next(iter(plan["parents"].values()))["identity"]
    block = store.manifest(parent)["value_sources"][0]["identity"]
    with store.transaction() as connection:
        connection.execute("DELETE FROM materialization_headers WHERE identity=?", (block,))
        connection.execute("DELETE FROM materializations WHERE identity=?", (block,))
    for verify in (store.describe, store.verify_identity):
        with pytest.raises(ValueError, match="original value authority is missing"):
            verify(maintained["identity"])


def test_admission_checks_scale_with_unique_authorities_and_keep_proof_ancestry(tmp_path, monkeypatch):
    from collections import Counter
    from bagelquant_core import finite_contract_maintenance as maintenance
    from bagelquant_core.metadata_memo import MetadataMemo
    before = financial_source(321)
    expression = Graph.from_dsl("a = x + x\noutput = pct_change_from_last_change(a)", inputs={"x": before})
    graph, store, _ = global_graph(tmp_path, expression)
    with monkeypatch.context() as old:
        old.setattr(pct_change_from_last_change, "contract",
            replace(pct_change_from_last_change.contract, history=lambda _: None))
        receipt = finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max()))
    admission, = prepare_all(before, graph, store, receipt)
    maintained = store.apply_input_proof_rebind(admission)
    manifest = store.manifest(maintained["identity"])
    calls = Counter()
    verify = maintenance._verify_original_authority

    def counted(*args):
        calls[args[2]["identity"]] += 1
        return verify(*args)

    monkeypatch.setattr(maintenance, "_verify_original_authority", counted)
    memo = MetadataMemo(32 * 1024 * 1024)
    parent = next(iter(admission["parents"].values()))["identity"]
    with store.read_transaction() as connection:
        store._verify_reference_sources(connection, manifest, memo=memo)
        assert calls == {admission["identity"]: 1}
        for index in range(4):
            memo.put(("manifest", f"large-{index}"), "body", 3 * 1024 * 1024)
            store._verify_reference_sources(connection, manifest, memo=memo)
        assert calls == {admission["identity"]: 1}
        for selected_memo in (None, memo):
            with pytest.raises(ValueError, match="cyclic|cycle"):
                store._verify_reference_sources(connection, manifest, visiting={parent}, memo=selected_memo)
    memo.close()


def test_partial_admission_cannot_cache_complete_block_authority(tmp_path, monkeypatch):
    from bagelquant_core.finite_contract_maintenance import verify_maintenance
    from bagelquant_core.metadata_memo import MetadataMemo
    before, graph, store, _, receipt = initial(tmp_path, monkeypatch)
    admission = prepare_all(before, graph, store, receipt)[0]
    store.apply_input_proof_rebind(admission)
    memo = MetadataMemo(8 * 1024 * 1024)
    key = MaterializationKey.from_dict(admission["replacement_key"])
    block = MaterializationKey.from_dict(admission["blocks"][0]["key"]).identity
    with store.transaction() as connection:
        verify_maintenance(type(store), connection, admission, key, admission["policy"], require_blocks=False, memo=memo)
        connection.execute("DELETE FROM computation_index WHERE identity=?", (block,))
        with pytest.raises(ValueError, match="canonical block index authority"):
            verify_maintenance(type(store), connection, admission, key, admission["policy"], memo=memo)
    memo.close()


def test_warm_reader_cannot_supply_closing_or_audit_authority(tmp_path, monkeypatch):
    from bagelquant_core.materialization import NodeMaterialization
    before, graph, store, _, receipt = initial(tmp_path, monkeypatch)
    admission = prepare_all(before, graph, store, receipt)[0]
    result = store.apply_input_proof_rebind(admission)
    key = MaterializationKey.from_dict(admission["replacement_key"])
    with store.read_context(max_metadata_bytes=32 * 1024 * 1024) as reader:
        saved = reader.lookup(key, evidence=False)
        assert reader.describe(result["identity"])
        with store.transaction() as connection:
            connection.execute("DELETE FROM computation_index WHERE identity=?", (admission["original_identity"],))
        with pytest.raises(ValueError, match="original result index authority"):
            reader.verify_identity(key.identity)
        closing = replace(key, context_identity="fresh-publication")
        with pytest.raises(ValueError, match="original result index authority"):
            reader.publish(NodeMaterialization(closing, saved.panel), execution_policy=admission["policy"],
                value_references=(key.identity,))
        state = graph.snapshot()
        with pytest.raises(ValueError, match="original result index authority"):
            reader.commit_update(graph.graph_id, revision=state["revision"], state_revision=state["state_revision"],
                expected_previous=state["current_update"], results={"main": {key.node_id: key.identity}},
                through=str(before.domain.times.max()), mode="branch")
        assert store.graph_state(graph.graph_id)["current_update"] == receipt["identity"]


@pytest.mark.parametrize("field", ["node", "implementation", "type", "traces"])
def test_compact_value_authority_preserves_the_consumers_typed_contract(tmp_path, monkeypatch, field):
    from copy import deepcopy
    from bagelquant_core.metadata_memo import MetadataMemo
    before, graph, store, _, receipt = initial(tmp_path, monkeypatch)
    admission = prepare_all(before, graph, store, receipt)[0]
    identity = store.apply_input_proof_rebind(admission)["identity"]
    manifest = store.manifest(identity)
    memo = MetadataMemo(8 * 1024 * 1024)
    with store.read_transaction() as connection:
        store._verify_reference_sources(connection, manifest, memo=memo)
        changed = deepcopy(manifest)
        changed["identity"] = "new-consumer"
        changed.pop("proof_rebinding")
        if field == "node":
            changed["node_id"] = "other"
        elif field == "implementation":
            changed["key"]["implementation_id"] = "other"
        elif field == "type":
            changed["value_type"] = "category"
        else:
            changed["trace_columns"] = []
        for selected in (None, memo):
            with pytest.raises(ValueError, match="original value authority differs"):
                store._verify_reference_sources(connection, changed, memo=selected)
    memo.close()
