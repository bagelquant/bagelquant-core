"""Indexed completion/candidate reads preserve public numerical authority."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
import sqlite3

import pytest

from bagelquant_core import CoreStore, NodeMaterialization
from bagelquant_core.hashing import hash_mapping
from bagelquant_core.operator import negate
from bagelquant_core.preparation import ExecutionPreparation, preparation_scope
from test_global_graph import finish, global_graph, source


def completed(tmp_path):
    value = source(32)
    graph, store, root = global_graph(tmp_path, negate(value))
    receipt = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    return graph, store, root, receipt


def test_normalized_updates_preserve_receipt_identity_and_empty_context(tmp_path, monkeypatch):
    graph, store, root, original = completed(tmp_path)
    state = graph._state()
    receipt = store.commit_update("global", revision=state["revision"], state_revision=state["state_revision"],
        expected_previous=state["current_update"], through=original["through"], mode="branch", request_id="empty-context",
        results={"main": original["results"]["main"], "empty": {}})
    assert store.describe_update(receipt["identity"]) == receipt
    assert store.find_update("global", "empty-context") == receipt
    assert hash_mapping({key: value for key, value in receipt.items() if key != "identity"}) == receipt["identity"]
    with sqlite3.connect(store.meta_path) as db:
        assert "receipt_json" not in {row[1] for row in db.execute("PRAGMA table_info(updates)")}
        assert db.execute("SELECT count(*) FROM update_contexts WHERE update_id=?", (receipt["identity"],)).fetchone()[0] == 2
    statements = []
    original_read = store.read_transaction

    @contextmanager
    def observed():
        with original_read() as db:
            db.set_trace_callback(statements.append)
            yield db

    monkeypatch.setattr(store, "read_transaction", observed)
    monkeypatch.setattr(store, "describe_update", lambda *_: pytest.fail("completion decoded an update receipt"))
    assert store.completed_materializations("global", "main", root) == (original["results"]["main"][root],)
    assert not any("json_each" in statement.lower() for statement in statements)


def test_candidate_filters_use_headers_without_manifest_decoding(tmp_path, monkeypatch):
    _, store, root, receipt = completed(tmp_path)
    result = store.read(receipt["results"]["main"][root])
    policy = store.manifest(result.key.identity)["execution_policy"]
    wrong = replace(result.key, context_identity="unrelated-policy")
    store.publish(NodeMaterialization(wrong, result.panel), execution_policy={**policy, "name": "unrelated"})
    policy_key = hash_mapping({key: value for key, value in policy.items() if key != "parameter_bindings"})
    monkeypatch.setattr(store, "manifest", lambda *_: pytest.fail("candidate decoded a complete manifest"))
    candidates = store.materialization_candidates(root, result.key.implementation_id, policy_key=policy_key)
    assert candidates
    assert wrong.identity not in {row["identity"] for row in candidates}
    assert all(row["policy_name"] == policy["name"] for row in candidates)
    with sqlite3.connect(store.meta_path) as db:
        plan = db.execute("EXPLAIN QUERY PLAN SELECT identity FROM materialization_headers "
            "WHERE node_id=? AND implementation_id=? AND policy_key=?", (root, result.key.implementation_id, policy_key)).fetchall()
        assert any("materialization_policy" in row[3] for row in plan)


def test_append_planners_do_not_scan_inventory(tmp_path, monkeypatch):
    graph, store, root, _ = completed(tmp_path)
    monkeypatch.setattr(store, "inventory", lambda **_: pytest.fail("incremental planner scanned all materializations"))
    value = source(33)
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    finish(plan)
    assert plan.resource_usage["main"][root]["computed_output_sessions"] == 1
    assert plan.resource_usage["main"][root]["kernel_sessions"] == 1


def test_finite_metadata_memo_is_shared_bounded_and_mutation_safe(tmp_path):
    _, store, _, receipt = completed(tmp_path)
    identity = next(iter(receipt["results"]["main"].values()))
    events = []
    with store.read_context(max_metadata_bytes=1024 * 1024, metadata_observer=lambda key, amount: events.append((key, amount))) as reader:
        first = reader.manifest(identity)
        first["through"] = "tampered"
        with ThreadPoolExecutor(max_workers=2) as pool:
            frames = list(pool.map(reader.manifest, [identity, identity]))
        assert all(frame["through"] == receipt["through"] for frame in frames)
        reader.interval_identity(identity, start="2024-01-01", end="2024-01-02")
        reader.interval_identity(identity, start="2024-01-01", end="2024-01-02")
        memo = reader._metadata_memo
        assert memo.bytes <= memo.maximum_bytes
    assert memo.closed and memo.bytes == 0
    assert sum(amount for key, amount in events if key == "manifest_reads") == 1
    assert sum(amount for key, amount in events if key == "interval_index_reads") == 1
    with store.read_context(max_metadata_bytes=0) as reader:
        reader.manifest(identity)
        assert reader._metadata_memo.bytes == 0


def test_composed_source_bindings_are_checked_once_per_finite_reader(tmp_path):
    _, store, _, receipt = completed(tmp_path)
    original = store.read(next(iter(receipt["results"]["main"].values())))
    original = NodeMaterialization(replace(original.key, context_identity="original-leaf"), original.panel)
    store.publish(original)
    identities = [original.key.identity]
    for index in range(12):
        key = replace(original.key, context_identity=f"composed-{index}")
        store.publish(NodeMaterialization(key, original.panel), value_references=(identities[-1],))
        identities.append(key.identity)
    counts = []
    for budget in (0, 1024 * 1024):
        events = []
        with store.read_context(max_metadata_bytes=budget,
                metadata_observer=lambda key, amount: events.extend([key] * amount)) as reader:
            for identity in reversed(identities):
                assert reader.describe(identity)["identity"] == identity
            counts.append(events.count("source_binding_checks"))
            assert reader._metadata_memo.bytes <= reader._metadata_memo.maximum_bytes
    assert counts == [sum(range(2, 14)), 13]


@pytest.mark.parametrize("scope", ["reader", "preparation"])
def test_source_binding_memo_never_bypasses_full_audit_or_publication(tmp_path, scope):
    _, store, _, receipt = completed(tmp_path)
    original = store.read(next(iter(receipt["results"]["main"].values())))
    parent_key = replace(original.key, context_identity="composed-parent")
    store.publish(NodeMaterialization(parent_key, original.panel), value_references=(original.key.identity,))
    tip_key = replace(original.key, context_identity="composed-tip")
    store.publish(NodeMaterialization(tip_key, original.panel), value_references=(parent_key.identity,))
    @contextmanager
    def reader_scope():
        if scope == "reader":
            with store.read_context() as reader:
                yield reader
        else:
            preparation = ExecutionPreparation(1024 * 1024)
            try:
                with preparation_scope(preparation):
                    yield store
            finally:
                preparation.clear()
    with reader_scope() as reader:
        reader.describe(tip_key.identity)
        with sqlite3.connect(store.meta_path) as db:
            manifest = json.loads(db.execute("SELECT manifest_json FROM materializations WHERE identity=?",
                (original.key.identity,)).fetchone()[0])
            manifest["through"] = "2099-01-01"
            db.execute("UPDATE materializations SET manifest_json=? WHERE identity=?",
                (json.dumps(manifest), original.key.identity))
        # Already admitted immutable descriptions stay local to this reader;
        # explicit audits and the final publication transaction remain fresh.
        assert reader.describe(tip_key.identity)["identity"] == tip_key.identity
        with pytest.raises(ValueError, match="corrupt Core manifest"):
            reader.verify_identity(tip_key.identity)
        new_key = replace(original.key, context_identity="rejected-composition")
        with pytest.raises(ValueError, match="corrupt Core manifest"):
            reader.publish(NodeMaterialization(new_key, original.panel), value_references=(tip_key.identity,))
        assert reader.manifest(new_key.identity) is None
    with store.read_context() as reader:
        with pytest.raises(ValueError, match="corrupt Core manifest"):
            reader.describe(tip_key.identity)


def test_corrupt_candidate_and_update_metadata_fail(tmp_path):
    _, store, root, receipt = completed(tmp_path)
    identity = receipt["results"]["main"][root]
    implementation = store.describe(identity)["key"]["implementation_id"]
    with sqlite3.connect(store.meta_path) as db:
        db.execute("UPDATE materialization_headers SET through='2099-01-01' WHERE identity=?", (identity,))
        db.execute("UPDATE updates SET through='2099-01-01' WHERE identity=?", (receipt["identity"],))
    with pytest.raises(ValueError, match="candidate header"):
        store.materialization_candidates(root, implementation)
    with pytest.raises(ValueError, match="receipt identity"):
        store.describe_update(receipt["identity"])


@pytest.mark.parametrize("version", [2, 3])
def test_older_schemas_are_rejected_without_migration(tmp_path, version):
    store = CoreStore(tmp_path / "old.sqlite", tmp_path / "values")
    with sqlite3.connect(store.meta_path) as db:
        db.execute("CREATE TABLE materializations(identity TEXT PRIMARY KEY, manifest_json TEXT)")
        db.execute(f"PRAGMA user_version={version}")
    original = store.meta_path.read_bytes()
    assert store.inspect()["status"] == "incompatible"
    with pytest.raises(ValueError, match="incompatible Core"):
        store.manifest("retained")
    assert store.meta_path.read_bytes() == original
