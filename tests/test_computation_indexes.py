"""Indexed interval proofs include the whole scoped Domain and typed channels."""
from datetime import timedelta

import polars as pl
import pytest

from bagelquant_core import Domain, Node
from bagelquant_core import computation_index
from bagelquant_core.operator import ewm_mean
from test_global_graph import source, global_graph, finish
from test_selective_store_reads import fixture


def test_exact_update_hit_never_reads_parquet_schema_or_values(tmp_path, monkeypatch):
    x = source(35)
    graph, store, _ = global_graph(tmp_path, ewm_mean(x, alpha=0.4))
    first = finish(graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max())
    def fail(*args, **kwargs):
        pytest.fail("metadata hit read Parquet")
    with monkeypatch.context() as patch:
        patch.setattr(pl, "read_parquet", fail)
        patch.setattr(pl.LazyFrame, "collect_schema", fail)
        second = finish(plan)
    assert second["results"] == first["results"]
    store.verify_update(first["identity"])


@pytest.mark.parametrize("changed", ["membership", "calendar", "traces"])
def test_equal_external_content_token_cannot_hide_changed_prefix_domain(tmp_path, changed):
    def opaque(value, domain=None, traces=("available_date",)):
        frame = value.collect(dense=False, include_traces=True)
        if traces != ("available_date",):
            frame = frame.rename({"available_date": traces[0]})
        return Node.from_domain(frame, domain or value.domain, identity="external:constant",
            name="x", source_key="x", trace_columns=traces)
    first = opaque(source(35))
    graph, store, root = global_graph(tmp_path / "append", ewm_mean(first, alpha=0.4))
    def proof(*_):
        return "same-selected-data"
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max(), input_proof=proof))
    full = source(45)
    if changed == "membership":
        grid = full.domain.grid_lazy().collect().filter(~((pl.col("asset_id") == "b") & (pl.col("time") <= first.domain.times.max())))
        second = opaque(full, Domain(calendar=full.domain.times, universe=grid.with_columns(pl.lit(True).alias("active"))))
    elif changed == "calendar":
        calendar = full.domain.times.filter(full.domain.times != full.domain.times[0] + timedelta(days=10))
        second = opaque(full, Domain(calendar=calendar, universe=["a", "b"]))
    else:
        second = opaque(full, traces=("other_available",))
    plan = graph.plan_update({"main": {"x": second}}, through=second.domain.times.max(), input_proof=proof)
    with pytest.raises(ValueError, match="proven preceding state"):
        finish(plan)


def test_index_missing_is_unknown_and_explicit_build_does_not_rewrite_receipt(tmp_path):
    store, record, days = fixture(tmp_path, dynamic=True)
    original = store.manifest(record.key.identity)
    with store.transaction() as db:
        db.execute("DELETE FROM computation_index")
    assert store.interval_identity(record.key.identity, start=days[0], end=days[-1]) is None
    assert store.build_index(store.index_plan())["status"] == "complete"
    assert store.manifest(record.key.identity) == original
    assert store.interval_identity(record.key.identity, start=days[0], end=days[-1]) is not None
    store.verify_identity(record.key.identity)


@pytest.mark.parametrize("missing", ["row", "version"])
def test_domain_publication_uses_existing_indexed_evidence_without_warming_older_receipts(tmp_path, monkeypatch, missing):
    from dataclasses import replace
    import bagelquant_core.store as owner
    from bagelquant_core import NodeMaterialization
    store, record, days = fixture(tmp_path, dynamic=True)
    old = store.manifest(record.key.identity)
    second_key = replace(record.key, inputs=("another-original-source",))
    store.publish(NodeMaterialization(second_key, record.panel))
    second = store.manifest(second_key.identity)
    with store.transaction() as db:
        if missing == "row":
            db.execute("DELETE FROM computation_index WHERE identity=?", (record.key.identity,))
        else:
            db.execute("UPDATE computation_index SET version='old-index' WHERE identity=?", (record.key.identity,))
    def forbidden(*args, **kwargs):
        pytest.fail("record-backed Domain publication audited or warmed old history")
    monkeypatch.setattr(owner, "_checksum", forbidden)
    monkeypatch.setattr(store, "build_index", forbidden)
    new_key = replace(record.key, inputs=("reference-publication",))
    store.publish(NodeMaterialization(new_key, record.panel), value_references=(second_key.identity,))
    result = store.manifest(new_key.identity)
    assert result["partitions"] == second["partitions"]
    assert result["domain"] == second["domain"]
    assert store.manifest(record.key.identity) == old
    assert store.interval_identity(record.key.identity, start=days[0], end=days[-1]) is None
    assert store.interval_identity(new_key.identity, start=days[0], end=days[-1]) == store.interval_identity(
        second_key.identity, start=days[0], end=days[-1])


def test_full_audit_rejects_self_consistent_index_change(tmp_path):
    store, record, _ = fixture(tmp_path)
    manifest = store.manifest(record.key.identity)
    with store.transaction() as db:
        indexed = computation_index.read(db, manifest)
        indexed["calendar"] = []
        computation_index.save(db, record.key.identity, manifest["content_identity"], indexed)
    with pytest.raises(ValueError, match="index differs"):
        store.verify_identity(record.key.identity)
    assert store.build_index(store.index_plan())["status"] == "complete"
    store.verify_identity(record.key.identity)


def test_new_publication_indexes_only_new_result_in_index_free_store(tmp_path):
    from bagelquant_core import MaterializationKey, NodeMaterialization
    store, record, days = fixture(tmp_path)
    with store.transaction() as db:
        db.execute("DROP TABLE computation_index")
    store.initialize()
    assert store.interval_identity(record.key.identity, start=days[0], end=days[-1]) is None
    key = MaterializationKey(record.key.node_id, record.key.implementation_id,
        ("new-input",), record.key.domain_identity)
    with pytest.raises(ValueError, match="Domain identity needs explicit"):
        store.publish(NodeMaterialization(key, record.panel))
    from bagelquant_core import Domain, Node
    domain = Domain(calendar=days, universe=["different-asset"])
    panel = Node.from_domain(pl.DataFrame({"time": days, "asset_id": ["different-asset"]*len(days),
        "value": [1.0]*len(days)}), domain)
    key = MaterializationKey(record.key.node_id, record.key.implementation_id,
        ("new-input",), domain.signature)
    store.publish(NodeMaterialization(key, panel))
    assert store.interval_identity(key.identity, start=days[0], end=days[-1]) is not None
    assert store.interval_identity(record.key.identity, start=days[0], end=days[-1]) is None


def test_empty_index_plan_is_passive_on_uninitialized_store(tmp_path):
    from bagelquant_core import CoreStore
    store = CoreStore(tmp_path / "new" / "core.sqlite", tmp_path / "new" / "values")
    assert store.index_plan()["identities"] == []
    assert store.build_index(store.index_plan())["materializations"] == 0
    assert not store.meta_path.parent.exists()


def test_index_maintenance_admission_returns_partial_before_full_decode(tmp_path, monkeypatch):
    import bagelquant_core.resources as resources
    store, _, _ = fixture(tmp_path)
    with store.transaction() as db:
        db.execute("DELETE FROM computation_index")
    monkeypatch.setattr(resources, "admit_parquet_materialization", lambda _: False)
    monkeypatch.setattr(pl, "read_parquet", lambda *_a, **_k: pytest.fail("unadmitted Domain decoded"))
    result = store.build_index(store.index_plan())
    assert result["status"] == "partial"
    assert result["materializations"] == 0 and result["unknown_materializations"] == 1
def test_source_proof_memo_distinguishes_actual_domains_with_same_opaque_identity(monkeypatch):
    from datetime import date, datetime
    from types import SimpleNamespace
    from bagelquant_core import Domain, Node
    from bagelquant_core.update import UpdatePlan
    from bagelquant_core.preparation import ExecutionPreparation
    from bagelquant_core.computation_index import describe, interval
    from bagelquant_core.hashing import hash_mapping

    days = [date(2024, 1, 2), date(2024, 1, 3)]
    frame = pl.DataFrame({"time": days, "asset_id": ["A", "A"], "value": [1., 2.]})
    values = [Node.from_domain(frame, Domain(calendar=days, universe=assets, identity="same-opaque-id"))
        for assets in (["A"], ["A", "B"])]
    def forbidden(*args, **kwargs):
        pytest.fail("source proof read numerical values")
    monkeypatch.setattr(Node, "collect", forbidden)
    expected = [hash_mapping({"source": "token", "domain": interval(
        describe(value.domain.window(*days), days={}, value_type=value.value_type,
            trace_columns=value.trace_columns), *days)}) for value in values]
    assert expected[0] != expected[1]
    plan = object.__new__(UpdatePlan)
    plan.store = SimpleNamespace(interval_identity=lambda *a, **k: None)
    plan.spec = SimpleNamespace(nodes=[SimpleNamespace(node_id="source", node_type="input", input_key="x")])
    plan._input_proof = lambda *a: "token"
    plan._preparation = ExecutionPreparation(1024 * 1024)
    for context, value, token in zip(("first", "second"), values, expected, strict=True):
        assert value.source_interval_identity("token", *days) == token
        assert plan._proof_reader(context)("source", value, *days) == token
    for first, last in ((str(days[0]), str(days[1])), (datetime(2024, 1, 2), days[0])):
        with pytest.raises(ValueError, match="Date bounds"):
            values[0].source_interval_identity("token", first, last)
    with pytest.raises(ValueError, match="end precedes"):
        values[0].source_interval_identity("token", days[1], days[0])
