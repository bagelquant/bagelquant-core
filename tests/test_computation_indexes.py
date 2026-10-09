"""Indexed interval proofs include the whole scoped Domain and typed channels."""
from datetime import timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

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
    updated = finish(plan)
    cold, fresh, cold_root = global_graph(tmp_path / "cold", ewm_mean(second, alpha=0.4))
    expected = finish(cold.plan_update({"main": {"x": second}}, through=second.domain.times.max(), input_proof=proof))
    assert plan.resource_usage["main"][root]["checkpoint_through"] is None
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True),
        fresh.read_values(expected["results"]["main"][cold_root], include_traces=True), check_exact=True)


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
    store.publish(NodeMaterialization(key, record.panel))
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
