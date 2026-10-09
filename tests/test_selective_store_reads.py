"""Published metadata, finite byte proofs and selected typed value reads."""
from datetime import date
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import polars as pl
import pytest

from bagelquant_core import CoreStore, Domain, MaterializationKey, Node, NodeMaterialization
import bagelquant_core.store as storage


def fixture(tmp_path, *, dynamic=False):
    days = [date(2024, month, 2) for month in range(1, 7)]
    universe = pl.DataFrame({"time": days, "asset_id": ["A"] * 6,
        "active": [True, False, True, True, False, True]}) if dynamic else ["A", "B"]
    domain = Domain(calendar=days, universe=universe)
    frame = domain.grid_lazy().with_columns(pl.lit(1.0).alias("value"),
        pl.col("time").alias("available")).collect()
    key = MaterializationKey("fixture", "operator.v1", ("source",), domain.signature)
    record = NodeMaterialization(key, Node.from_domain(frame, domain,
        value_type="prediction", trace_columns=("available",)), artifacts={"diagnostic": frame})
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    store.publish(record)
    return store, record, days


@pytest.mark.parametrize("dynamic", [False, True])
def test_window_hashes_only_domain_and_selected_partition(tmp_path, monkeypatch, dynamic):
    store, record, days = fixture(tmp_path, dynamic=dynamic)
    hashes = []
    original = storage._checksum
    def checksum(path):
        hashes.append(path)
        return original(path)
    monkeypatch.setattr(storage, "_checksum", checksum)
    assert store.describe(record.key.identity)["value_type"] == "prediction"
    assert store.describe_evidence(record.key.identity)["artifacts"] == ("diagnostic",)
    assert hashes == []
    result = store.read_values(record.key.identity, start=days[-1], end=days[-1], include_traces=True)
    assert len(hashes) == 0
    assert result.equals(record.panel.collect(dense=False, include_traces=True).filter(pl.col("time") == days[-1]))
    manifest = store.manifest(record.key.identity)
    store._file(manifest["partitions"][0]["$file"]).write_bytes(b"damage")
    assert store.read_values(record.key.identity, start=days[-1], end=days[-1]).height > 0
    assert store.check_integrity()[0]["valid"] is False
    with pytest.raises(ValueError, match="artifact"):
        store.read_values(record.key.identity, start=days[0], end=days[0])


def test_proof_deduplicates_threads_and_expires_after_context(tmp_path, monkeypatch):
    store, record, days = fixture(tmp_path)
    hashes = []
    original = storage._checksum
    def checksum(path):
        hashes.append(path)
        return original(path)
    monkeypatch.setattr(storage, "_checksum", checksum)
    with store.read_context() as reader:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: reader.read_values(record.key.identity, start=days[-1], end=days[-1]), range(3)))
        assert len(hashes) == 0
    reader.read_values(record.key.identity, start=days[-1], end=days[-1])
    assert len(hashes) == 0
    with store.read_context() as reader:
        reader.verify_identity(record.key.identity)
        manifest = reader.manifest(record.key.identity)
        path = reader._file(manifest["partitions"][-1]["$file"])
        original_bytes = path.read_bytes()
        path.write_bytes(b"x" * len(original_bytes))
        with pytest.raises(ValueError, match="artifact"):
            reader.read_values(record.key.identity, start=days[-1], end=days[-1])


def test_graph_reads_do_not_initialize_and_wal_reader_sees_committed_state(tmp_path):
    store = CoreStore(tmp_path / "new" / "core.sqlite", tmp_path / "new" / "artifacts")
    assert store.graph_state("graph")["revision"] == 0
    assert not store.meta_path.parent.exists()
    store.register_context("graph", "context", {"anchor": "2024-01-02"})
    with sqlite3.connect(store.meta_path) as writer:
        assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE graphs SET revision=99 WHERE graph_id='graph'")
        assert store.graph_state("graph")["revision"] == 1
        with store.read_transaction() as reader:
            assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError):
                reader.execute("UPDATE graphs SET revision=2")
        writer.rollback()


def test_empty_window_rechecks_schema_file_after_read(tmp_path, monkeypatch):
    store, record, _days = fixture(tmp_path)
    original = pl.read_parquet_schema
    def changing_schema(path, **kwargs):
        schema = original(path, **kwargs)
        path.write_bytes(b"changed after reading schema")
        return schema
    monkeypatch.setattr(pl, "read_parquet_schema", changing_schema)
    assert store.read_values(record.key.identity, start=date(2025, 1, 1)).is_empty()
    with pytest.raises(ValueError, match="artifact"):
        store.verify_identity(record.key.identity)


def test_auxiliary_evidence_rechecks_consumed_file_after_decode(tmp_path, monkeypatch):
    store, record, _days = fixture(tmp_path)
    manifest = store.manifest(record.key.identity)
    reference = manifest["artifacts"]["diagnostic"]
    original = CoreStore._decode
    def changing_decode(reader, value):
        result = original(reader, value)
        if value == reference:
            reader._file(reference["$file"]).write_bytes(b"changed after decoding")
        return result
    monkeypatch.setattr(CoreStore, "_decode", changing_decode)
    assert store.evidence(record.key.identity)["artifacts"]["diagnostic"].height > 0
    with pytest.raises(ValueError, match="artifact"):
        store.verify_identity(record.key.identity)
