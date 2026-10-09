"""Public readiness inspection must never create or rewrite a store."""

from contextlib import closing
import sqlite3
import subprocess
import sys

import pytest

from bagelquant_core import CoreStore
from bagelquant_core.inspection import open_metadata_snapshot


def test_inspect_missing_store_is_read_only(tmp_path):
    store = CoreStore(tmp_path / "missing" / "meta.sqlite", tmp_path / "artifacts")
    assert store.inspect()["status"] == "uninitialized"
    assert list(tmp_path.iterdir()) == []


def test_inspect_ready_store_keeps_bytes_and_artifacts(tmp_path):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    before = store.meta_path.read_bytes()
    assert store.inspect() == {"status": "ready", "schema_version": 1, "reason": None}
    assert store.meta_path.read_bytes() == before
    assert list(store.artifact_path.iterdir()) == []


def test_inspect_incompatible_and_corrupt_store_preserves_evidence(tmp_path):
    path = tmp_path / "meta.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE old_evidence(value TEXT)")
    store = CoreStore(path, tmp_path / "artifacts")
    before = path.read_bytes()
    assert store.inspect()["status"] == "incompatible"
    assert path.read_bytes() == before
    path.write_bytes(b"retained corrupt evidence")
    assert store.inspect()["reason"] == "metadata_unreadable"
    assert path.read_bytes() == b"retained corrupt evidence"
    assert not store.artifact_path.exists()


def test_matching_version_with_incomplete_columns_is_incompatible(tmp_path):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with sqlite3.connect(store.meta_path) as connection:
        connection.execute(
            "ALTER TABLE contexts RENAME COLUMN context_id TO invalid_column"
        )
    before = store.meta_path.read_bytes()
    assert store.inspect() == {
        "status": "incompatible",
        "schema_version": 1,
        "reason": "schema_incompatible",
    }
    assert store.meta_path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 2])
def test_checkpointed_wal_inspection_creates_no_sidecars(tmp_path, version):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(f"PRAGMA user_version={version}")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert set(before) == {"meta.sqlite"}
    assert store.inspect()["status"] == ("ready" if version == 1 else "incompatible")
    assert {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    } == before


@pytest.mark.parametrize("base_version,version", [(1, 2), (2, 1)])
def test_active_wal_inspection_reads_committed_version_without_touching_index(
    tmp_path, base_version, version
):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        version_before = version
        version = base_version
        connection.execute(f"PRAGMA user_version={version}")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        version = version_before
        connection.execute(f"PRAGMA user_version={version}")
        connection.commit()
        before = {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        }
        assert store.inspect()["status"] == (
            "ready" if version == 1 else "incompatible"
        )
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before


@pytest.mark.parametrize("committed_version", [1, 2])
def test_active_rollback_journal_never_confers_uncommitted_readiness(
    tmp_path, committed_version
):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as connection:
        connection.execute("CREATE TABLE pressure(payload BLOB)")
        version = committed_version
        connection.execute(f"PRAGMA user_version={version}")
        connection.commit()
        connection.execute("PRAGMA cache_size=1")
        connection.execute("PRAGMA cache_spill=ON")
        connection.execute("BEGIN IMMEDIATE")
        version = 3 - committed_version
        connection.execute(f"PRAGMA user_version={version}")
        connection.executemany("INSERT INTO pressure VALUES(?)", [(b"x" * 65536,)] * 40)
        journal = store.meta_path.with_name(store.meta_path.name + "-journal")
        assert any(journal.read_bytes()[:8])
        before = {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        }
        assert store.inspect()["reason"] == "metadata_unreadable"
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before
        connection.rollback()
        assert store.inspect()["status"] == (
            "ready" if committed_version == 1 else "incompatible"
        )


def test_cold_persist_journal_is_readable_without_recovery(tmp_path):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as connection:
        connection.execute("PRAGMA journal_mode=PERSIST")
        connection.execute("CREATE TABLE retained(value TEXT)")
        connection.commit()
    journal = store.meta_path.with_name(store.meta_path.name + "-journal")
    assert journal.stat().st_size > 0 and not any(journal.read_bytes()[:8])
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert store.inspect()["status"] == "ready"
    assert {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    } == before


def test_runtime_inspection_reads_committed_wal_without_copying(tmp_path, monkeypatch):
    import bagelquant_core.inspection as inspection
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA user_version=2")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("PRAGMA user_version=1")
        writer.commit()
        before = store.meta_path.read_bytes()
        monkeypatch.setattr(inspection.shutil, "copyfile", lambda *args: pytest.fail("runtime copied metadata"))
        assert store.inspect(runtime=True) == {"status": "ready", "schema_version": 1, "reason": None}
        assert store.meta_path.read_bytes() == before
        assert list(store.artifact_path.iterdir()) == []


def test_runtime_read_view_stays_consistent_across_concurrent_wal_commit(tmp_path):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE marker(value INTEGER)")
        writer.execute("INSERT INTO marker VALUES(1)")
        writer.commit()
        with open_metadata_snapshot(store.meta_path, runtime=True) as reader:
            assert reader.in_transaction
            assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
            writer.execute("UPDATE marker SET value=2")
            writer.execute("PRAGMA user_version=2")
            writer.commit()
            assert reader.execute("SELECT value FROM marker").fetchone()[0] == 1
            assert reader.execute("PRAGMA user_version").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError):
                reader.execute("UPDATE marker SET value=3")
        assert store.inspect(runtime=True)["schema_version"] == 2


def test_runtime_allows_reserved_rollback_writer_but_never_uncommitted_schema(tmp_path):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("PRAGMA user_version=2")
        assert store.inspect(runtime=True)["status"] == "ready"
        writer.rollback()
        assert store.inspect()["status"] == "ready"


def test_runtime_hot_abandoned_journal_fails_closed_without_recovery(tmp_path):
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    script = """
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute('CREATE TABLE pressure(payload BLOB)')
db.commit()
db.execute('PRAGMA cache_size=1')
db.execute('PRAGMA cache_spill=ON')
db.execute('BEGIN IMMEDIATE')
db.execute('PRAGMA user_version=2')
db.executemany('INSERT INTO pressure VALUES(?)', [(b'x' * 65536,)] * 40)
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, str(store.meta_path)], check=True)
    journal = store.meta_path.with_name(store.meta_path.name + "-journal")
    assert any(journal.read_bytes()[:8])
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    assert store.inspect(runtime=True)["reason"] == "metadata_unreadable"
    assert store.inspect()["reason"] == "metadata_unreadable"
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


def test_runtime_missing_store_does_not_initialize(tmp_path):
    store = CoreStore(tmp_path / "missing" / "meta.sqlite", tmp_path / "artifacts")
    assert store.inspect(runtime=True)["status"] == "uninitialized"
    with pytest.raises(sqlite3.OperationalError):
        with open_metadata_snapshot(store.meta_path, runtime=True):
            pass
    assert list(tmp_path.iterdir()) == []
