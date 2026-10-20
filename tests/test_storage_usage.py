"""Accounting is independent of schema, contents, and database initialization."""

import os

import pytest

from bagelquant_core import CoreStore
from bagelquant_core.storage_usage import file_inventory


def test_complete_inventory_counts_unknown_sidecars_sparse_links_and_hidden(tmp_path):
    meta = tmp_path / "meta.sqlite"
    meta.write_bytes(b"uninitialized evidence")
    (tmp_path / "meta.sqlite-wal").write_bytes(b"wal")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / ".unknown").write_bytes(b"unknown")
    os.link(meta, artifacts / "hardlink")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large").write_bytes(b"x" * 10_000)
    (artifacts / "link").symlink_to(outside, target_is_directory=True)
    with (artifacts / "sparse").open("wb") as stream:
        stream.truncate(10 * 1024 * 1024)
    usage = CoreStore(meta, artifacts).storage_usage()
    assert usage["file_count"] == 5
    assert not any(entry["path"].endswith("/large") for entry in usage["entries"])
    assert usage["bytes"] > usage["allocated_bytes"]
    assert len(usage["entries"]) == 7  # Includes the directory and hard-link alias.
    assert meta.read_bytes() == b"uninitialized evidence"


def test_missing_store_is_passive(tmp_path):
    store = CoreStore(tmp_path / "missing.sqlite", tmp_path / "missing")
    assert store.storage_usage()["bytes"] == 0
    assert list(tmp_path.iterdir()) == []


def test_inventory_surfaces_permission_errors(tmp_path, monkeypatch):
    original = os.scandir

    def denied(path):
        if str(path) == str(tmp_path):
            raise PermissionError("unreadable")
        return original(path)

    monkeypatch.setattr(os, "scandir", denied)
    with pytest.raises(PermissionError, match="unreadable"):
        file_inventory([tmp_path])


def test_exclusions_prevent_owner_directory_traversal(tmp_path, monkeypatch):
    excluded = tmp_path / "backend"
    excluded.mkdir()
    original = os.scandir

    def guarded(path):
        assert str(path) != str(excluded)
        return original(path)

    monkeypatch.setattr(os, "scandir", guarded)
    assert len(file_inventory([tmp_path], exclude=[excluded])) == 1


def test_selected_root_does_not_traverse_external_owned_artifacts(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    root.mkdir()
    meta = root / "core.sqlite"
    meta.write_bytes(b"metadata")
    external = tmp_path / "external"
    external.mkdir()
    original = os.scandir

    def guarded(path):
        assert str(path) != str(external)
        return original(path)

    monkeypatch.setattr(os, "scandir", guarded)
    usage = CoreStore(meta, external).storage_usage(within_root=root)
    assert usage["bytes"] == len(b"metadata")
    assert usage["file_count"] == 1
