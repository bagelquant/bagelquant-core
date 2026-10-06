"""Storage acceptance at its backend owner; no application package is installed."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path
import shutil

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import CoreStore, Domain, MaterializationKey, Node, NodeMaterialization


def value(domain, *, node_id="node", prediction=False, sparse=False):
    frame = domain.grid_lazy().with_columns(pl.lit(1.).alias("value"), pl.col("time").alias("available")).collect()
    if sparse:
        frame = frame.filter(pl.col("time").dt.month() != 2)
    key = MaterializationKey(node_id, "operator.v1", ("immutable-source",), domain.signature)
    node = Node.from_domain(frame, domain, trace_columns=("available",), value_type="prediction" if prediction else "numeric")
    return NodeMaterialization(key, node, artifacts={"diagnostic": frame}, checkpoint={"through": domain.times.max()},
        training_audits=({"fit_date": domain.times.min(), "model": "fixture"},))


@pytest.mark.parametrize("sparse", [False, True])
def test_bounded_monthly_publication_restart_and_all_evidence(tmp_path, monkeypatch, sparse):
    store = CoreStore(tmp_path/"core.sqlite", tmp_path/"artifacts")
    domain = Domain(calendar=[date(2024, 1, 2), date(2024, 2, 1), date(2024, 3, 1)], universe=["A", "B"])
    record = value(domain, prediction=True, sparse=sparse)
    with monkeypatch.context() as patch:
        patch.setattr(Node, "collect", lambda *_a, **_k: pytest.fail("publisher collected the entire Node"))
        store.publish(record)
        store.publish(record)
    manifest = store.manifest(record.key.identity)
    assert [part["rows"] for part in manifest["partitions"]] == ([2, 0, 2] if sparse else [2, 2, 2])
    restarted = CoreStore(store.meta_path, store.artifact_path)
    saved = restarted.read(record.key.identity)
    assert saved.panel.value_type == "prediction"
    assert_frame_equal(saved.panel.collect(dense=False, include_traces=True), record.panel.collect(dense=False, include_traces=True))
    assert saved.checkpoint == record.checkpoint
    assert saved.training_audits == record.training_audits
    assert_frame_equal(saved.artifacts["diagnostic"], record.artifacts["diagnostic"])
    assert restarted.check_integrity() == ({"identity": record.key.identity, "valid": True},)
    # Integrity verification still reads authoritative bytes on a repeated hit.
    member = store._file(manifest["domain"]["membership"]["$file"])
    member.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        restarted.read(record.key.identity)


def test_concurrent_publication_keeps_domains_and_evidence_distinct(tmp_path):
    store = CoreStore(tmp_path/"core.sqlite", tmp_path/"artifacts")
    domains = [Domain(calendar=[date(2024, 1, 2)], universe=[asset]) for asset in ("A", "B")]
    records = [value(domains[i % 2], node_id=str(i)) for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(store.publish, records))
    for record in records:
        saved = store.read(record.key.identity)
        assert saved.panel.domain.membership.equals(record.panel.domain.membership)
    with pytest.raises(ValueError, match="conflict"):
        store.publish(replace(records[0], checkpoint={"through": date(2030, 1, 1)}))


def orphan(store, identity="a"*64):
    directory = store.artifact_path/"values"/identity/("b"*32)
    directory.mkdir(parents=True)
    for i in range(2):
        pl.DataFrame({"value": [float(i)]}).write_parquet(directory/f"{i:06}.parquet")
    return directory


def test_cleanup_preserves_receipts_and_rejects_unapproved_members(tmp_path):
    store = CoreStore(tmp_path/"core.sqlite", tmp_path/"artifacts")
    record = value(Domain(calendar=[date(2024, 1, 2)], universe=["A"]))
    store.publish(record)
    directory = orphan(store)
    plan = store.cleanup_plan()
    assert len(plan["candidates"]) == 1
    assert plan["candidates"][0]["token"] == directory.relative_to(store.artifact_path).as_posix()
    extra = directory/"new-directory"
    extra.mkdir()
    with pytest.raises((ValueError, RuntimeError), match="changed|member"):
        store.apply_cleanup(plan)
    extra.rmdir()
    result = store.apply_cleanup(plan)
    assert store.apply_cleanup(plan) == result
    assert not directory.exists()
    store.verify_identity(record.key.identity)


def test_cleanup_resumes_unchanged_partial_deletion(tmp_path, monkeypatch):
    store = CoreStore(tmp_path/"core.sqlite", tmp_path/"artifacts")
    store.initialize()
    directory = orphan(store)
    plan = store.cleanup_plan()
    original = shutil.rmtree
    def interrupted(path, *args, **kwargs):
        next(Path(path).glob("*.parquet")).unlink()
        raise OSError("interrupted removal")
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "rmtree", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            store.apply_cleanup(plan)
    assert len(list(directory.iterdir())) == 1
    store.apply_cleanup(plan)
    assert not directory.exists()
    assert original is shutil.rmtree


def test_cleanup_timestamp_preview_skips_unknown_and_dangling_members(tmp_path):
    from datetime import datetime, UTC
    store = CoreStore(tmp_path/"meta.sqlite", tmp_path/"artifacts")
    store.initialize()
    directory = orphan(store)
    (directory/"broken.parquet").symlink_to(tmp_path/"absent")
    assert store.cleanup_plan(created_before=datetime.now(UTC))["candidates"] == []
    assert directory.exists()
