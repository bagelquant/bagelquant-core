"""Public artifact proofs and original-reference repair stay metadata bounded."""
from dataclasses import replace
from datetime import timedelta
import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Node, NodeMaterialization, RevisionConflict
from bagelquant_core.hashing import hash_mapping
from bagelquant_core.operator import negate
from bagelquant_core.store import _manifest_content
from test_global_graph import finish, global_graph, source
from test_rebalance_incremental import _expression, _inputs


def test_artifact_prefix_and_bounded_tail_keep_status_reason_and_skip_original(tmp_path, monkeypatch):
    first = _inputs(35, empty=(0, 34))
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    old_id = old["results"]["main"][root]
    updated = _inputs(36, empty=(0, 34, 35))
    current = finish(graph.plan_update({"main": updated}, through=updated["x"].domain.times.max()))
    current_id = current["results"]["main"][root]
    start, end = first["x"].domain.times.min(), first["x"].domain.times.max()
    with monkeypatch.context() as patch:
        patch.setattr(pl, "read_parquet", lambda *_a, **_k: pytest.fail("artifact proof decoded a historical file"))
        assert store.artifact_interval_identity(old_id, "decisions", start=start, end=end) == store.artifact_interval_identity(current_id, "decisions", start=start, end=end)
        assert store.describe_artifact(current_id, "decisions")["indexed"] is True
    scanned = []
    scan = pl.scan_parquet
    monkeypatch.setattr(pl, "scan_parquet", lambda path, **kwargs: (scanned.append(str(path)), scan(path, **kwargs))[1])
    day = updated["x"].domain.times.max()
    tail = store.read_artifact(current_id, "decisions", start=day, end=day)
    assert tail.height == 1 and tail["reason"][0] == "empty_universe"
    assert len(scanned) == 1
    assert store.manifest(old_id)["artifacts"]["decisions"]["$file"] not in scanned[0]
    scanned.clear()
    assert store.read_artifact(current_id, "decisions", start=day + timedelta(days=1), end=day + timedelta(days=1)).is_empty()
    assert not scanned


def test_selected_value_references_repair_one_day_without_duplicate_month_rows(tmp_path, monkeypatch):
    inputs = source(6)
    graph, store, root = global_graph(tmp_path, negate(inputs))
    old = finish(graph.plan_update({"main": {"x": inputs}}, through=inputs.domain.times.max()))
    old_id = old["results"]["main"][root]
    prior = store.read(old_id)
    old_bytes = {part["$file"]: store._file(part["$file"]).read_bytes() for part in store.manifest(old_id)["partitions"]}
    days = prior.panel.domain.times.to_list()
    repaired_frame = prior.panel.collect(include_traces=True).filter(pl.col("time") == days[2]).with_columns(pl.lit(-99.0).alias("value"))
    repaired_domain = prior.panel.domain.window(days[2], days[2])
    repaired_panel = Node.from_domain(repaired_frame, repaired_domain,
        value_type=prior.panel.value_type, trace_columns=prior.panel.trace_columns,
        trace_identity=prior.panel.trace_identity)
    repair = NodeMaterialization(replace(prior.key, domain_identity=repaired_domain.signature, context_identity="repair"), repaired_panel)
    store.publish(repair)
    current = NodeMaterialization(replace(prior.key, context_identity="repaired-composition"), prior.panel)
    with monkeypatch.context() as patch:
        patch.setattr(pl, "read_parquet", lambda *_a, **_k: pytest.fail("reference composition decoded history"))
        store.publish(current, value_references=[
            {"identity": old_id, "start": days[0], "end": days[1]},
            repair.key.identity,
            {"identity": old_id, "start": days[3], "end": days[-1]},
        ])
    expected = prior.panel.collect(include_traces=True).with_columns(pl.when(pl.col("time") == days[2]).then(-99.0).otherwise(pl.col("value")).alias("value"))
    assert_frame_equal(store.read(current.key.identity).panel.collect(include_traces=True), expected, check_exact=True)
    reads = []
    scan = pl.scan_parquet
    monkeypatch.setattr(pl, "scan_parquet", lambda path, **kwargs: (reads.append(str(path)), scan(path, **kwargs))[1])
    assert_frame_equal(store.read_values(current.key.identity, start=days[2], end=days[2], include_traces=True), repaired_frame)
    domain_paths = {str(store._file(reference["$file"]))
        for name, reference in store.manifest(current.key.identity)["domain"].items() if name != "dynamic"}
    value_reads = [path for path in reads if path not in domain_paths]
    assert value_reads == [str(store._file(store.manifest(repair.key.identity)["partitions"][0]["$file"]))]
    assert all(path not in value_reads[0] for path in old_bytes)
    assert all(store._file(path).read_bytes() == raw for path, raw in old_bytes.items())


def test_legacy_artifact_index_is_explicit_preserves_original_and_binds_authority(tmp_path, monkeypatch):
    inputs = source(4)
    graph, store, root = global_graph(tmp_path, negate(inputs))
    result = finish(graph.plan_update({"main": {"x": inputs}}, through=inputs.domain.times.max()))
    identity = result["results"]["main"][root]
    saved = store.read(identity)
    decisions = pl.DataFrame({"time": inputs.domain.times, "status": ["hold"] * 4, "reason": ["saved"] * 4})
    key = replace(saved.key, context_identity="legacy-artifact-fixture")
    store.publish(NodeMaterialization(key, saved.panel, artifacts={"decisions": decisions}))
    # Model a retained pre-index receipt; its registered content hash and old
    # table bytes are immutable throughout the explicit maintenance below.
    manifest = store.manifest(key.identity)
    artifact = manifest["artifacts"]["decisions"]
    artifact.pop("days")
    artifact.pop("day_rows")
    manifest["content_identity"] = hash_mapping(_manifest_content(manifest))
    with store.transaction() as connection:
        connection.execute("UPDATE materializations SET manifest_json=? WHERE identity=?", (json.dumps(manifest), key.identity))
    # Rebind only the already-existing Domain/value index in this fixture. The
    # public artifact maintainer neither derives nor changes numerical values.
    with store.transaction() as connection:
        connection.execute("UPDATE computation_index SET content_identity=? WHERE identity=?", (manifest["content_identity"], key.identity))
    start, end = inputs.domain.times.min(), inputs.domain.times.max()
    with monkeypatch.context() as patch:
        patch.setattr(pl, "read_parquet", lambda *_a, **_k: pytest.fail("ordinary proof backfilled legacy index"))
        assert store.artifact_interval_identity(key.identity, "decisions", start=start, end=end) is None
        with pytest.raises(ValueError, match="explicit"):
            store.read_artifact(key.identity, "decisions", start=end, end=end)
    before = store._file(artifact["$file"]).read_bytes()
    original = store.manifest(key.identity)
    assert store.build_artifact_index(store.artifact_index_plan(key.identity, "decisions"))["status"] == "complete"
    assert store.manifest(key.identity) == original
    assert store._file(artifact["$file"]).read_bytes() == before
    assert store.artifact_interval_identity(key.identity, "decisions", start=start, end=end)
    assert_frame_equal(store.read_artifact(key.identity, "decisions", start=end, end=end), decisions.tail(1))
    broken = store.artifact_index_plan(key.identity, "decisions")
    broken["source_content"] = "0" * 64
    with pytest.raises(RevisionConflict, match="changed"):
        store.build_artifact_index(broken)
