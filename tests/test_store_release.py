"""Explicit retirement preserves shared evidence and permits exact recomputation."""
from datetime import date

import pytest
from bagelquant_core import CoreStore, Domain
from test_core_store import value


def test_release_shared_domain_and_republish_identical_key(tmp_path):
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    domain = Domain(calendar=[date(2024, 1, 2)], universe=["A", "B"])
    first, second = value(domain, node_id="first"), value(domain, node_id="second")
    store.publish(first)
    store.publish(second)
    old = store.manifest(first.key.identity)
    plan = store.plan_release([first.key.identity], retained=[second.key.identity])
    assert plan["released"] == [first.key.identity]
    assert plan["files"]
    assert old["domain"]["calendar"]["$file"] not in {item["path"] for item in plan["files"]}
    assert store.apply_release(plan) == store.apply_release(plan)
    assert store.lookup(first.key) is None
    assert first.key.identity not in {item["identity"] for item in store.inventory()}
    store.verify_identity(second.key.identity)
    store.publish(first)
    assert store.manifest(first.key.identity) == old
    store.verify_identity(first.key.identity)


def test_release_retry_and_stale_preview(tmp_path, monkeypatch):
    from pathlib import Path
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    domain = Domain(calendar=[date(2024, 1, 2)], universe=["A"])
    first = value(domain)
    store.publish(first)
    plan = store.plan_release([first.key.identity])
    real = Path.unlink
    calls = []
    def interrupted(path, *args, **kwargs):
        if str(path).endswith('.parquet'):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("interrupted unlink")
        return real(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            store.apply_release(plan)
    assert store.lookup(first.key) is None
    store.apply_release(plan)
    assert not list(store.artifact_path.rglob("*.parquet"))
    store.publish(first)
    stale = store.plan_release([first.key.identity])
    store.publish(value(domain, node_id="other"))
    with pytest.raises(RuntimeError, match="changed"):
        store.apply_release(stale)


def test_retirement_preserves_immutable_update_and_rebuilds_only_on_explicit_run(tmp_path):
    from bagelquant_core.operator import negate
    from test_global_graph import finish, global_graph, source
    panel = source(3)
    graph, store, root = global_graph(tmp_path, negate(panel))
    original = finish(graph.plan_update({"main": {"x": panel}}, through=panel.domain.times.max()))
    identity = original["results"]["main"][root]
    plan = store.plan_release([identity])
    store.apply_release(plan)
    assert store.describe_update(original["identity"]) == original
    assert store.verify_update(original["identity"]) == original
    assert not store.update_covers(original["identity"], "main", node_ids=[root])
    pending = graph.plan_update({"main": {"x": panel}}, through=panel.domain.times.max())
    assert not pending.complete
    finish(pending)
    store.verify_identity(identity)


def test_release_retry_keeps_newly_republished_files(tmp_path, monkeypatch):
    from pathlib import Path
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    record = value(Domain(calendar=[date(2024, 1, 2)], universe=["A"]))
    store.publish(record)
    plan = store.plan_release([record.key.identity])
    real = Path.unlink
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", lambda path, *a, **kw: (_ for _ in ()).throw(OSError("interrupt")) if path.suffix == ".parquet" else real(path, *a, **kw))
        with pytest.raises(OSError, match="interrupt"):
            store.apply_release(plan)
    assert store.release_status(plan["id"]) == "pending"
    store.publish(record)
    result = store.apply_release(plan)
    assert result["bytes"] == 0
    assert store.release_status(plan["id"]) == "completed"
    store.verify_identity(record.key.identity)


def test_release_refuses_corrupt_surviving_metadata(tmp_path):
    import json
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    domain = Domain(calendar=[date(2024, 1, 2)], universe=["A"])
    first, kept = value(domain), value(domain, node_id="kept")
    store.publish(first)
    store.publish(kept)
    with store.transaction() as db:
        manifest = store.manifest(kept.key.identity)
        manifest["logical_hash"] = "bad"
        db.execute("UPDATE materializations SET manifest_json=? WHERE identity=?", (json.dumps(manifest), kept.key.identity))
    with pytest.raises(ValueError, match="corrupt"):
        store.plan_release([first.key.identity])
    assert store.release_status("unknown") == "unknown"


def test_normal_reads_and_publication_do_not_scan_unrelated_retirements(tmp_path, monkeypatch):
    import bagelquant_core.release as release
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    domain = Domain(calendar=[date(2024, 1, 2)], universe=["A"])
    old, current = value(domain), value(domain, node_id="current")
    store.publish(old)
    store.apply_release(store.plan_release([old.key.identity]))
    with store.transaction() as db:
        db.executemany("INSERT INTO materialization_releases VALUES(?,?)", [(f"{index:064x}", "unrelated") for index in range(1000)])
    monkeypatch.setattr(release, "released", lambda *_: pytest.fail("normal lookup scanned all retired history"))
    store.publish(current)
    assert store.manifest(current.key.identity)
    store.read(current.key.identity)


def test_composed_values_and_artifacts_revive_cold_with_originals_preserved(tmp_path):
    from dataclasses import replace
    import polars as pl
    from polars.testing import assert_frame_equal
    from bagelquant_core import NodeMaterialization
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    whole = value(Domain(calendar=dates, universe=["A"]))
    parts = [value(Domain(calendar=[day], universe=["A"])) for day in dates]
    for part in parts:
        store.publish(part)
    table = pl.DataFrame({"time": dates, "status": ["held", "traded"]})
    # The artifact is also composed; its physical leaf layout differs cold.
    first = replace(parts[0], artifacts={"decisions": table.head(1)})
    first = replace(first, key=replace(first.key, context_identity="artifact-prefix"))
    store.publish(first)
    composed = replace(whole, artifacts={"decisions": store.compose_artifact(first.key.identity, "decisions", table.tail(1))})
    store.publish(composed, value_references=[part.key.identity for part in parts])
    original = store.manifest(whole.key.identity)
    expected = store.read_values(whole.key.identity, include_traces=True)
    cold = NodeMaterialization(whole.key, whole.panel, artifacts={"decisions": table},
        checkpoint=whole.checkpoint, training_audits=whole.training_audits)
    with pytest.raises(ValueError, match="immutable materialization conflict"):
        store.publish(cold)
    with store.read_context() as reader:
        token = reader.interval_identity(whole.key.identity, start=dates[0], end=dates[-1])
        for _ in range(2):
            plan = store.plan_release([whole.key.identity])
            store.apply_release(plan)
            assert reader.manifest(whole.key.identity) is None
            assert reader.interval_identity(whole.key.identity, start=dates[0], end=dates[-1]) is None
            store.publish(cold)
            current = reader.manifest(whole.key.identity)
            assert not current.get("value_sources")
            assert reader.interval_identity(whole.key.identity, start=dates[0], end=dates[-1]) == token
            assert_frame_equal(reader.read_values(whole.key.identity, include_traces=True), expected)
            assert_frame_equal(reader.read_artifact(whole.key.identity, "decisions"), table)
            store.verify_identity(whole.key.identity)
        with store.read_transaction() as db:
            archived = store._verified_reference_manifest(db, whole.key.identity, original["content_identity"])
            assert archived == original
    assert all(store.manifest(part.key.identity) is None for part in parts)
    assert store.manifest(first.key.identity) is None


def test_revival_preserves_historical_region_and_cutoff_proof_authority(tmp_path):
    from dataclasses import replace
    from bagelquant_core import MaterializationKey, Node
    from test_cutoff_proof import prepared
    before, _after, graph, store, root, update, kwargs = prepared(tmp_path)
    originals = {identity: store.manifest(identity) for identity in update["results"]["main"].values()}
    records = {identity: store.read(identity) for identity in originals}
    records = {identity: replace(record, panel=Node.from_domain(record.panel.collect(include_traces=True),
        record.panel.domain, value_type=record.panel.value_type, trace_columns=record.panel.trace_columns,
        trace_identity=record.panel.trace_identity)) for identity, record in records.items()}
    plan = store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root], **kwargs)
    release = store.plan_release([item["identity"] for item in store.inventory()])
    store.apply_release(release)
    for identity, record in records.items():
        store.publish(record, execution_policy=originals[identity]["execution_policy"])
    assert any(store.manifest(identity)["content_identity"] != manifest["content_identity"] for identity, manifest in originals.items())
    assert store.describe_update(update["identity"]) == update
    # The pre-deletion admission still authenticates the original physical
    # evidence, even though the active result now has cold partitions.
    result = store.apply_input_proof_rebind(plan)
    store.verify_identity(result["results"][root])
    assert store.read_values(result["results"][root]).height == before.collect().height
    for identity, original in originals.items():
        with store.read_transaction() as db:
            archived = store._verified_reference_manifest(db, identity, original["content_identity"])
            assert archived == original
    assert store.lookup(MaterializationKey.from_dict(originals[update["results"]["main"][root]]["key"])) is not None


def test_released_incremental_region_runs_again_without_changing_old_updates(tmp_path):
    from bagelquant_core import Graph
    from test_global_graph import finish, global_graph, source
    first = source(5)
    expression = Graph.from_dsl("a = negate(x)\noutput = rankpct(a)", inputs={"x": first})
    graph, store, root = global_graph(tmp_path, expression)
    initial = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    current = source(6)
    appended = finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    identity = appended["results"]["main"][root]
    original = store.manifest(identity)
    assert original.get("value_sources")
    expected = store.read_values(identity, include_traces=True)
    store.apply_release(store.plan_release([identity]))
    rebuilt = finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    assert rebuilt["results"]["main"][root] == identity
    assert store.read_values(identity, include_traces=True).equals(expected)
    assert store.describe_update(initial["identity"]) == initial
    assert store.verify_update(appended["identity"]) == appended


def test_later_index_enrichment_survives_repeated_release_and_cold_revival(tmp_path):
    from bagelquant_core.computation_index import read
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    whole = value(Domain(calendar=dates, universe=["A"]))
    parts = [value(Domain(calendar=[day], universe=["A"])) for day in dates]
    for part in parts:
        store.publish(part)
    store.publish(whole, value_references=[part.key.identity for part in parts])
    original = store.manifest(whole.key.identity)
    with store.transaction() as db:
        expected = read(db, original)
        db.execute("DELETE FROM computation_index WHERE identity=?", (whole.key.identity,))
    store.apply_release(store.plan_release([whole.key.identity]))
    for part in parts:
        store.publish(part)
    store.publish(whole, value_references=[part.key.identity for part in parts])
    assert store.manifest(whole.key.identity) == original
    with store.read_transaction() as db:
        assert read(db, original) == expected
    store.apply_release(store.plan_release([whole.key.identity]))
    store.publish(whole)
    assert store.manifest(whole.key.identity)["content_identity"] != original["content_identity"]
    with store.read_transaction() as db:
        assert store._verified_reference_manifest(db, whole.key.identity, original["content_identity"]) == original
        assert read(db, original) == expected
