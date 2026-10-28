"""Explicit retirement of caller-selected numerical results, without rewriting history."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

from .hashing import hash_mapping

DDL = (
    "CREATE TABLE IF NOT EXISTS materialization_releases(identity TEXT PRIMARY KEY,plan_id TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS artifact_release_plans(identity TEXT PRIMARY KEY,plan_json TEXT NOT NULL,completed INTEGER NOT NULL DEFAULT 0,removed_json TEXT NOT NULL DEFAULT '{}')",
    "CREATE TABLE IF NOT EXISTS materialization_originals(identity TEXT NOT NULL,content_identity TEXT NOT NULL,manifest_json TEXT NOT NULL,index_json TEXT,PRIMARY KEY(identity,content_identity))",
)


def released(connection):
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='materialization_releases'").fetchone():
        return set()
    return {row[0] for row in connection.execute("SELECT identity FROM materialization_releases")}


def is_released(connection, identity):
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='materialization_releases'").fetchone():
        return False
    return connection.execute("SELECT 1 FROM materialization_releases WHERE identity=?", (identity,)).fetchone() is not None


def original_manifest(connection, identity, content_identity):
    """Select immutable evidence, independently of current value availability."""
    row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (identity,)).fetchone()
    if row is not None and json.loads(row[0])["content_identity"] == content_identity:
        return row[0]
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='materialization_originals'").fetchone():
        row = connection.execute("SELECT manifest_json FROM materialization_originals WHERE identity=? AND content_identity=?",
            (identity, content_identity)).fetchone()
        if row is not None:
            return row[0]
    raise KeyError((identity, content_identity))


def archive_original(connection, identity):
    """Preserve exact pre-retirement metadata and coordinate evidence once."""
    raw = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (identity,)).fetchone()[0]
    manifest = json.loads(raw)
    row = connection.execute("SELECT content_identity,version,payload_json,payload_identity FROM computation_index WHERE identity=?",
        (identity,)).fetchone() if connection.execute("SELECT 1 FROM sqlite_master WHERE name='computation_index'").fetchone() else None
    if row is not None:
        from .computation_index import read, VERSION
        indexed = read(connection, manifest)
        if indexed is None or row[0] != manifest["content_identity"] or row[1] != VERSION:
            row = None
    index = None if row is None else json.dumps(list(row))
    connection.execute("INSERT INTO materialization_originals VALUES(?,?,?,?) "
        "ON CONFLICT(identity,content_identity) DO UPDATE SET index_json=excluded.index_json "
        "WHERE materialization_originals.index_json IS NULL AND excluded.index_json IS NOT NULL",
        (identity, manifest["content_identity"], raw, index))


def files(value):
    result = {}
    def visit(item):
        if isinstance(item, dict):
            if "$file" in item:
                path, checksum = item["$file"], item["sha256"]
                if path in result and result[path] != checksum:
                    raise ValueError("conflicting Core file authority")
                result[path] = checksum
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
    visit(value)
    return result


def safe_file(store, relative):
    if not isinstance(relative, str) or not re.fullmatch(r"values/[a-f0-9]{64}/[a-f0-9]{32}/[0-9]{6}\.parquet", relative):
        raise ValueError("release cannot remove unknown or non-Core files")
    raw = store.artifact_path / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe Core release path")
    for member in (raw, *raw.parents):
        if member == store.artifact_path.parent:
            break
        if member.is_symlink():
            raise ValueError("Core release cannot follow symbolic links")
    return store._file(relative)


def plan_release(store, identities, *, retained=()):
    """Retire a selected ancestry except anything needed by another retained result."""
    roots, keep = sorted(set(identities)), sorted(set(retained))
    if not store.meta_path.is_file():
        manifests, retired, seals, epoch = {}, set(), {}, 0
    else:
        with store.read_transaction() as db:
            manifests = {row[0]: json.loads(row[1]) for row in db.execute("SELECT identity,manifest_json FROM materializations")}
            retired = released(db)
            epoch = db.execute("SELECT count(*) FROM artifact_release_plans").fetchone()[0] if db.execute("SELECT 1 FROM sqlite_master WHERE name='artifact_release_plans'").fetchone() else 0
            seals = {}
            for identity, manifest in manifests.items():
                store._verify_metadata(manifest, identity)
                rows = store._region_evidence_rows(db, manifest, "seal")
                seals[identity] = store._decode_region_evidence(manifest, "seal", rows)
    active = set(manifests) - retired
    if set(roots) - active or set(keep) - active:
        raise ValueError("Core release requires available registered roots")
    def links(item):
        found = set()
        if isinstance(item, str) and item in active:
            found.add(item)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                found.update(links(key))
                found.update(links(child))
        elif isinstance(item, (tuple, list)):
            for child in item:
                found.update(links(child))
        return found
    parents = {identity: links([manifest, seals.get(identity, [])]) - {identity}
        for identity, manifest in manifests.items() if identity in active}
    def closure(selected):
        found, pending = set(), list(selected)
        while pending:
            identity = pending.pop()
            if identity in found:
                continue
            found.add(identity)
            pending.extend(parents.get(identity, set()) - found)
        return found
    candidates = closure(roots)
    protected = closure((active - candidates) | set(keep))
    selected = sorted(candidates - protected)
    kept_files = files([manifests[key] for key in active - set(selected)])
    removed_files = files([manifests[key] for key in selected])
    candidates_files = []
    for relative, checksum in sorted(removed_files.items()):
        if relative not in kept_files:
            path = safe_file(store, relative)
            candidates_files.append({"path": relative, "sha256": checksum,
                "bytes": path.stat().st_size if path.is_file() else 0})
    body = {"schema": "core.release.v1", "roots": roots, "retained": keep, "released": selected,
        "epoch": epoch, "registry": hash_mapping({key: manifests[key]["content_identity"] for key in sorted(active)}),
        "files": candidates_files}
    return {**body, "id": hash_mapping(body), "bytes": sum(item["bytes"] for item in candidates_files)}


def apply_release(store, plan):
    """Persist retirement before unlinking; retry the exact request after interruption."""
    from .store import RevisionConflict, _checksum, _json
    body = {key: value for key, value in plan.items() if key not in {"id", "bytes"}}
    if body.get("schema") != "core.release.v1" or hash_mapping(body) != plan.get("id"):
        raise ValueError("Core release plan identity differs")
    store.initialize()
    with store.transaction() as db:
        for statement in DDL:
            db.execute(statement)
        previous = db.execute("SELECT plan_json,completed,removed_json FROM artifact_release_plans WHERE identity=?", (plan["id"],)).fetchone()
        if previous:
            if json.loads(previous[0]) != dict(plan):
                raise ValueError("Core release request differs")
            if previous[1]:
                return {"id": plan["id"], "released": plan["released"], "bytes": sum(json.loads(previous[2]).values())}
        else:
            if plan_release(store, plan["roots"], retained=plan["retained"]) != dict(plan):
                raise RevisionConflict("Core release inventory changed; review a new preview")
            db.execute("INSERT INTO artifact_release_plans VALUES(?,?,0,'{}')", (plan["id"], _json(plan)))
            for identity in plan["released"]:
                archive_original(db, identity)
            db.executemany("INSERT INTO materialization_releases VALUES(?,?)", [(key, plan["id"]) for key in plan["released"]])
            db.executemany("DELETE FROM node_completions WHERE identity=?", [(key,) for key in plan["released"]])
    # One publication fence protects the retained-file snapshot for the unlink
    # phase. Interruption leaves the committed retirement plan for exact retry.
    with store.transaction() as db:
        retired = released(db)
        retained_manifests = []
        for identity, raw in db.execute("SELECT identity,manifest_json FROM materializations"):
            if identity not in retired:
                manifest = json.loads(raw)
                store._verify_metadata(manifest, identity)
                retained_manifests.append(manifest)
        retained_files = files(retained_manifests)
        removed = json.loads(db.execute("SELECT removed_json FROM artifact_release_plans WHERE identity=?", (plan["id"],)).fetchone()[0])
        for item in plan["files"]:
            if item["path"] in retained_files:
                continue
            path = safe_file(store, item["path"])
            if path.exists():
                if not path.is_file() or _checksum(path) != item["sha256"]:
                    raise RevisionConflict("Core release file differs from its registered authority")
                path.unlink()
            parent = path.parent
            while parent != store.artifact_path and parent.is_relative_to(store.artifact_path):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
            removed[item["path"]] = item["bytes"]
            db.execute("UPDATE artifact_release_plans SET removed_json=? WHERE identity=?", (_json(removed), plan["id"]))
    with store.transaction() as db:
        db.execute("UPDATE artifact_release_plans SET completed=1 WHERE identity=?", (plan["id"],))
        removed = json.loads(db.execute("SELECT removed_json FROM artifact_release_plans WHERE identity=?", (plan["id"],)).fetchone()[0])
    return {"id": plan["id"], "released": plan["released"], "bytes": sum(removed.values())}
