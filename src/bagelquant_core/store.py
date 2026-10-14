"""Caller-located, immutable graph and numerical artifact storage."""
from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from typing import Any, Iterator, Mapping, Sequence
import uuid

import polars as pl
from .inspection import open_metadata_snapshot
from .artifact_verification import ArtifactVerification

from .domain import Domain
from .hashing import hash_mapping, hash_dataframe
from .logical import LogicalGraphSpec
from .materialization import (
    MaterializationKey, MaterializationLookup, MaterializationStatus, NodeMaterialization,
)
from .node import Node, ValueType

SCHEMA_VERSION = 2


class RevisionConflict(RuntimeError):
    """The graph changed after an update was planned."""


@dataclass(frozen=True)
class ArtifactReference:
    """Registered original auxiliary table with an optional disjoint typed tail."""
    identity: str
    name: str
    content_identity: str
    tail: pl.DataFrame | None = None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest_content(item):
    if isinstance(item, dict):
        return {key: _manifest_content(child) for key, child in item.items()
                if key != "content_identity" and not (key == "$file" and "sha256" in item)}
    if isinstance(item, list):
        return [_manifest_content(child) for child in item]
    return item


def _result_content(manifest):
    """Numerical/evidence identity independent of physical value fragmentation."""
    return {key: _manifest_content(manifest[key]) for key in (
        "key", "value_type", "trace_columns", "trace_identity", "domain", "days", "logical_hash",
        "artifacts", "checkpoint", "training_audits", "execution_policy")}


class CoreStore:
    """Own Core state at two explicit paths; never discover application storage.

    Graph mutation and complete-update promotion use SQLite transactions. Artifact
    bytes are immutable, checksum-verified files. Unreferenced completed bytes may
    survive interruption; incomplete writes never become readable receipts.
    """

    def __init__(self, meta_path: str | Path, artifact_path: str | Path):
        self.meta_path = Path(meta_path).resolve()
        self.artifact_path = Path(artifact_path).resolve()
        if self.meta_path == self.artifact_path:
            raise ValueError("metadata and artifact paths must be distinct")
        self._verification: ArtifactVerification | None = None

    @contextmanager
    def read_context(self) -> Iterator[CoreStore]:
        """Share finite checksum proofs across this operation's worker threads.

        Each metadata read owns its SQLite connection; graph revision guards
        always read fresh committed state. The returned reader expires on exit.
        """
        reader = copy(self)
        reader._verification = ArtifactVerification()
        try:
            yield reader
        finally:
            reader._verification.close()
            reader._verification = None

    @contextmanager
    def read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Read committed metadata without initialization or writer ownership."""
        connection = sqlite3.connect(self.meta_path.as_uri() + "?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.rollback()
            connection.close()

    def inspect(self, *, runtime: bool = False) -> dict[str, Any]:
        """Inspect schema readiness without creating storage or recovering work.

        This reports the metadata contract, not artifact integrity. Call
        ``check_integrity`` explicitly to inspect committed numerical evidence.
        Runtime inspection uses SQLite read coordination during live writes;
        default inspection preserves every original file and sidecar unchanged.
        """
        if not self.meta_path.exists():
            return {"status": "uninitialized", "schema_version": None, "reason": "metadata_missing"}
        if not self.meta_path.is_file():
            return {"status": "incompatible", "schema_version": None, "reason": "metadata_not_file"}
        try:
            with open_metadata_snapshot(self.meta_path, runtime=runtime) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
                required = {
                    "materializations": {"identity", "node_id", "key_json", "manifest_json"},
                    "graphs": {"graph_id", "revision", "state_revision", "spec_json", "current_update"},
                    "graph_nodes": {"graph_id", "node_id", "status"},
                    "contexts": {"graph_id", "context_id", "definition_json"},
                    "cleanup_receipts": {"identity", "result_json"},
                    "updates": {"identity", "graph_id", "receipt_json"},
                    "node_completions": {"graph_id", "context_id", "node_id", "identity", "definition_digest", "upstream_digest", "through", "checked_through"},
                }
                if not tables and version == 0:
                    return {"status": "uninitialized", "schema_version": 0, "reason": "metadata_empty"}
                if version != SCHEMA_VERSION or not set(required).issubset(tables):
                    return {"status": "incompatible", "schema_version": version, "reason": "schema_incompatible"}
                if any(not columns.issubset({row[1] for row in connection.execute(f"PRAGMA table_info({table})")})
                       for table, columns in required.items()):
                    return {"status": "incompatible", "schema_version": version, "reason": "schema_incompatible"}
                return {"status": "ready", "schema_version": version, "reason": None}
        except (sqlite3.DatabaseError, OSError):
            return {"status": "incompatible", "schema_version": None, "reason": "metadata_unreadable"}

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.meta_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if version not in (0, SCHEMA_VERSION) or (version == 0 and tables):
                raise ValueError("incompatible Core database; use a new explicit storage location")
            for statement in (
                "CREATE TABLE IF NOT EXISTS materializations(identity TEXT PRIMARY KEY,node_id TEXT NOT NULL,key_json TEXT NOT NULL,manifest_json TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS materialization_node ON materializations(node_id)",
                "CREATE TABLE IF NOT EXISTS graphs(graph_id TEXT PRIMARY KEY,revision INTEGER NOT NULL,state_revision INTEGER NOT NULL,spec_json TEXT NOT NULL,current_update TEXT)",
                "CREATE TABLE IF NOT EXISTS graph_nodes(graph_id TEXT NOT NULL REFERENCES graphs(graph_id),node_id TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN ('active','sleeping')),PRIMARY KEY(graph_id,node_id))",
                "CREATE TABLE IF NOT EXISTS contexts(graph_id TEXT NOT NULL REFERENCES graphs(graph_id),context_id TEXT NOT NULL,definition_json TEXT NOT NULL,PRIMARY KEY(graph_id,context_id))",
                "CREATE TABLE IF NOT EXISTS cleanup_receipts(identity TEXT PRIMARY KEY,result_json TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS updates(identity TEXT PRIMARY KEY,graph_id TEXT NOT NULL REFERENCES graphs(graph_id),receipt_json TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS node_completions(graph_id TEXT NOT NULL,context_id TEXT NOT NULL,node_id TEXT NOT NULL,identity TEXT NOT NULL REFERENCES materializations(identity),definition_digest TEXT NOT NULL,upstream_digest TEXT NOT NULL,through TEXT NOT NULL,checked_through TEXT NOT NULL,PRIMARY KEY(graph_id,context_id,node_id),FOREIGN KEY(graph_id,context_id) REFERENCES contexts(graph_id,context_id),FOREIGN KEY(graph_id,node_id) REFERENCES graph_nodes(graph_id,node_id))",
            ):
                connection.execute(statement)
            if not tables:
                from .computation_index import DDL
                connection.execute(DDL)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        with sqlite3.connect(self.meta_path, timeout=30) as connection:
            connection.execute("PRAGMA journal_mode=WAL")

    def _file(self, relative: str) -> Path:
        path = (self.artifact_path / relative).resolve()
        if not path.is_relative_to(self.artifact_path) or path == self.artifact_path:
            raise ValueError("artifact path escapes Core storage")
        return path

    def manifest(self, identity: str) -> dict[str, Any] | None:
        if not self.meta_path.is_file():
            return None
        with self.read_transaction() as connection:
            row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (identity,)).fetchone()
        if row is None:
            return None
        manifest = json.loads(row[0])
        self._verify_metadata(manifest, identity)
        if manifest.get("proof_rebinding") is not None:
            with self.read_transaction() as connection:
                self._verify_proof_rebinding(connection, manifest["proof_rebinding"],
                    MaterializationKey.from_dict(manifest["key"]), manifest["execution_policy"])
        if manifest.get("value_sources") or any(isinstance(artifact, Mapping) and "$artifact_concat" in artifact
                for artifact in manifest["artifacts"].values()):
            with self.read_transaction() as connection:
                self._verify_artifact_sources(connection, manifest)
        return manifest

    def _verify_artifact_sources(self, connection, manifest, *, visiting=None):
        visiting = set() if visiting is None else visiting
        identity = manifest["identity"]
        if identity in visiting:
            raise ValueError("cyclic original artifact authority")
        visiting.add(identity)
        try:
            for source in manifest.get("value_sources", ()):
                row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (source["identity"],)).fetchone()
                if row is None:
                    raise ValueError("original value authority is missing")
                original = json.loads(row[0])
                self._verify_metadata(original, source["identity"])
                if (original["content_identity"] != source["source_content"]
                        or original["node_id"] != manifest["node_id"]
                        or original["key"]["implementation_id"] != manifest["key"]["implementation_id"]
                        or original["value_type"] != manifest["value_type"]
                        or original["trace_columns"] != manifest["trace_columns"]):
                    raise ValueError("original value authority differs")
                self._verify_artifact_sources(connection, original, visiting=visiting)
            for artifact in manifest["artifacts"].values():
                if not isinstance(artifact, Mapping) or "$artifact_concat" not in artifact:
                    continue
                source = artifact["original_source"]
                row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (source["identity"],)).fetchone()
                if row is None:
                    raise ValueError("original artifact authority missing")
                original = json.loads(row[0])
                self._verify_metadata(original, source["identity"])
                if (original["content_identity"] != source["source_content"]
                        or original["node_id"] != manifest["node_id"]
                        or original["key"]["implementation_id"] != manifest["key"]["implementation_id"]):
                    raise ValueError("original artifact authority differs")
                original_artifact = original["artifacts"].get(source["name"])
                if hash_mapping(original_artifact) != source["artifact_identity"]:
                    raise ValueError("original artifact channel differs")
                leaves = original_artifact.get("$artifact_concat", [original_artifact])
                if artifact["$artifact_concat"][:len(leaves)] != leaves or artifact["schema"] != original_artifact["schema"]:
                    raise ValueError("artifact composition lost original references")
                if original.get("proof_rebinding") is not None:
                    self._verify_proof_rebinding(connection, original["proof_rebinding"],
                        MaterializationKey.from_dict(original["key"]), original["execution_policy"])
                self._verify_artifact_sources(connection, original, visiting=visiting)
        finally:
            visiting.remove(identity)

    @classmethod
    def _verify_proof_rebinding(cls, connection, plan, key, policy):
        from .proof_maintenance import verify_rebinding
        verify_rebinding(cls, connection, plan, key, policy)

    @staticmethod
    def _verify_metadata(manifest: Mapping[str, Any], identity: str) -> None:
        key = MaterializationKey.from_dict(manifest["key"])
        if (manifest.get("content_identity") != hash_mapping(_manifest_content(manifest))
                or manifest.get("identity") != identity or key.identity != identity
                or manifest.get("node_id") != key.node_id):
            raise ValueError("corrupt Core manifest metadata")

    def verify(self, manifest: Mapping[str, Any]) -> None:
        self._verify_metadata(manifest, manifest["identity"])
        with self.read_transaction() as connection:
            self._verify_artifact_sources(connection, manifest)
        if manifest.get("proof_rebinding") is not None:
            with self.read_transaction() as connection:
                self._verify_proof_rebinding(connection, manifest["proof_rebinding"],
                    MaterializationKey.from_dict(manifest["key"]), manifest["execution_policy"])
        self._verify_files(manifest, receipt=manifest["content_identity"])

    def _verify_files(self, value, *, receipt: str) -> None:
        verification = self._verification or ArtifactVerification()
        def visit(value):
            if isinstance(value, Mapping):
                if "$file" in value:
                    path = self._file(value["$file"])
                    if not path.is_file():
                        raise ValueError(f"corrupt Core artifact: {value['$file']}")
                    verification.verify(path, value["sha256"], receipt=receipt, checksum=_checksum)
                for child in value.values():
                    visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)
        visit(value)

    def _decode(self, value):
        if isinstance(value, dict):
            if "$artifact_concat" in value:
                frames = [self._decode(item) for item in value["$artifact_concat"]]
                result = pl.concat(frames)
                if {name: str(dtype) for name, dtype in result.schema.items()} != value["schema"]:
                    raise ValueError("composed artifact schema differs from its registered receipt")
                return result
            if "$file" in value:
                try:
                    return pl.read_parquet(self._file(value["$file"]))
                except pl.exceptions.PolarsError as error:
                    raise ValueError(f"corrupt or unreadable Core artifact: {value['$file']}") from error
            if "$date" in value:
                return date.fromisoformat(value["$date"])
            if "$datetime" in value:
                return datetime.fromisoformat(value["$datetime"])
            if "$float" in value:
                return float(value["$float"])
            return {key: self._decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._decode(item) for item in value]
        return value

    def read_checkpoint(self, identity: str) -> Mapping[str, Any] | None:
        """Read only the registered checkpoint, without decoding other artifacts."""
        manifest = self.manifest(identity)
        if manifest is None:
            raise KeyError(f"unknown Core materialization: {identity}")
        return self._decode(manifest["checkpoint"])

    def compose_artifact(self, identity: str, name: str,
                         tail: pl.DataFrame | None = None) -> ArtifactReference:
        """Reference a registered dated artifact and append a nonoverlapping tail.

        Admission and publication validate original metadata, schema, dates and
        files. Missing legacy table metadata needs explicit owner maintenance;
        ordinary composition never reads or hashes historical numerical bytes.
        """
        manifest = self.manifest(identity)
        if manifest is None or name not in manifest["artifacts"]:
            raise ValueError("artifact composition requires a registered original channel")
        original = manifest["artifacts"][name]
        if not isinstance(original, Mapping) or not {"schema", "rows", "start", "end"} <= original.keys():
            raise ValueError("artifact composition needs explicit dated-artifact metadata maintenance")
        if original["schema"].get("time") != "Date" or original.get("unique_time") is not True:
            raise ValueError("artifact composition requires a unique dated original table")
        if tail is not None:
            if not isinstance(tail, pl.DataFrame) or {column: str(dtype) for column, dtype in tail.schema.items()} != original["schema"]:
                raise ValueError("artifact composition schema differs")
            if tail.schema.get("time") != pl.Date or tail["time"].null_count():
                raise ValueError("artifact composition requires nonnull Date time")
            if tail["time"].n_unique() != tail.height or not tail["time"].is_sorted():
                raise ValueError("artifact composition requires ordered unique Date time")
            if tail.height and (tail["time"].min() <= date.fromisoformat(manifest["through"])
                    or original["end"] is not None and str(tail["time"].min()) <= original["end"]):
                raise ValueError("artifact composition tail overlaps completed history")
        return ArtifactReference(identity, name, manifest["content_identity"], None if tail is None else tail.clone())

    def lookup(self, key: MaterializationKey, *, domain=None, evidence=True) -> NodeMaterialization | None:
        manifest = self.manifest(key.identity)
        if manifest is None:
            return None
        if manifest["key"] != key.to_dict():
            raise ValueError("materialization identity collision")
        saved_domain = manifest["domain"]
        if domain is None:
            calendar = self._decode(saved_domain["calendar"])["time"]
            membership = self._decode(saved_domain["membership"])
            universe = membership.with_columns(pl.lit(True).alias("active")) if saved_domain["dynamic"] else membership["asset_id"]
            domain = Domain(calendar=calendar, universe=universe, identity=key.domain_identity)
        elif domain.signature != key.domain_identity:
            raise ValueError("materialization Domain does not match its key")
        frame = self._value_plan(manifest)
        # This is an already published typed plan. Defer Parquet schema/value
        # I/O until a caller actually consumes it.
        value = Node._from_plan(frame, domain=domain, name=key.node_id, metadata=None,
            value_type=manifest["value_type"], identity=key.identity, dense_output=False,
            trace_columns=manifest["trace_columns"], trace_identity=manifest["trace_identity"])
        return NodeMaterialization(key, value, artifacts=self._decode(manifest["artifacts"]) if evidence else {},
            checkpoint=self._decode(manifest["checkpoint"]) if evidence else None,
            training_audits=tuple(self._decode(manifest["training_audits"])) if evidence else ())

    def query(self, key: MaterializationKey) -> MaterializationLookup:
        exact = self.lookup(key)
        if exact is not None:
            return MaterializationLookup(MaterializationStatus.HIT, exact)
        if not self.meta_path.is_file():
            return MaterializationLookup(MaterializationStatus.MISS)
        with self.read_transaction() as connection:
            row = connection.execute("SELECT key_json FROM materializations WHERE node_id=? ORDER BY identity LIMIT 1", (key.node_id,)).fetchone()
        if row is not None:
            return MaterializationLookup(MaterializationStatus.PARTIAL, None, "different context or coverage; reuse requires proof")
        return MaterializationLookup(MaterializationStatus.MISS)

    def publish(self, value: NodeMaterialization, *, execution_policy=None,
                value_references: Sequence[str | Mapping[str, Any]] | None = None, _proof_rebinding=None) -> None:
        self.initialize()
        if value.panel.domain.signature != value.key.domain_identity:
            raise ValueError("materialization Domain does not match its key")
        generation = uuid.uuid4().hex
        relative = f"values/{value.key.identity}/{generation}"
        temporary = self.artifact_path / ".pending" / generation
        final = self._file(relative)
        temporary.mkdir(parents=True)
        counter = 0
        artifact_sources = []
        value_sources = []

        def encode(item, *, dated_artifact=False):
            nonlocal counter
            if isinstance(item, ArtifactReference):
                admitted = self.compose_artifact(item.identity, item.name, item.tail)
                source = self.manifest(item.identity)
                if source["content_identity"] != item.content_identity or admitted.content_identity != item.content_identity:
                    raise ValueError("original artifact authority changed before publication")
                if source["node_id"] != value.key.node_id or source["key"]["implementation_id"] != value.key.implementation_id:
                    raise ValueError("artifact composition numerical owner differs")
                if (date.fromisoformat(source["through"]) > value.panel.domain.times.max()
                        or item.tail is not None and item.tail.height and item.tail["time"].max() > value.panel.domain.times.max()):
                    raise ValueError("artifact composition exceeds completed output coverage")
                original = source["artifacts"][item.name]
                artifact_sources.append((item.identity, source["content_identity"]))
                leaves = list(original.get("$artifact_concat", [original]))
                if item.tail is not None and item.tail.height:
                    leaves.append(encode(item.tail, dated_artifact=True))
                for leaf in leaves:
                    if not self._file(leaf["$file"]).is_file() and not leaf["$file"].startswith(relative + "/"):
                        raise ValueError("original artifact file missing before publication")
                return {"$artifact_concat": leaves, "schema": original["schema"], "unique_time": True,
                    "original_source": {"identity": item.identity, "name": item.name,
                        "source_content": source["content_identity"], "artifact_identity": hash_mapping(original)},
                    "rows": sum(leaf["rows"] for leaf in leaves),
                    "start": next((leaf["start"] for leaf in leaves if leaf["start"] is not None), None),
                    "end": next((leaf["end"] for leaf in reversed(leaves) if leaf["end"] is not None), None)}
            if isinstance(item, pl.LazyFrame):
                item = item.collect()
            if isinstance(item, pl.DataFrame):
                filename = f"{counter:06d}.parquet"
                counter += 1
                path = temporary / filename
                item.write_parquet(path)
                result = {"$file": f"{relative}/{filename}", "sha256": _checksum(path)}
                if dated_artifact and item.schema.get("time") == pl.Date and not item["time"].null_count():
                    result.update(schema={name: str(dtype) for name, dtype in item.schema.items()}, rows=item.height,
                        unique_time=item["time"].n_unique() == item.height and item["time"].is_sorted(),
                        start=None if not item.height else str(item["time"].min()),
                        end=None if not item.height else str(item["time"].max()),
                        days={str(day["time"][0]): hash_dataframe(day) for day in item.partition_by("time", maintain_order=True)},
                        day_rows={str(day["time"][0]): day.height for day in item.partition_by("time", maintain_order=True)})
                return result
            if isinstance(item, datetime):
                return {"$datetime": item.isoformat()}
            if isinstance(item, date):
                return {"$date": item.isoformat()}
            if isinstance(item, Mapping):
                return {str(key): encode(child, dated_artifact=dated_artifact) for key, child in item.items()}
            if isinstance(item, (tuple, list)):
                return [encode(child, dated_artifact=dated_artifact) for child in item]
            if isinstance(item, float):
                import math
                if not math.isfinite(item):
                    return {"$float": str(item)}
            return item

        try:
            domain = value.panel.domain
            plan = value.panel.lazy(dense=False, include_traces=True)
            calendar = domain.times
            partitions, days = [], {}
            domain_index = None
            from . import computation_index
            if value_references is not None:
                domain_index = computation_index.describe(domain, days={}, value_type=value.panel.value_type.value,
                    trace_columns=value.panel.trace_columns)
                covered = []
                for reference in value_references:
                    if isinstance(reference, str):
                        identity, lower, upper = reference, None, None
                    elif isinstance(reference, Mapping) and set(reference) == {"identity", "start", "end"}:
                        identity = reference["identity"]
                        lower = date.fromisoformat(str(reference["start"]))
                        upper = date.fromisoformat(str(reference["end"]))
                        if lower > upper:
                            raise ValueError("value composition interval is reversed")
                    else:
                        raise ValueError("value references require an identity and exact start/end")
                    source = self.manifest(identity)
                    if source is None:
                        raise ValueError("value composition requires registered original materializations")
                    if (source["node_id"] != value.key.node_id
                            or source["key"]["implementation_id"] != value.key.implementation_id
                            or source["value_type"] != value.panel.value_type.value
                            or source["trace_columns"] != list(value.panel.trace_columns)):
                        raise ValueError("value composition numerical contract differs")
                    with self.read_transaction() as connection:
                        index = computation_index.read(connection, source)
                    if index is None:
                        raise ValueError("value composition needs explicit Core interval index maintenance")
                    selected_days = [day for day in index["calendar"]
                        if (lower is None or day >= lower.isoformat()) and (upper is None or day <= upper.isoformat())]
                    if lower is not None and (not selected_days or selected_days[0] != lower.isoformat()
                            or selected_days[-1] != upper.isoformat()):
                        raise ValueError("value composition interval requires original calendar bounds")
                    for day in selected_days:
                        if domain_index["membership"].get(day) != index["membership"][day]:
                            raise ValueError("value composition Domain membership differs")
                    covered.extend(selected_days)
                    for part in source["partitions"]:
                        part_start = part.get("start")
                        part_end = part.get("end")
                        if not selected_days or part_end is not None and part_end < selected_days[0] or part_start is not None and part_start > selected_days[-1]:
                            continue
                        partitions.append({**part,
                            "start": max(part_start or selected_days[0], selected_days[0]),
                            "end": min(part_end or selected_days[-1], selected_days[-1])})
                    days.update({day: token for day, token in source["days"].items() if day in selected_days})
                    value_sources.append({"identity": identity, "source_content": source["content_identity"],
                        "start": selected_days[0] if selected_days else None, "end": selected_days[-1] if selected_days else None})
                if covered != domain_index["calendar"]:
                    raise ValueError("value composition must cover the complete ordered calendar exactly once")
            else:
                months = calendar.dt.truncate("1mo").unique().sort()
                for month in months:
                    dates = calendar.filter(calendar.dt.truncate("1mo") == month)
                    frame = value.panel._validate_collected(plan.filter(pl.col("time").is_between(dates.min(), dates.max())).collect())
                    partitions.append({**encode(frame), "start": str(dates.min()), "end": str(dates.max()), "rows": frame.height})
                    for day_frame in frame.partition_by("time", maintain_order=True):
                        days[str(day_frame["time"][0])] = hash_dataframe(day_frame)
                if not partitions:
                    partitions.append(encode(plan.collect()))
            indexed = {**domain_index, "days": days} if domain_index is not None else computation_index.describe(
                domain, days=days, value_type=value.panel.value_type.value, trace_columns=value.panel.trace_columns)
            with self.read_transaction() as connection:
                prior_domain = self._registered_domain(connection, domain.signature)
            saved_domain = None
            if prior_domain is not None:
                existing_domain = self.manifest(prior_domain[0])
                with self.read_transaction() as connection:
                    previous_index = computation_index.read(connection, existing_domain)
                if previous_index is None:
                    raise ValueError("Registered Domain identity needs explicit Core interval-index maintenance")
                if previous_index is not None:
                    if (previous_index["calendar"] != indexed["calendar"]
                            or previous_index["membership"] != indexed["membership"]):
                        raise ValueError("registered Domain identity differs from actual calendar/membership")
                    saved_domain = existing_domain["domain"]
            manifest = {"schema": SCHEMA_VERSION, "key": value.key.to_dict(), "value_type": value.panel.value_type.value,
                "trace_columns": list(value.panel.trace_columns), "trace_identity": value.panel.trace_identity,
                "domain": saved_domain or {"dynamic": domain.is_dynamic, "calendar": encode(pl.DataFrame({"time": domain.times})),
                    "membership": encode(domain.grid_lazy().collect() if domain.is_dynamic else pl.DataFrame({"asset_id": domain.asset_ids}))},
                "identity": value.key.identity, "node_id": value.key.node_id,
                "through": str(calendar.max()), "coverage_start": str(calendar.min()),
                "days": days, "logical_hash": hash_mapping(days),
                "partitions": partitions, "artifacts": encode(value.artifacts, dated_artifact=True),
                "checkpoint": encode(value.checkpoint), "training_audits": encode(value.training_audits),
                "execution_policy": execution_policy}
            if value_sources:
                manifest["value_sources"] = value_sources
            if _proof_rebinding is not None:
                manifest["proof_rebinding"] = _proof_rebinding
            manifest["content_identity"] = hash_mapping(_manifest_content(manifest))
            serialized = _json(manifest)
            final.parent.mkdir(parents=True, exist_ok=True)
            with self.transaction() as connection:
                for identity, content_identity in artifact_sources:
                    row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (identity,)).fetchone()
                    if row is None:
                        raise ValueError("original artifact authority missing at publication")
                    original = json.loads(row[0])
                    self._verify_metadata(original, identity)
                    if original["content_identity"] != content_identity:
                        raise ValueError("original artifact authority changed at publication")
                for source in value_sources:
                    row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (source["identity"],)).fetchone()
                    if row is None:
                        raise ValueError("original value authority is missing at publication")
                    original = json.loads(row[0])
                    self._verify_metadata(original, source["identity"])
                    if original["content_identity"] != source["source_content"]:
                        raise ValueError("original value authority changed at publication")
                def require_artifact_files(item):
                    if isinstance(item, Mapping):
                        if "$file" in item and not item["$file"].startswith(relative + "/") and not self._file(item["$file"]).is_file():
                            raise ValueError("original artifact file missing at publication")
                        for child in item.values():
                            require_artifact_files(child)
                    elif isinstance(item, (tuple, list)):
                        for child in item:
                            require_artifact_files(child)
                require_artifact_files(manifest["artifacts"])
                require_artifact_files(manifest["partitions"])
                self._verify_artifact_sources(connection, manifest)
                if _proof_rebinding is not None:
                    self._verify_proof_rebinding(connection, _proof_rebinding, value.key, execution_policy)
                connection.execute(computation_index.DDL)
                registered_domain = self._registered_domain(connection, domain.signature)
                if registered_domain is not None:
                    registered = json.loads(registered_domain[1])
                    if (registered["identity"] != MaterializationKey.from_dict(registered["key"]).identity
                            or registered["content_identity"] != hash_mapping(_manifest_content(registered))):
                        raise ValueError("registered Domain metadata identity differs")
                    registered_index = computation_index.read(connection, registered)
                    if registered_index is None:
                        raise ValueError("Registered Domain identity needs explicit Core interval-index maintenance")
                    if (registered_index["calendar"] != indexed["calendar"]
                            or registered_index["membership"] != indexed["membership"]):
                        raise ValueError("registered Domain identity differs from actual calendar/membership")
                old = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (value.key.identity,)).fetchone()
                if old is not None:
                    existing = json.loads(old[0])
                    if _result_content(existing) != _result_content(manifest):
                        raise ValueError("immutable materialization conflict: different results or evidence for the same key")
                    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='computation_index'").fetchone():
                        computation_index.save(connection, value.key.identity, existing["content_identity"], indexed)
                    return
                os.replace(temporary, final)
                connection.execute("INSERT INTO materializations VALUES(?,?,?,?)",
                    (value.key.identity, value.key.node_id, _json(value.key.to_dict()), serialized))
                if connection.execute("SELECT 1 FROM sqlite_master WHERE name='computation_index'").fetchone():
                    computation_index.save(connection, value.key.identity, manifest["content_identity"], indexed)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def read(self, identity: str) -> NodeMaterialization:
        manifest = self.manifest(identity)
        if manifest is None:
            raise KeyError(f"unknown Core materialization: {identity}")
        return self.lookup(MaterializationKey.from_dict(manifest["key"]))

    @staticmethod
    def _registered_domain(connection, signature):
        """Select available registered coordinate evidence at both publication fences."""
        from .computation_index import VERSION
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='computation_index'").fetchone():
            # Older missing optional indexes do not invalidate another original
            # receipt's available proof. Its binding, payload and coordinates
            # are still validated normally; no index is warmed implicitly.
            return connection.execute("SELECT m.identity,m.manifest_json FROM materializations m "
                "LEFT JOIN computation_index i ON i.identity=m.identity "
                "WHERE json_extract(m.key_json,'$.domain_identity')=? "
                "ORDER BY CASE WHEN i.version=? AND i.content_identity="
                "json_extract(m.manifest_json,'$.content_identity') THEN 0 ELSE 1 END,m.rowid LIMIT 1",
                (signature, VERSION)).fetchone()
        return connection.execute("SELECT identity,manifest_json FROM materializations "
            "WHERE json_extract(key_json,'$.domain_identity')=? ORDER BY rowid LIMIT 1", (signature,)).fetchone()

    def inventory(self, *, node_id: str | None = None,
                  implementation_id: str | None = None) -> tuple[dict[str, Any], ...]:
        """Read registered numerical keys, optionally scoped before decoding."""
        if not self.meta_path.is_file():
            return ()
        clauses, parameters = [], []
        if node_id is not None:
            clauses.append("node_id=?")
            parameters.append(node_id)
        if implementation_id is not None:
            clauses.append("json_extract(key_json,'$.implementation_id')=?")
            parameters.append(implementation_id)
        with self.read_transaction() as connection:
            rows = connection.execute("SELECT identity,node_id,key_json FROM materializations"
                + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY identity",
                parameters).fetchall()
        return tuple({"identity": row[0], "node_id": row[1], "key": json.loads(row[2])} for row in rows)

    def describe(self, identity: str) -> dict[str, Any] | None:
        """Public metadata receipt containing no artifact locations."""
        manifest = self.manifest(identity)
        if manifest is None:
            return None
        from .computation_index import read
        with self.read_transaction() as connection:
            indexed = read(connection, manifest)
        return {**{key: manifest[key] for key in ("identity", "content_identity", "node_id", "key", "value_type", "through",
            "coverage_start", "days", "logical_hash", "trace_columns", "execution_policy")},
            "calendar": None if indexed is None else indexed["calendar"]}

    def interval_identity(self, identity: str, *, start, end) -> str | None:
        """Prove selected values, full calendar and membership from index records."""
        from .computation_index import read, interval
        manifest = self.manifest(identity)
        if manifest is None:
            return None
        with self.read_transaction() as connection:
            indexed = read(connection, manifest)
        return None if indexed is None else interval(indexed, start, end)

    def _artifact_channel(self, identity, name):
        visited = set()
        def visit(current):
            if current in visited:
                return None
            visited.add(current)
            manifest = self.manifest(current)
            if manifest is None:
                return None
            if name in manifest["artifacts"]:
                return manifest, manifest["artifacts"][name]
            for parent in manifest["key"]["inputs"]:
                found = visit(parent)
                if found is not None:
                    return found
            return None
        return visit(identity)

    def describe_artifact(self, identity: str, name: str) -> dict[str, Any] | None:
        """Describe the same original channel selected by recursive evidence reads."""
        found = self._artifact_channel(identity, name)
        if found is None:
            return None
        manifest, artifact = found
        with self.read_transaction() as connection:
            indexed = self._indexed_artifact(connection, manifest, name)
        artifact = indexed or artifact
        if not isinstance(artifact, Mapping) or "schema" not in artifact:
            return {"identity": manifest["identity"], "content_identity": manifest["content_identity"], "name": name, "indexed": False}
        return {"identity": manifest["identity"], "content_identity": manifest["content_identity"],
            "name": name, **{key: artifact.get(key) for key in ("schema", "rows", "start", "end", "unique_time")},
            "indexed": indexed is not None}

    def _indexed_artifact(self, connection, manifest, name):
        artifact = manifest["artifacts"][name]
        if not isinstance(artifact, Mapping):
            return None
        leaves = artifact.get("$artifact_concat", [artifact])
        if all("days" in leaf and "day_rows" in leaf for leaf in leaves):
            return artifact
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='artifact_interval_index'").fetchone():
            row = connection.execute("SELECT source_content,artifact_identity,projection_json FROM artifact_interval_index WHERE identity=? AND name=?",
                (manifest["identity"], name)).fetchone()
            if row is not None:
                if row[0] != manifest["content_identity"] or row[1] != hash_mapping(artifact):
                    raise ValueError("Core artifact interval index original binding differs")
                projection = json.loads(row[2])
                payload = {key: value for key, value in projection.items() if key != "digest"}
                if projection.get("digest") != hash_mapping(payload):
                    raise ValueError("corrupt Core artifact interval index")
                return projection["artifact"]
        if "$artifact_concat" in artifact:
            source = artifact["original_source"]
            original = self.manifest(source["identity"])
            if original is None:
                raise ValueError("Core artifact interval index original authority missing")
            before = self._indexed_artifact(connection, original, source["name"])
            if before is not None:
                prefix = before.get("$artifact_concat", [before])
                tail = leaves[len(prefix):]
                if all("days" in leaf and "day_rows" in leaf for leaf in tail):
                    return {**artifact, "$artifact_concat": [*prefix, *tail]}
        return None

    def artifact_index_plan(self, identity: str, name: str) -> dict[str, Any]:
        """Freeze explicit original-channel index maintenance, without decoding."""
        found = self._artifact_channel(identity, name)
        if found is None:
            raise KeyError(f"unknown Core artifact channel: {name}")
        manifest, artifact = found
        return {"version": "core.artifact.index.v1", "meta_path": str(self.meta_path),
            "identity": manifest["identity"], "name": name, "source_content": manifest["content_identity"],
            "artifact_identity": hash_mapping(artifact)}

    def build_artifact_index(self, plan, *, check_canceled=lambda: None) -> dict[str, Any]:
        """Explicitly audit a retained artifact and register its bounded-read proof."""
        if plan.get("version") != "core.artifact.index.v1" or plan.get("meta_path") != str(self.meta_path):
            raise ValueError("Core artifact index plan belongs to another authority/version")
        check_canceled()
        manifest = self.manifest(plan["identity"])
        if (manifest is None or manifest["content_identity"] != plan["source_content"]
                or hash_mapping(manifest["artifacts"].get(plan["name"])) != plan["artifact_identity"]):
            raise RevisionConflict("Core artifact index original authority changed")
        original = manifest["artifacts"][plan["name"]]
        if not isinstance(original, Mapping):
            raise ValueError("Core artifact index requires an original table")
        leaves, schema = [], None
        for leaf in original.get("$artifact_concat", [original]):
            check_canceled()
            path = self._file(leaf["$file"])
            from .resources import admit_parquet_materialization
            if not admit_parquet_materialization(path):
                return {"status": "unknown", "reason": "artifact exceeds explicit maintenance materialization budget"}
            if _checksum(path) != leaf["sha256"]:
                raise ValueError("corrupt original Core artifact")
            frame = pl.read_parquet(path)
            if frame.schema.get("time") != pl.Date or frame["time"].null_count():
                raise ValueError("Core artifact index requires a nonnull Date time")
            current_schema = {key: str(dtype) for key, dtype in frame.schema.items()}
            if schema is not None and schema != current_schema:
                raise ValueError("Core artifact index original leaf schemas differ")
            schema = current_schema
            leaves.append({**leaf, "schema": schema, "rows": frame.height,
                "unique_time": frame["time"].n_unique() == frame.height and frame["time"].is_sorted(),
                "start": None if frame.is_empty() else str(frame["time"].min()),
                "end": None if frame.is_empty() else str(frame["time"].max()),
                "days": {str(day["time"][0]): hash_dataframe(day) for day in frame.partition_by("time", maintain_order=True)},
                "day_rows": {str(day["time"][0]): day.height for day in frame.partition_by("time", maintain_order=True)}})
        if not leaves:
            raise ValueError("Core artifact index requires original typed leaves")
        artifact = leaves[0] if "$artifact_concat" not in original else {**original,
            "$artifact_concat": leaves, "schema": schema, "rows": sum(leaf["rows"] for leaf in leaves)}
        projection = {"version": plan["version"], "source_content": plan["source_content"],
            "artifact_identity": plan["artifact_identity"], "artifact": artifact}
        projection["digest"] = hash_mapping(projection)
        with self.transaction() as connection:
            check_canceled()
            row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (plan["identity"],)).fetchone()
            if row is None or json.loads(row[0])["content_identity"] != plan["source_content"]:
                raise RevisionConflict("Core artifact index original authority changed")
            connection.execute("CREATE TABLE IF NOT EXISTS artifact_interval_index(identity TEXT NOT NULL,name TEXT NOT NULL,source_content TEXT NOT NULL,artifact_identity TEXT NOT NULL,projection_json TEXT NOT NULL,PRIMARY KEY(identity,name))")
            connection.execute("INSERT OR REPLACE INTO artifact_interval_index VALUES(?,?,?,?,?)",
                (plan["identity"], plan["name"], plan["source_content"], plan["artifact_identity"], _json(projection)))
        return {"status": "complete", "identity": plan["identity"], "name": plan["name"], "rows": artifact["rows"]}

    def artifact_interval_identity(self, identity: str, name: str, *, start, end) -> str | None:
        """Prove all original dated artifact columns, including decision reasons.

        This uses original-bound indexes and complete calendar coverage only.
        Missing optional authority returns unknown without reading numerical
        files. Empty artifact dates remain bound to the original calendar axis.
        """
        lower, upper = date.fromisoformat(str(start)), date.fromisoformat(str(end))
        if lower > upper:
            raise ValueError("artifact interval is reversed")
        found = self._artifact_channel(identity, name)
        if found is None:
            return None
        manifest, artifact = found
        with self.read_transaction() as connection:
            artifact = self._indexed_artifact(connection, manifest, name)
        if artifact is None:
            return None
        if not isinstance(artifact, Mapping) or artifact.get("schema", {}).get("time") != "Date":
            return None
        leaves = artifact.get("$artifact_concat", [artifact])
        days = {}
        for leaf in leaves:
            if "days" not in leaf:
                return None
            for day, token in leaf["days"].items():
                if lower.isoformat() <= day <= upper.isoformat():
                    if day in days:
                        raise ValueError("artifact interval has overlapping original coordinates")
                    days[day] = token
        from .computation_index import read
        with self.read_transaction() as connection:
            indexed = read(connection, manifest)
        if indexed is None:
            return None
        calendar = [day for day in indexed["calendar"] if lower.isoformat() <= day <= upper.isoformat()]
        if (not indexed["calendar"] or lower.isoformat() < indexed["calendar"][0]
                or upper.isoformat() > indexed["calendar"][-1]):
            return None
        return hash_mapping({"version": "core.artifact.interval.v1", "name": name,
            "schema": artifact["schema"], "start": lower.isoformat(), "end": upper.isoformat(),
            "calendar": calendar, "days": days})

    def read_artifact(self, identity: str, name: str, *, start=None, end=None) -> pl.DataFrame:
        """Read one dated artifact with leaf pruning and Parquet predicate pushdown."""
        found = self._artifact_channel(identity, name)
        if found is None:
            raise KeyError(f"unknown Core artifact channel: {name}")
        manifest, artifact = found
        with self.read_transaction() as connection:
            artifact = self._indexed_artifact(connection, manifest, name)
        if artifact is None:
            raise ValueError("bounded artifact reads need explicit dated-artifact index maintenance")
        if not isinstance(artifact, Mapping) or artifact.get("schema", {}).get("time") != "Date":
            raise ValueError("bounded artifact reads need explicit dated-artifact index maintenance")
        lower = None if start is None else date.fromisoformat(str(start))
        upper = None if end is None else date.fromisoformat(str(end))
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("artifact read interval is reversed")
        frames = []
        for leaf in artifact.get("$artifact_concat", [artifact]):
            if "days" not in leaf:
                raise ValueError("bounded artifact reads need explicit dated-artifact index maintenance")
            selected = {day: token for day, token in leaf["days"].items()
                if (lower is None or day >= lower.isoformat()) and (upper is None or day <= upper.isoformat())}
            if not selected:
                continue
            plan = pl.scan_parquet(self._file(leaf["$file"]))
            if lower is not None:
                plan = plan.filter(pl.col("time") >= lower)
            if upper is not None:
                plan = plan.filter(pl.col("time") <= upper)
            frame = plan.collect()
            if {column: str(dtype) for column, dtype in frame.schema.items()} != artifact["schema"]:
                raise ValueError("Core artifact typed schema differs")
            if set(str(day) for day in frame["time"].unique()) != set(selected):
                raise ValueError("Core artifact selected coordinate coverage differs")
            if frame.height != sum(leaf["day_rows"][day] for day in selected):
                raise ValueError("Core artifact selected row count differs")
            frames.append(frame)
        if frames:
            return pl.concat(frames)
        try:
            return pl.DataFrame(schema={key: getattr(pl, dtype) for key, dtype in artifact["schema"].items()})
        except (AttributeError, TypeError) as error:
            raise ValueError("Core artifact selection needs supported indexed types") from error

    def domain_matches(self, identity: str, domain: Domain) -> bool | None:
        """Compare actual complete coordinates against original indexed evidence.

        Missing optional evidence returns unknown; this never decodes saved
        Domain files or warms an index. Caller-supplied Domain IDs are not
        coordinate evidence.
        """
        from .computation_index import read, describe
        manifest = self.manifest(identity)
        if manifest is None:
            return None
        with self.read_transaction() as connection:
            indexed = read(connection, manifest)
        if indexed is None:
            return None
        if (indexed.get("domain_identity") != manifest["key"]["domain_identity"]
                or indexed.get("value_type") != manifest["value_type"]
                or indexed.get("trace_columns") != manifest["trace_columns"]):
            raise ValueError("corrupt Core Domain index binding")
        actual = describe(domain, days={}, value_type=manifest["value_type"],
                          trace_columns=manifest["trace_columns"])
        return (indexed["calendar"] == actual["calendar"]
                and indexed["membership"] == actual["membership"])

    def plan_input_proof_rebind(self, spec, identity: str, **source_bindings) -> dict[str, Any]:
        """Explicitly plan source-format maintenance against original projections.

        See ConsumedInputEvidence for caller/source-owner authority. This
        explicit maintenance operation registers immutable plan metadata after
        validation; it is not a read-only prerequisite or preview. It never
        infers equivalence from saved output values or changes original receipts.
        """
        from .proof_maintenance import plan_rebinding
        return plan_rebinding(self, spec, identity, **source_bindings)

    def apply_input_proof_rebind(self, plan, *, check_canceled=lambda: None) -> dict[str, Any]:
        """Admit a planned reference-only proof update, preserving originals."""
        from .proof_maintenance import apply_rebinding
        return apply_rebinding(self, plan, check_canceled=check_canceled)

    def index_plan(self) -> dict[str, Any]:
        """Freeze explicit historical proof-index maintenance using metadata."""
        from .computation_index import VERSION
        return {"version": VERSION, "meta_path": str(self.meta_path), "identities": [
            {"identity": item["identity"], "content_identity": self.manifest(item["identity"])["content_identity"]}
            for item in self.inventory()]}

    def build_index(self, plan, *, check_canceled=lambda: None, progress=lambda *_: None):
        from . import computation_index
        from .resources import admit_parquet_materialization
        if plan.get("version") != computation_index.VERSION or plan.get("meta_path") != str(self.meta_path):
            raise ValueError("Core computation index plan belongs to another authority/version")
        if not plan["identities"]:
            return {"status": "complete", "materializations": 0, "unknown_materializations": 0}
        completed, unknown = 0, 0
        with self.transaction() as connection:
            connection.execute(computation_index.DDL)
        for position, item in enumerate(plan["identities"], 1):
            check_canceled()
            manifest = self.manifest(item["identity"])
            if manifest is None or manifest["content_identity"] != item["content_identity"]:
                raise RevisionConflict("Core index plan changed")
            self.verify(manifest)
            if not all(admit_parquet_materialization(self._file(manifest["domain"][name]["$file"]))
                       for name in ("calendar", "membership")):
                unknown += 1
                progress(position, len(plan["identities"]))
                continue
            value = self.lookup(MaterializationKey.from_dict(manifest["key"]), evidence=False)
            indexed = computation_index.describe(value.panel.domain, days=manifest["days"],
                value_type=manifest["value_type"], trace_columns=manifest["trace_columns"])
            with self.transaction() as connection:
                check_canceled()
                current = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (item["identity"],)).fetchone()
                if current is None or json.loads(current[0])["content_identity"] != item["content_identity"]:
                    raise RevisionConflict("Core index plan changed")
                computation_index.save(connection, item["identity"], item["content_identity"], indexed)
            completed += 1
            progress(position, len(plan["identities"]))
        return {"status": "partial" if unknown else "complete", "materializations": completed,
                "unknown_materializations": unknown}

    def verify_identity(self, identity: str) -> None:
        manifest = self.manifest(identity)
        if manifest is None:
            raise ValueError(f"missing Core materialization: {identity}")
        self.verify(manifest)
        from . import computation_index
        with self.read_transaction() as connection:
            indexed = computation_index.read(connection, manifest)
            if indexed is None and connection.execute("SELECT 1 FROM sqlite_master WHERE name='computation_index'").fetchone():
                row = connection.execute("SELECT version FROM computation_index WHERE identity=?", (identity,)).fetchone()
                if row is not None and row[0] == computation_index.VERSION:
                    raise ValueError("Core computation index source binding differs")
        if indexed is not None:
            value = self.lookup(MaterializationKey.from_dict(manifest["key"]), evidence=False)
            expected = computation_index.describe(value.panel.domain, days=manifest["days"],
                value_type=manifest["value_type"], trace_columns=manifest["trace_columns"])
            if indexed != expected:
                raise ValueError("Core computation index differs from original evidence")

    def read_values(self, identity: str, *, start=None, end=None, include_traces=False) -> pl.DataFrame:
        """Decode only selected typed value partitions and their Domain."""
        try:
            return self._read_values(identity, start=start, end=end, include_traces=include_traces)
        except pl.exceptions.PolarsError as error:
            raise ValueError("corrupt or unreadable Core artifact") from error

    def _value_plan(self, manifest, *, lower=None, upper=None):
        """Apply every retained original interval before Parquet decompression."""
        frames = []
        for part in manifest["partitions"]:
            start = None if part.get("start") is None else date.fromisoformat(part["start"])
            end = None if part.get("end") is None else date.fromisoformat(part["end"])
            if lower is not None:
                start = lower if start is None else max(lower, start)
            if upper is not None:
                end = upper if end is None else min(upper, end)
            if start is not None and end is not None and start > end:
                continue
            frame = pl.scan_parquet(self._file(part["$file"]))
            if start is not None:
                frame = frame.filter(pl.col("time") >= start)
            if end is not None:
                frame = frame.filter(pl.col("time") <= end)
            frames.append(frame)
        if frames:
            return pl.concat(frames)
        return pl.DataFrame(schema=pl.read_parquet_schema(self._file(manifest["partitions"][0]["$file"]))).lazy()

    def _read_domain_rows(self, reference, *, expected_schema, lower=None, upper=None) -> pl.DataFrame:
        """Prune canonical observation keys before eager Domain validation."""
        frame = pl.scan_parquet(self._file(reference["$file"]))
        if frame.collect_schema() != pl.Schema(expected_schema):
            raise ValueError("Core Domain artifact schema differs from its published type")
        if "time" in expected_schema:
            selected = pl.lit(True)
            if lower is not None:
                selected &= pl.col("time") >= lower
            if upper is not None:
                selected &= pl.col("time") <= upper
            # Invalid observation keys have no provable scope; do not hide them.
            frame = frame.filter(pl.col("time").is_null() | selected)
        result = frame.collect()
        if result.select(pl.any_horizontal(pl.all().is_null()).any()).item():
            raise ValueError("Core Domain keys must be valid")
        return result

    def _read_values(self, identity: str, *, start=None, end=None, include_traces=False) -> pl.DataFrame:
        if self._verification is None:
            with self.read_context() as reader:
                return reader.read_values(identity, start=start, end=end, include_traces=include_traces)
        manifest = self.manifest(identity)
        if manifest is None:
            raise KeyError(f"unknown Core materialization: {identity}")
        lower = None if start is None else date.fromisoformat(str(start))
        upper = None if end is None else date.fromisoformat(str(end))
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("end precedes start")
        saved = manifest["domain"]
        bounded = lower is not None or upper is not None
        if bounded:
            calendar = self._read_domain_rows(saved["calendar"], expected_schema={"time": pl.Date},
                lower=lower, upper=upper)["time"]
            membership = self._read_domain_rows(saved["membership"], expected_schema=(
                {"time": pl.Date, "asset_id": pl.String} if saved["dynamic"] else {"asset_id": pl.String}),
                lower=lower, upper=upper)
        else:
            decoded_calendar = self._decode(saved["calendar"])
            membership = self._decode(saved["membership"])
            if not isinstance(decoded_calendar, pl.DataFrame) or not isinstance(membership, pl.DataFrame):
                raise ValueError("Core Domain artifacts must be typed tables")
            calendar = decoded_calendar["time"]
        frame = self._value_plan(manifest, lower=lower, upper=upper)
        if calendar.is_empty():
            # Domain requires a nonempty calendar. Validate the consumed schema
            # and return an empty typed view without inventing an observation.
            if pl.scan_parquet(self._file(saved["calendar"]["$file"])).select(pl.len()).collect().item() == 0:
                raise ValueError("calendar must contain at least one time")
            if not saved["dynamic"]:
                from .frame import normalize_asset_ids
                normalize_asset_ids(membership["asset_id"])
            elif membership.select(pl.struct("time", "asset_id").is_duplicated().any()).item():
                raise ValueError("dynamic universe must be unique by (time, asset_id)")
            empty = Node._normalize_source(frame, manifest["trace_columns"],
                value_type=ValueType(manifest["value_type"])).filter(pl.lit(False)).collect()
            return empty.select("time", "asset_id", "value",
                *(manifest["trace_columns"] if include_traces else ()))
        universe = membership.with_columns(pl.lit(True).alias("active")) if saved["dynamic"] else membership["asset_id"]
        domain = Domain(calendar=calendar, universe=universe, identity=manifest["key"]["domain_identity"])
        panel = Node.from_domain(frame, domain, name=manifest["node_id"],
            value_type=manifest["value_type"], trace_columns=manifest["trace_columns"],
            trace_identity=manifest["trace_identity"])
        result = panel.collect(dense=False, include_traces=include_traces)
        return result

    def evidence(self, identity: str, *, channels=None) -> dict[str, Any]:
        """Read typed auxiliary artifacts, checkpoints and fit audits recursively."""
        if self._verification is None:
            with self.read_context() as reader:
                return reader.evidence(identity, channels=channels)
        visited, audits, checkpoints, artifacts = set(), [], {}, {}
        def visit(current):
            if current in visited or self.manifest(current) is None:
                return
            visited.add(current)
            manifest = self.manifest(current)
            for name, value in manifest["artifacts"].items():
                if channels is None or name in channels:
                    artifacts.setdefault(name, self._decode(value))
            if channels is None or "training_audits" in channels:
                audits.extend({**audit, "node_id": manifest["node_id"]} for audit in self._decode(manifest["training_audits"]))
            if manifest["checkpoint"] is not None and (channels is None or "checkpoints" in channels):
                checkpoints[manifest["node_id"]] = self._decode(manifest["checkpoint"])
            for parent in manifest["key"]["inputs"]:
                visit(parent)
        visit(identity)
        return {"artifacts": artifacts, "checkpoints": checkpoints, "training_audits": tuple(audits)}

    def describe_evidence(self, identity: str) -> dict[str, Any]:
        """Describe published auxiliary channels without decoding their values."""
        visited, artifacts, checkpoints, audits = set(), set(), set(), 0
        def visit(current):
            nonlocal audits
            if current in visited:
                return
            visited.add(current)
            manifest = self.manifest(current)
            if manifest is None:
                return
            artifacts.update(manifest["artifacts"])
            audits += len(manifest["training_audits"])
            if manifest["checkpoint"] is not None:
                checkpoints.add(manifest["node_id"])
            for parent in manifest["key"]["inputs"]:
                visit(parent)
        visit(identity)
        return {"artifacts": tuple(sorted(artifacts)), "checkpoint_nodes": tuple(sorted(checkpoints)),
            "training_audit_count": audits}

    def check_integrity(self, identities=None) -> tuple[dict[str, Any], ...]:
        selected = [item["identity"] for item in self.inventory()] if identities is None else identities
        results = []
        for identity in selected:
            try:
                self.verify_identity(identity)
                results.append({"identity": identity, "valid": True})
            except (ValueError, OSError) as error:
                results.append({"identity": identity, "valid": False, "reason": str(error)})
        return tuple(results)

    def cleanup_plan(self, *, tokens=None, created_before=None) -> dict[str, Any]:
        """Inspect completed unreferenced generations; retain every committed receipt.

        Pending writes and unknown storage are deliberately outside this plan.
        Numerical values named by any immutable receipt remain historical evidence.
        """
        if not self.meta_path.is_file():
            body = {"receipts": [], "candidates": []}
            return {"identity": hash_mapping(body), **body, "bytes": 0}
        with self.transaction() as connection:
            receipts = [dict(row) for row in connection.execute("SELECT identity,manifest_json FROM materializations ORDER BY identity")]
            referenced = set()
            def paths(item):
                if isinstance(item, dict):
                    if "$file" in item:
                        referenced.add(PurePosixPath(item["$file"]).parent.as_posix())
                    for value in item.values():
                        paths(value)
                elif isinstance(item, list):
                    for value in item:
                        paths(value)
            for receipt in receipts:
                paths(json.loads(receipt["manifest_json"]))
            candidates = []
            for directory in sorted((self.artifact_path/"values").glob("*/*")):
                relative = directory.relative_to(self.artifact_path).as_posix()
                if relative in referenced or directory.is_symlink() or not directory.is_dir():
                    continue
                if len(directory.name) != 32 or len(directory.parent.name) != 64:
                    continue
                files = list(directory.iterdir())
                if tokens is not None and relative not in tokens:
                    continue
                if not files or any(member.is_symlink() or not member.is_file() or member.suffix != ".parquet" for member in files):
                    continue
                if created_before is not None and any(member.stat().st_mtime > created_before.timestamp() for member in files):
                    continue
                candidates.append({"token": relative, "files": {member.name: _checksum(member) for member in sorted(files)},
                    "bytes": sum(member.stat().st_size for member in files)})
        body = {"receipts": [hash_mapping(receipt) for receipt in receipts], "candidates": candidates}
        return {"identity": hash_mapping(body), **body, "bytes": sum(item["bytes"] for item in candidates)}

    def apply_cleanup(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Apply only the exact reviewed Core plan, idempotently."""
        self.initialize()
        expected = hash_mapping({key: plan[key] for key in ("receipts", "candidates")})
        if expected != plan["identity"]:
            raise ValueError("Core cleanup plan identity mismatch")
        # A single writer boundary prevents concurrent publication from adopting
        # a generation while cleanup removes it.
        with self.transaction() as connection:
            old = connection.execute("SELECT result_json FROM cleanup_receipts WHERE identity=?", (expected,)).fetchone()
            if old is not None:
                return json.loads(old[0])
            receipts = [dict(row) for row in connection.execute("SELECT identity,manifest_json FROM materializations ORDER BY identity")]
            if [hash_mapping(receipt) for receipt in receipts] != plan["receipts"]:
                raise RevisionConflict("Core receipts changed after cleanup preview")
            referenced = set()
            def record_paths(item):
                if isinstance(item, dict):
                    if "$file" in item:
                        referenced.add(PurePosixPath(item["$file"]).parent.as_posix())
                    for child in item.values():
                        record_paths(child)
                elif isinstance(item, list):
                    for child in item:
                        record_paths(child)
            for receipt in receipts:
                record_paths(json.loads(receipt["manifest_json"]))
            import re
            for candidate in plan["candidates"]:
                token = candidate["token"]
                if not re.fullmatch(r"values/[0-9a-f]{64}/[0-9a-f]{32}", token) or token in referenced:
                    raise ValueError("cleanup cannot remove a referenced or non-Core generation")
                raw = self.artifact_path / token
                if any(path.is_symlink() for path in (raw, raw.parent, raw.parent.parent)):
                    raise ValueError("unsafe Core cleanup path")
                directory = self._file(token)
                if not directory.exists():
                    continue  # Resume an interrupted unlink while still holding the same proof.
                if directory.is_symlink() or any(member.is_symlink() for member in directory.iterdir()):
                    raise ValueError("unsafe Core cleanup generation")
                members = list(directory.iterdir())
                if any(not member.is_file() or member.is_symlink() for member in members):
                    raise RevisionConflict("Core generation membership changed after cleanup preview")
                files = {member.name: _checksum(member) for member in members}
                if any(candidate["files"].get(name) != digest for name, digest in files.items()):
                    raise RevisionConflict("Core generation changed after cleanup preview")
            for candidate in plan["candidates"]:
                directory = self._file(candidate["token"])
                if directory.exists():
                    shutil.rmtree(directory)
            result = {"identity": expected, "removed_generations": len(plan["candidates"]), "bytes": plan["bytes"]}
            connection.execute("INSERT INTO cleanup_receipts VALUES(?,?)", (expected, _json(result)))
        return result

    def graph_state(self, graph_id: str) -> dict[str, Any]:
        empty = {"revision": 0, "state_revision": 0, "spec": LogicalGraphSpec({}, ()),
            "statuses": {}, "contexts": {}, "current_update": None}
        if not self.meta_path.is_file():
            return empty
        with self.read_transaction() as connection:
            row = connection.execute("SELECT * FROM graphs WHERE graph_id=?", (graph_id,)).fetchone()
            if row is None:
                return empty
            status = dict(connection.execute("SELECT node_id,status FROM graph_nodes WHERE graph_id=?", (graph_id,)).fetchall())
            contexts = {item[0]: json.loads(item[1]) for item in connection.execute("SELECT context_id,definition_json FROM contexts WHERE graph_id=? ORDER BY context_id", (graph_id,))}
        return {"revision": row["revision"], "state_revision": row["state_revision"], "spec": LogicalGraphSpec.from_dict(json.loads(row["spec_json"])),
                "statuses": status, "contexts": contexts, "current_update": row["current_update"]}

    def save_graph(self, graph_id, specification, statuses, *, expected_revision, expected_state_revision, state_change=False):
        self.initialize()
        with self.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO graphs VALUES(?,0,0,?,NULL)",
                (graph_id, _json(LogicalGraphSpec({}, ()).to_dict())))
            row = connection.execute("SELECT revision,state_revision FROM graphs WHERE graph_id=?", (graph_id,)).fetchone()
            if row is None or tuple(row) != (expected_revision, expected_state_revision):
                raise RevisionConflict("graph or node state changed")
            connection.execute("UPDATE graphs SET spec_json=?,revision=revision+?,state_revision=state_revision+? WHERE graph_id=?",
                (_json(specification.to_dict()), int(not state_change), int(state_change), graph_id))
            for node in specification.nodes:
                connection.execute("INSERT INTO graph_nodes VALUES(?,?,?) ON CONFLICT(graph_id,node_id) DO UPDATE SET status=excluded.status",
                    (graph_id, node.node_id, statuses[node.node_id]))

    def register_context(self, graph_id: str, context_id: str, definition: Mapping[str, Any]) -> None:
        if not context_id:
            raise ValueError("context_id must be nonempty")
        self.initialize()
        serialized = _json(definition)
        with self.transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO graphs VALUES(?,0,0,?,NULL)",
                (graph_id, _json(LogicalGraphSpec({}, ()).to_dict())))
            old = connection.execute("SELECT definition_json FROM contexts WHERE graph_id=? AND context_id=?", (graph_id, context_id)).fetchone()
            if old is None or old[0] != serialized:
                connection.execute("INSERT INTO contexts VALUES(?,?,?) ON CONFLICT(graph_id,context_id) DO UPDATE SET definition_json=excluded.definition_json", (graph_id, context_id, serialized))
                connection.execute("UPDATE graphs SET revision=revision+1 WHERE graph_id=?", (graph_id,))

    def assert_revision(self, graph_id, revision, state_revision):
        """Check compact execution fences without decoding graph definitions."""
        current = (0, 0)
        if self.meta_path.is_file():
            with self.read_transaction() as connection:
                row = connection.execute("SELECT revision,state_revision FROM graphs WHERE graph_id=?", (graph_id,)).fetchone()
                if row is not None:
                    current = tuple(row)
        if current != (revision, state_revision):
            raise RevisionConflict("graph/state changed during execution")

    def commit_update(self, graph_id: str, *, revision: int, state_revision: int,
                      expected_previous: str | None, results: Mapping[str, Mapping[str, str]],
                      through: str, mode: str = "global", request_id: str = "") -> dict[str, Any]:
        """Verify every frozen active context before CAS promotion."""
        state = self.graph_state(graph_id)
        required = {node.node_id for node in state["spec"].nodes if node.node_type == "operator" and state["statuses"][node.node_id] == "active"}
        if mode == "global":
            if not state["contexts"] or set(results) != set(state["contexts"]):
                raise ValueError("global update requires every registered context")
            if any(set(values) != required for values in results.values()):
                raise ValueError("global update requires every active computed node")
        elif mode != "branch":
            raise ValueError("update mode must be global or branch")
        for values in results.values():
            for node_id, identity in values.items():
                if state["statuses"].get(node_id) != "active":
                    raise RevisionConflict("a result belongs to an inactive node")
                record = self.describe(identity)
                if record is None or record["coverage_start"] == "None" or record["through"] != through:
                    raise ValueError("result Domain does not cover the requested final session")
                if record["node_id"] != node_id:
                    raise ValueError("result receipt belongs to a different node")
        body = {"request_id": request_id, "graph_id": graph_id, "revision": revision, "state_revision": state_revision,
                "previous": expected_previous, "results": {k: dict(v) for k, v in results.items()}, "through": through, "mode": mode}
        receipt = {"identity": hash_mapping(body), **body}
        with self.transaction() as connection:
            row = connection.execute("SELECT revision,state_revision,current_update FROM graphs WHERE graph_id=?", (graph_id,)).fetchone()
            if tuple(row[:2]) != (revision, state_revision):
                raise RevisionConflict("graph/state changed before publication")
            if mode == "global" and row[2] not in (expected_previous, receipt["identity"]):
                raise RevisionConflict("current global version changed before publication")
            connection.execute("INSERT OR IGNORE INTO updates VALUES(?,?,?)", (receipt["identity"], graph_id, _json(receipt)))
            if mode == "global":
                connection.execute("UPDATE graphs SET current_update=? WHERE graph_id=?", (receipt["identity"], graph_id))
        return receipt

    def completed_materializations(self, graph_id: str, context_id: str, node_id: str) -> tuple[str, ...]:
        """Read successful results in one admitted context, including branches."""
        with self.read_transaction() as connection:
            current = connection.execute("SELECT identity FROM node_completions WHERE graph_id=? AND context_id=? AND node_id=?",
                (graph_id, context_id, node_id)).fetchone()
            rows = connection.execute("SELECT u.identity,n.value FROM updates u,"
                "json_each(u.receipt_json,'$.results') c,json_each(c.value) n "
                "WHERE u.graph_id=? AND c.key=? AND n.key=?",
                (graph_id, context_id, node_id)).fetchall()
        results = [] if current is None else [current[0]]
        for identity, materialization in rows:
            receipt = self.describe_update(identity)
            if receipt["results"][context_id][node_id] != materialization:
                raise ValueError("completed materialization differs from registered update")
            results.append(materialization)
        return tuple(dict.fromkeys(results))

    def node_completions(self, graph_id: str, context_id: str, *, node_ids: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
        """Read compact successful-node flags in one committed metadata query.

        Each flag refers to immutable numerical authority. It records successful
        node publication independently of complete branch/global promotion, so
        interrupted plans can reuse their completed frontier. Historical updates
        never replace a flag at a later observation/information boundary.
        This reads no numerical files, interval indexes or update receipt bodies.
        """
        if not self.meta_path.is_file() or node_ids is not None and not node_ids:
            return {}
        clause = ""
        arguments = [graph_id, context_id]
        if node_ids is not None:
            clause = " AND c.node_id IN (" + ",".join("?" for _ in node_ids) + ")"
            arguments.extend(node_ids)
        with self.read_transaction() as connection:
            rows = connection.execute("SELECT c.*,m.key_json FROM node_completions c "
                "LEFT JOIN materializations m ON m.identity=c.identity WHERE c.graph_id=? AND c.context_id=?"
                + clause + " ORDER BY c.node_id", arguments).fetchall()
        result = {}
        for row in rows:
            item = dict(row)
            serialized_key = item.pop("key_json")
            if serialized_key is None:
                raise ValueError("completed node numerical authority is missing; explicit Core recovery is required")
            key = MaterializationKey.from_dict(json.loads(serialized_key))
            if key.identity != item["identity"] or key.node_id != item["node_id"]:
                raise ValueError("completed node differs from its immutable numerical authority")
            result[item["node_id"]] = item
        return result

    def complete_node(self, graph_id: str, context_id: str, key: MaterializationKey, *,
                      revision: int, state_revision: int, through: str,
                      definition_digest: str, upstream_digest: str,
                      checked_through: str | None = None) -> None:
        """Advance one compact flag only after fenced durable numerical publication."""
        through = date.fromisoformat(str(through)).isoformat()
        checked_through = date.fromisoformat(str(checked_through or through)).isoformat()
        if any(not isinstance(value, str) or not value for value in (definition_digest, upstream_digest)):
            raise ValueError("completion digests must be nonempty immutable tokens")
        with self.transaction() as connection:
            graph = connection.execute("SELECT revision,state_revision FROM graphs WHERE graph_id=?", (graph_id,)).fetchone()
            if graph is None or tuple(graph) != (revision, state_revision):
                raise RevisionConflict("graph/state changed before node completion")
            node = connection.execute("SELECT status FROM graph_nodes WHERE graph_id=? AND node_id=?", (graph_id, key.node_id)).fetchone()
            if node is None or node[0] != "active":
                raise RevisionConflict("completed node is inactive")
            row = connection.execute("SELECT key_json,manifest_json FROM materializations WHERE identity=?", (key.identity,)).fetchone()
            if row is None:
                raise ValueError("node completion requires a durable numerical receipt")
            manifest = json.loads(row[1])
            if json.loads(row[0]) != key.to_dict() or manifest["through"] != through or manifest["coverage_start"] == "None":
                raise ValueError("node completion does not match its result or requested coverage")
            connection.execute("INSERT INTO node_completions VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(graph_id,context_id,node_id) DO UPDATE SET identity=excluded.identity,"
                "definition_digest=excluded.definition_digest,upstream_digest=excluded.upstream_digest,"
                "through=excluded.through,checked_through=excluded.checked_through "
                "WHERE excluded.through>=node_completions.through AND excluded.checked_through>=node_completions.checked_through",
                (graph_id, context_id, key.node_id, key.identity, definition_digest, upstream_digest, through, checked_through))

    def find_update(self, graph_id: str, request_id: str) -> dict[str, Any] | None:
        """Recover a committed receipt by the caller's stable request token."""
        if not self.meta_path.is_file():
            return None
        with self.read_transaction() as connection:
            rows = connection.execute("SELECT receipt_json FROM updates WHERE graph_id=?", (graph_id,)).fetchall()
        matches = [json.loads(row[0]) for row in rows if json.loads(row[0]).get("request_id") == request_id]
        if len(matches) > 1:
            raise RevisionConflict("request token names more than one update")
        return self.describe_update(matches[0]["identity"]) if matches else None

    @contextmanager
    def reference_publication(self, identity: str):
        """Fence graph/state writers while a caller binds an immutable receipt.

        Core has already committed. This holds only a state lock; the caller's
        transaction succeeds or fails independently and Core evidence survives.
        Historical reads should use verify_update without a live-state fence.
        """
        receipt = self.describe_update(identity)
        with self.transaction() as connection:
            row = connection.execute("SELECT revision,state_revision,current_update FROM graphs WHERE graph_id=?", (receipt["graph_id"],)).fetchone()
            if row is None or tuple(row[:2]) != (receipt["revision"], receipt["state_revision"]):
                raise RevisionConflict("graph/state changed before reference publication")
            if receipt["mode"] == "global" and row[2] != identity:
                raise RevisionConflict("global version changed before reference publication")
            yield receipt

    def verify_update(self, identity: str) -> dict[str, Any]:
        receipt = self.describe_update(identity)
        for results in receipt["results"].values():
            for materialization in results.values():
                self.verify_identity(materialization)
                self.read(materialization)
        return receipt

    def describe_update(self, identity: str) -> dict[str, Any]:
        """Validate a committed completion receipt without reading artifact bytes.

        Publication completeness is distinct from a current full integrity audit.
        Historical receipts do not require the current graph revision to match.
        """
        with self.read_transaction() as connection:
            row = connection.execute("SELECT receipt_json FROM updates WHERE identity=?", (identity,)).fetchone()
        if row is None:
            raise KeyError(f"unknown update receipt: {identity}")
        receipt = json.loads(row[0])
        if hash_mapping({k: v for k, v in receipt.items() if k != "identity"}) != identity:
            raise ValueError("update receipt identity mismatch")
        return receipt
