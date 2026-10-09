"""Caller-located, immutable graph and numerical artifact storage."""
from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from typing import Any, Iterator, Mapping
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
from .node import Node

SCHEMA_VERSION = 1


class RevisionConflict(RuntimeError):
    """The graph changed after an update was planned."""


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
        return manifest

    @staticmethod
    def _verify_metadata(manifest: Mapping[str, Any], identity: str) -> None:
        key = MaterializationKey.from_dict(manifest["key"])
        if (manifest.get("content_identity") != hash_mapping(_manifest_content(manifest))
                or manifest.get("identity") != identity or key.identity != identity
                or manifest.get("node_id") != key.node_id):
            raise ValueError("corrupt Core manifest metadata")

    def verify(self, manifest: Mapping[str, Any]) -> None:
        self._verify_metadata(manifest, manifest["identity"])
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
        frame = pl.scan_parquet([self._file(part["$file"]) for part in manifest["partitions"]])
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

    def publish(self, value: NodeMaterialization, *, execution_policy=None) -> None:
        self.initialize()
        if value.panel.domain.signature != value.key.domain_identity:
            raise ValueError("materialization Domain does not match its key")
        generation = uuid.uuid4().hex
        relative = f"values/{value.key.identity}/{generation}"
        temporary = self.artifact_path / ".pending" / generation
        final = self._file(relative)
        temporary.mkdir(parents=True)
        counter = 0

        def encode(item):
            nonlocal counter
            if isinstance(item, pl.LazyFrame):
                item = item.collect()
            if isinstance(item, pl.DataFrame):
                filename = f"{counter:06d}.parquet"
                counter += 1
                path = temporary / filename
                item.write_parquet(path)
                return {"$file": f"{relative}/{filename}", "sha256": _checksum(path)}
            if isinstance(item, datetime):
                return {"$datetime": item.isoformat()}
            if isinstance(item, date):
                return {"$date": item.isoformat()}
            if isinstance(item, Mapping):
                return {str(key): encode(child) for key, child in item.items()}
            if isinstance(item, (tuple, list)):
                return [encode(child) for child in item]
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
            months = calendar.dt.truncate("1mo").unique().sort()
            for month in months:
                dates = calendar.filter(calendar.dt.truncate("1mo") == month)
                frame = value.panel._validate_collected(plan.filter(pl.col("time").is_between(dates.min(), dates.max())).collect())
                partitions.append({**encode(frame), "start": str(dates.min()), "end": str(dates.max()), "rows": frame.height})
                for day_frame in frame.partition_by("time", maintain_order=True):
                    days[str(day_frame["time"][0])] = hash_dataframe(day_frame)
            if not partitions:
                partitions.append(encode(plan.collect()))
            manifest = {"schema": SCHEMA_VERSION, "key": value.key.to_dict(), "value_type": value.panel.value_type.value,
                "trace_columns": list(value.panel.trace_columns), "trace_identity": value.panel.trace_identity,
                "domain": {"dynamic": domain.is_dynamic, "calendar": encode(pl.DataFrame({"time": domain.times})),
                    "membership": encode(domain.grid_lazy().collect() if domain.is_dynamic else pl.DataFrame({"asset_id": domain.asset_ids}))},
                "identity": value.key.identity, "node_id": value.key.node_id,
                "through": str(calendar.max()), "coverage_start": str(calendar.min()),
                "days": days, "logical_hash": hash_mapping(days),
                "partitions": partitions, "artifacts": encode(value.artifacts),
                "checkpoint": encode(value.checkpoint), "training_audits": encode(value.training_audits),
                "execution_policy": execution_policy}
            manifest["content_identity"] = hash_mapping(_manifest_content(manifest))
            from . import computation_index
            indexed = computation_index.describe(domain, days=days, value_type=manifest["value_type"],
                                                  trace_columns=manifest["trace_columns"])
            serialized = _json(manifest)
            final.parent.mkdir(parents=True, exist_ok=True)
            with self.transaction() as connection:
                connection.execute(computation_index.DDL)
                old = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (value.key.identity,)).fetchone()
                if old is not None:
                    existing = json.loads(old[0])
                    if existing.get("content_identity") != manifest["content_identity"]:
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

    def inventory(self) -> tuple[dict[str, Any], ...]:
        if not self.meta_path.is_file():
            return ()
        with self.read_transaction() as connection:
            rows = connection.execute("SELECT identity,node_id,key_json FROM materializations ORDER BY identity").fetchall()
        return tuple({"identity": row[0], "node_id": row[1], "key": json.loads(row[2])} for row in rows)

    def describe(self, identity: str) -> dict[str, Any] | None:
        """Public metadata receipt containing no artifact locations."""
        manifest = self.manifest(identity)
        if manifest is None:
            return None
        from .computation_index import read
        with self.read_transaction() as connection:
            indexed = read(connection, manifest)
        return {**{key: manifest[key] for key in ("identity", "node_id", "key", "value_type", "through",
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
        selected = [part for part in manifest["partitions"]
            if (lower is None or "end" not in part or date.fromisoformat(part["end"]) >= lower)
            and (upper is None or "start" not in part or date.fromisoformat(part["start"]) <= upper)]
        saved = manifest["domain"]
        calendar = self._decode(saved["calendar"])["time"]
        membership = self._decode(saved["membership"])
        universe = membership.with_columns(pl.lit(True).alias("active")) if saved["dynamic"] else membership["asset_id"]
        domain = Domain(calendar=calendar, universe=universe, identity=manifest["key"]["domain_identity"])
        if selected:
            frame = pl.scan_parquet([self._file(part["$file"]) for part in selected])
        else:
            # An empty window still needs the original typed schema.
            frame = pl.DataFrame(schema=pl.read_parquet_schema(self._file(manifest["partitions"][0]["$file"]))).lazy()
        if start is not None:
            frame = frame.filter(pl.col("time") >= start)
        if end is not None:
            frame = frame.filter(pl.col("time") <= end)
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
        state = self.graph_state(graph_id)
        if (state["revision"], state["state_revision"]) != (revision, state_revision):
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
