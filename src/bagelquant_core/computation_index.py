"""Derived, versioned Domain and interval proofs, without changing manifests."""
from __future__ import annotations

import json

from .hashing import hash_dataframe, hash_mapping

VERSION = "core.interval.v1"
DDL = """CREATE TABLE IF NOT EXISTS computation_index(
 identity TEXT PRIMARY KEY,content_identity TEXT NOT NULL,version TEXT NOT NULL,
 payload_json TEXT NOT NULL,payload_identity TEXT NOT NULL)"""


def describe(domain, *, days, value_type, trace_columns) -> dict:
    calendar = [str(day) for day in domain.times]
    if domain.is_dynamic:
        grid = domain.grid_lazy().sort("time", "asset_id").collect()
        members = {str(frame["time"][0]): hash_dataframe(frame) for frame in grid.partition_by("time", maintain_order=True)}
        empty = hash_dataframe(grid.head(0))
    else:
        # A static Domain needs no calendar x asset allocation for its proof.
        assets = hash_mapping({"assets": domain.asset_ids.to_list()})
        members = {day: hash_mapping({"day": day, "assets": assets}) for day in calendar}
        empty = hash_mapping({"assets": []})
    return {"calendar": calendar, "domain_identity": domain.signature,
            "membership": {day: members.get(day, empty) for day in calendar},
            "days": dict(days), "value_type": str(value_type),
            "trace_columns": list(trace_columns)}


def save(db, identity: str, content_identity: str, payload: dict) -> None:
    db.execute("INSERT OR REPLACE INTO computation_index VALUES(?,?,?,?,?)",
               (identity, content_identity, VERSION,
                json.dumps(payload, sort_keys=True), hash_mapping(payload)))


def read(db, manifest) -> dict | None:
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='computation_index'").fetchone() is None:
        return None
    row = db.execute("SELECT content_identity,version,payload_json,payload_identity "
                     "FROM computation_index WHERE identity=?", (manifest["identity"],)).fetchone()
    if row is None or row[0] != manifest["content_identity"] or row[1] != VERSION:
        return None
    value = json.loads(row[2])
    if hash_mapping(value) != row[3]:
        raise ValueError("corrupt Core computation index")
    return value


def interval(payload, start, end) -> str:
    calendar = [day for day in payload["calendar"] if str(start) <= day <= str(end)]
    return hash_mapping({"version": VERSION, "calendar": calendar,
        "membership": [(day, payload["membership"][day]) for day in calendar],
        "values": [(day, payload["days"].get(day)) for day in calendar],
        "value_type": payload["value_type"], "trace_columns": payload["trace_columns"]})
