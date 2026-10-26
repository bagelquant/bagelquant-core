"""Optional content-bound resource metadata, separate from numerical proofs."""
from __future__ import annotations

import json

from .hashing import hash_mapping

DDL = """CREATE TABLE IF NOT EXISTS value_read_bounds(
 identity TEXT PRIMARY KEY REFERENCES materializations(identity) ON DELETE CASCADE,content_identity TEXT NOT NULL,
 payload_json TEXT NOT NULL,payload_identity TEXT NOT NULL)"""


def describe(domain):
    calendar = [str(day) for day in domain.times]
    counts = {str(day): count for day, count in domain.grid_lazy().group_by("time").len().collect().iter_rows()} if domain.is_dynamic else {
        day: len(domain.asset_ids) for day in calendar}
    return {"calendar": calendar, "reader_rows": counts,
        "static_assets": None if domain.is_dynamic else len(domain.asset_ids)}


def save(connection, manifest, domain):
    payload = describe(domain)
    connection.execute(DDL)
    connection.execute("INSERT OR REPLACE INTO value_read_bounds VALUES(?,?,?,?)",
        (manifest["identity"], manifest["content_identity"], json.dumps(payload, sort_keys=True), hash_mapping(payload)))


def read(connection, manifest):
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='value_read_bounds'").fetchone() is None:
        return None
    row = connection.execute("SELECT content_identity,payload_json,payload_identity FROM value_read_bounds WHERE identity=?",
        (manifest["identity"],)).fetchone()
    if row is None or row[0] != manifest["content_identity"]:
        return None
    payload = json.loads(row[1])
    if hash_mapping(payload) != row[2]:
        raise ValueError("Corrupt Core value read bounds")
    return payload
