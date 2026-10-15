"""Stable hashing utilities for graph configs and panel payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_value(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return repr(value)


def hash_mapping(payload: Mapping[str, Any]) -> str:
    normalized = _normalize_value(payload)
    data = json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_dataframe(frame: pl.DataFrame) -> str:
    row_hashes = frame.hash_rows(seed=0, seed_1=0, seed_2=0, seed_3=0) if frame.width else None
    return _dataframe_digest(frame.schema, frame.height, frame.width, row_hashes)


def _dataframe_digest(schema, height, width, row_hashes) -> str:
    digest = hashlib.sha256()
    header = {
        "schema": {name: str(dtype) for name, dtype in schema.items()},
        "height": height,
        "width": width,
    }
    digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    if row_hashes is not None:
        digest.update(row_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def _hash_dataframe_days(frame: pl.DataFrame) -> dict[str, str]:
    """Original per-day digests with one physical row-hash evaluation.

    Row hashing is independent of other rows. Partition only the resulting
    hashes, preserving each group's original order and original schema header.
    This also supports interleaved dates without assuming contiguous groups.
    """
    if not frame.height:
        return {}
    hashes = pl.DataFrame({"time": frame.get_column("time"),
        "row_hash": frame.hash_rows(seed=0, seed_1=0, seed_2=0, seed_3=0)})
    return {str(day["time"][0]): _dataframe_digest(frame.schema, day.height, frame.width, day["row_hash"])
        for day in hashes.partition_by("time", maintain_order=True)}
