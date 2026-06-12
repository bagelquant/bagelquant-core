from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

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
    ordered = frame.sort(frame.columns) if frame.columns else frame
    payload = {
        "schema": {name: str(dtype) for name, dtype in ordered.schema.items()},
        "rows": ordered.to_dicts(),
    }
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
