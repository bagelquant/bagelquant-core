from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object as _hash_pandas_object

hash_pandas_object = cast(Any, _hash_pandas_object)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return repr(value)


def hash_mapping(payload: Mapping[str, Any]) -> str:
    normalized = _normalize_value(payload)
    data = json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_dataframe(frame: pd.DataFrame) -> str:
    row_hash = hash_pandas_object(frame, index=True).to_numpy()
    col_hash = hash_pandas_object(pd.Index(frame.columns)).to_numpy()
    digest = hashlib.sha256()
    digest.update(row_hash.tobytes())
    digest.update(col_hash.tobytes())
    return digest.hexdigest()
