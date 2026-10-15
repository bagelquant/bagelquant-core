"""Grouped preparation preserves the original per-frame SHA bytes."""
from datetime import date
import hashlib
import json

import numpy as np
import polars as pl
import pytest

from bagelquant_core.hashing import _hash_dataframe_days, hash_dataframe


def original_hash(frame):
    """Frozen original byte algorithm, independent of the grouped helper."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "height": frame.height, "width": frame.width}, sort_keys=True).encode("utf-8"))
    if frame.width:
        digest.update(frame.hash_rows(seed=0, seed_1=0, seed_2=0, seed_3=0).to_numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("kind", ("numeric", "category", "membership", "empty"))
def test_day_hashes_match_original_partition_hashes_with_one_row_hash(kind, monkeypatch):
    # Dates and keys are deliberately interleaved and reordered. Preserve NaN
    # payload bits, signed zero, nulls, Unicode, dates and a chunked input.
    days = [date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2),
        date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 3)]
    floats = np.array([0x8000000000000000, 0x7ff8000000000001, 0x7ff0000000000000,
        0xfff0000000000000, 0x0000000000000000, 0x7ff8000000000002], dtype=np.uint64).view(np.float64)
    frame = pl.DataFrame({"time": days, "asset_id": ["雪", "B", "A", "", "雪", "A"],
        "value": pl.Series(floats),
        "available": [None, days[3], days[1], None, days[4], days[2]]})
    frame = frame.with_columns(pl.when(pl.int_range(pl.len()) == 3).then(None).otherwise(pl.col("value")).alias("value"))
    if kind == "category":
        frame = frame.with_columns(pl.Series("value", ["one", None, "雪", "two", "one", ""]).cast(pl.Categorical))
    elif kind == "membership":
        frame = frame.select("time", "asset_id")
    elif kind == "empty":
        frame = frame.head(0)
    frame = pl.concat([frame.head(2), frame.slice(2)], rechunk=False)
    assert hash_dataframe(frame) == original_hash(frame)
    assert hash_dataframe(pl.DataFrame()) == original_hash(pl.DataFrame())
    expected = {}
    for part in frame.partition_by("time", maintain_order=True):
        expected[str(part["time"][0])] = original_hash(part)
        assert hash_dataframe(part) == expected[str(part["time"][0])]
    original = pl.DataFrame.hash_rows
    calls = []
    def hash_rows(value, *args, **kwargs):
        calls.append(value.height)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(pl.DataFrame, "hash_rows", hash_rows)
    assert _hash_dataframe_days(frame) == expected
    assert calls == ([] if kind == "empty" else [frame.height])
