"""Grouped preparation preserves the original per-frame SHA bytes."""
from datetime import date
import hashlib
import json

import numpy as np
import polars as pl
import pytest

from bagelquant_core.hashing import DATAFRAME_HASH_V1, DATAFRAME_HASH_V2, _hash_dataframe_days, hash_dataframe


def original_hash(frame):
    """Frozen original byte algorithm, independent of the grouped helper."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "height": frame.height, "width": frame.width}, sort_keys=True).encode("utf-8"))
    if frame.width:
        digest.update(frame.hash_rows(seed=0, seed_1=0, seed_2=0, seed_3=0).to_numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("kind", ("numeric", "category", "membership", "empty"))
@pytest.mark.parametrize("policy", (DATAFRAME_HASH_V1, DATAFRAME_HASH_V2))
def test_day_hashes_match_partition_hashes_with_one_row_hash(kind, policy, monkeypatch):
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
    reference = original_hash if policy == DATAFRAME_HASH_V1 else hash_dataframe
    assert hash_dataframe(frame, policy=policy) == reference(frame)
    assert hash_dataframe(pl.DataFrame(), policy=policy) == reference(pl.DataFrame())
    expected = {}
    for part in frame.partition_by("time", maintain_order=True):
        expected[str(part["time"][0])] = reference(part)
        assert hash_dataframe(part, policy=policy) == expected[str(part["time"][0])]
    original = pl.DataFrame.hash_rows
    calls = []
    def hash_rows(value, *args, **kwargs):
        calls.append(value.height)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(pl.DataFrame, "hash_rows", hash_rows)
    assert _hash_dataframe_days(frame, policy=policy) == expected
    assert calls == ([] if kind == "empty" else [frame.height])


def test_corrected_hash_binds_named_column_positions():
    first = pl.DataFrame({"gross_return": [.1], "net_return": [.2]})
    swapped = pl.DataFrame({"net_return": [.1], "gross_return": [.2]})
    assert original_hash(first) == original_hash(swapped)
    assert hash_dataframe(first) != hash_dataframe(swapped)
    assert hash_dataframe(first) != original_hash(first)
    with pytest.raises(ValueError, match="policy"):
        hash_dataframe(first, policy="unknown")


def test_canonical_input_and_domain_original_identities_are_preserved():
    from bagelquant_core import Domain, Node
    from bagelquant_core.hashing import hash_mapping
    days = [date(2024, 1, 1), date(2024, 1, 2)]
    domain = Domain(calendar=days, universe=["A"])
    frame = pl.DataFrame({"time": days, "asset_id": ["A"] * 2, "value": [1., 2.]})
    node = Node.from_domain(frame, domain)
    assert node.identity == "content:" + original_hash(frame)
    assert domain.signature == hash_mapping({
        "calendar": original_hash(pl.DataFrame({"time": days})),
        "assets": original_hash(pl.DataFrame({"asset_id": ["A"]})),
        "dynamic": False, "active": None})
    first = frame.with_columns(pl.Series("a", days), pl.Series("b", list(reversed(days))))
    second = frame.with_columns(pl.Series("b", days), pl.Series("a", list(reversed(days))))
    left = Node.from_domain(first, domain, trace_columns=("a", "b"))
    right = Node.from_domain(second, domain, trace_columns=("b", "a"))
    assert left.identity.startswith("content:v2:")
    assert left.identity != right.identity


def test_traced_source_policy_change_reuses_proven_original_blocks(tmp_path, monkeypatch):
    from bagelquant_core import ExecutionRuntime, Node
    from bagelquant_core.operator import negate
    from test_global_graph import finish, global_graph, source

    current = source(35)
    frame = current.collect(include_traces=True)
    original = Node.from_domain(frame, current.domain, source_key="x", name="x",
        trace_columns=current.trace_columns, identity="content:" + original_hash(frame))
    graph, store, root = global_graph(tmp_path, negate(original))
    # The source owner supplies its already-built original interval metadata;
    # a different enclosing source address alone is not historical replay.
    def proof(_context, _source, start, end):
        return current.interval_identity(start, end)
    receipt = finish(graph.plan_update({"main": {"x": original}},
        through=current.domain.times.max(), input_proof=proof))
    before = {part["$file"]: store._file(part["$file"]).read_bytes()
        for item in store.inventory() for part in store.manifest(item["identity"])["partitions"]}
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("source hash policy replayed history"))
        patch.setattr(pl, "read_parquet", lambda *_a, **_k: pytest.fail("source hash policy decoded historical values"))
        plan = graph.plan_update({"main": {"x": current}},
            through=current.domain.times.max(), input_proof=proof)
        updated = finish(plan)
    assert plan.resource_usage["main"][root]["computed_blocks"] == 0
    assert plan.resource_usage["main"][root]["kernel_sessions"] == 0
    assert plan.resource_usage["main"][root]["reused_blocks"] == 2
    assert all(store._file(path).read_bytes() == raw for path, raw in before.items())
    assert store.read_values(receipt["results"]["main"][root]).equals(
        store.read_values(updated["results"]["main"][root]))
