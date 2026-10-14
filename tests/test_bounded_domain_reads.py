"""Persisted observation windows bound Domain work without changing values."""
from datetime import date, timedelta
import struct

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from bagelquant_core import CoreStore, Domain, MaterializationKey, Node, NodeMaterialization


def published(tmp_path, *, dynamic=True, value_type="numeric"):
    days = [date(2024, 1, 2) + timedelta(days=2 * n) for n in range(96)]
    assets = [f"A{n:02}" for n in range(24)]
    grid = pl.DataFrame({"time": days}).join(pl.DataFrame({"asset_id": assets}), how="cross")
    if dynamic:
        grid = grid.with_row_index("position").filter(
            (pl.col("position") % 5 != 0)
            & ((pl.col("asset_id") < "A12") | (pl.col("time") >= days[40]))
        ).drop("position")
        universe = grid.with_columns(pl.lit(True).alias("active"))
    else:
        universe = assets
    domain = Domain(calendar=days, universe=universe)
    frame = grid.with_row_index("position").with_columns(
        (pl.when(pl.col("position") % 7 == 0).then(None)
         .otherwise(pl.when(pl.col("position") % 2 == 0).then(pl.lit("one")).otherwise(pl.lit("two")))
         if value_type == "category" else
         pl.when(pl.col("position") % 7 == 0).then(None)
         .when(pl.col("position") % 11 == 0).then(float("nan"))
         .when(pl.col("position") % 3 == 0).then(-0.0).otherwise(0.0)).alias("value"),
        pl.col("time").alias("available"), pl.lit("retained").alias("marker"),
    ).drop("position")
    key = MaterializationKey("bounded-fixture", "operator.v1", ("source",), domain.signature)
    panel = Node.from_domain(frame, domain, value_type=value_type, identity=key.identity,
        trace_columns=("available", "marker"), trace_identity="original-traces")
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    store.publish(NodeMaterialization(key, panel))
    return store, key.identity, days


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("value_type", ["numeric", "prediction", "weights", "category"])
@pytest.mark.parametrize("bounds", ["one", "32", "lower", "upper", "empty", "gap"])
def test_bounded_values_all_channels_equal_full_reference(tmp_path, monkeypatch, dynamic, value_type, bounds):
    store, identity, days = published(tmp_path, dynamic=dynamic, value_type=value_type)
    full = store.read_values(identity, include_traces=True)
    lower, upper = {
        "one": (days[47], days[47]), "32": (days[16], days[47]),
        "lower": (days[70], None), "upper": (None, days[12]),
        "empty": (days[-1] + timedelta(days=1), None),
        "gap": (days[20] + timedelta(days=1), days[21] - timedelta(days=1)),
    }[bounds]
    expected = full
    if lower is not None:
        expected = expected.filter(pl.col("time") >= lower)
    if upper is not None:
        expected = expected.filter(pl.col("time") <= upper)
    manifest = store.describe(identity)
    prior = store.inventory()
    calls = []
    original = Domain.__init__

    def domain(owner, *, calendar, universe, identity=None):
        calls.append((calendar, universe))
        if lower is not None:
            assert calendar.min() >= lower
        if upper is not None:
            assert calendar.max() <= upper
        if isinstance(universe, pl.DataFrame):
            if lower is not None:
                assert universe.is_empty() or universe["time"].min() >= lower
            if upper is not None:
                assert universe.is_empty() or universe["time"].max() <= upper
        return original(owner, calendar=calendar, universe=universe, identity=identity)

    monkeypatch.setattr(Domain, "__init__", domain)
    monkeypatch.setattr(CoreStore, "_decode", lambda *_: pytest.fail("bounded view decoded full Domain"))
    with store.read_context() as reader:
        for include_traces in (True, False):
            result = reader.read_values(identity, start=lower, end=upper, include_traces=include_traces)
            assert_frame_equal(result, expected if include_traces else expected.select("time", "asset_id", "value"))
            if value_type != "category":
                assert [None if v is None else struct.pack("!d", v) for v in result["value"]] == [
                    None if v is None else struct.pack("!d", v) for v in expected["value"]]
    if bounds in {"empty", "gap"}:
        assert calls == []
    elif dynamic:
        assert max(universe.height for _, universe in calls) <= expected.height
    assert store.describe(identity) == manifest
    assert store.inventory() == prior


def test_default_read_keeps_complete_domain_validation(tmp_path, monkeypatch):
    store, identity, days = published(tmp_path)
    decoded = []
    original = CoreStore._decode

    def decode(owner, reference):
        result = original(owner, reference)
        if isinstance(result, pl.DataFrame):
            decoded.append(result)
        return result

    monkeypatch.setattr(CoreStore, "_decode", decode)
    monkeypatch.setattr(CoreStore, "_read_domain_rows", lambda *_args, **_kwargs: pytest.fail("default read pruned Domain"))
    result = store.read_values(identity, include_traces=True)
    assert len(decoded) == 2
    assert decoded[0].height == len(days)
    assert decoded[1].height == result.height


@pytest.mark.parametrize("channel,corruption,message", [
    ("calendar", "type", "Domain artifact schema"),
    ("membership", "type", "Domain artifact schema"),
    ("membership", "extra", "Domain artifact schema"),
    ("calendar", "null", "keys must be valid"),
    ("membership", "null", "keys must be valid"),
    ("membership", "null_asset", "keys must be valid"),
    ("calendar", "duplicate", "times must be unique"),
    ("membership", "duplicate", "unique by"),
])
def test_scoped_domain_retains_typed_and_invalid_key_checks(tmp_path, channel, corruption, message):
    store, identity, days = published(tmp_path)
    manifest = store.manifest(identity)
    assert manifest is not None
    path = store._file(manifest["domain"][channel]["$file"])
    frame = pl.read_parquet(path)
    if corruption == "type":
        frame = frame.with_columns(pl.col("time").cast(pl.String))
    elif corruption == "extra":
        frame = frame.with_columns(pl.lit(1).alias("unexpected"))
    elif corruption in {"null", "null_asset"}:
        # Invalid observation has no safe scope even outside the requested date.
        column = "asset_id" if corruption == "null_asset" else "time"
        selected = frame.filter(pl.col("time") == (days[47] if corruption == "null_asset" else days[0])).head(1)
        frame = pl.concat([frame, selected.with_columns(pl.lit(None).cast(frame.schema[column]).alias(column))])
    else:
        frame = pl.concat([frame, frame.filter(pl.col("time") == days[47]).head(1)])
    frame.write_parquet(path)
    with pytest.raises(ValueError, match=message):
        store.read_values(identity, start=days[47], end=days[47])
    with pytest.raises(ValueError, match="artifact"):
        store.verify_identity(identity)


def test_unconsumed_duplicate_is_full_audit_concern(tmp_path):
    store, identity, days = published(tmp_path)
    expected = store.read_values(identity, start=days[-1], end=days[-1], include_traces=True)
    manifest = store.manifest(identity)
    assert manifest is not None
    path = store._file(manifest["domain"]["membership"]["$file"])
    frame = pl.read_parquet(path)
    pl.concat([frame, frame.filter(pl.col("time") == days[0]).head(1)]).write_parquet(path)
    assert_frame_equal(store.read_values(identity, start=days[-1], end=days[-1], include_traces=True), expected)
    with pytest.raises(ValueError, match="unique by"):
        store.read_values(identity)
    with pytest.raises(ValueError, match="artifact"):
        store.verify_identity(identity)


@pytest.mark.parametrize("corruption", ["value_type", "missing_trace", "empty_calendar", "gap_duplicate"])
def test_empty_window_still_validates_consumed_schema_and_domain(tmp_path, corruption):
    store, identity, days = published(tmp_path)
    manifest = store.manifest(identity)
    assert manifest is not None
    lower = days[-1] + timedelta(days=1)
    if corruption == "empty_calendar":
        path = store._file(manifest["domain"]["calendar"]["$file"])
        pl.read_parquet(path).clear().write_parquet(path)
        message = "calendar must contain"
    elif corruption == "gap_duplicate":
        lower = days[20] + timedelta(days=1)
        path = store._file(manifest["domain"]["membership"]["$file"])
        frame = pl.read_parquet(path)
        duplicate = frame.head(1).with_columns(pl.lit(lower).alias("time"))
        pl.concat([frame, duplicate, duplicate]).write_parquet(path)
        message = "unique by"
    else:
        path = store._file(manifest["partitions"][0]["$file"])
        frame = pl.read_parquet(path)
        frame = frame.with_columns(pl.col("value").cast(pl.String)) if corruption == "value_type" else frame.drop("available")
        frame.write_parquet(path)
        message = "value column must be numeric" if corruption == "value_type" else "missing required columns"
    with pytest.raises((TypeError, ValueError), match=message):
        store.read_values(identity, start=lower, end=lower, include_traces=True)


def test_reversed_bounds_and_unknown_identity(tmp_path):
    store, identity, days = published(tmp_path)
    with pytest.raises(ValueError, match="end precedes start"):
        store.read_values(identity, start=days[-1], end=days[0])
    with pytest.raises(KeyError, match="unknown Core materialization"):
        store.read_values("missing", start=days[0], end=days[1])


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("corruption", ["duplicate", "null", "type"])
def test_static_asset_keys_are_validated_even_for_empty_windows(tmp_path, empty, corruption):
    store, identity, days = published(tmp_path, dynamic=False)
    manifest = store.manifest(identity)
    assert manifest is not None
    path = store._file(manifest["domain"]["membership"]["$file"])
    frame = pl.read_parquet(path)
    if corruption == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
        message = "asset_id values must be unique"
    elif corruption == "null":
        frame = pl.concat([frame, pl.DataFrame({"asset_id": [None]}, schema={"asset_id": pl.String})])
        message = "keys must be valid"
    else:
        frame = frame.with_columns(pl.int_range(pl.len()).alias("asset_id"))
        message = "Domain artifact schema"
    frame.write_parquet(path)
    selected = days[-1] + timedelta(days=1) if empty else days[47]
    with pytest.raises(ValueError, match=message):
        store.read_values(identity, start=selected, end=selected)
