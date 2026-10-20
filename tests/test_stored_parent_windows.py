"""Finite consumers acquire published parents through bounded typed reads."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta
import gc
import sqlite3
from threading import Barrier, Lock
from typing import Callable, cast
import weakref

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from bagelquant_core import (
    CoreStore, Domain, ExecutionMode, InputDensity, Node, OPERATOR_REGISTRY, OperationContract,
    RevisionConflict, TraceRule,
)
from bagelquant_core.operator import (
    negate as _negate, operator, pct_change as _pct_change, rolling_mean as _rolling_mean,
)
from bagelquant_core.resources import ResourceLimits, active_resource_limits, resource_limits
from bagelquant_core.incremental import _parent_window
from bagelquant_core.preparation import ExecutionPreparation, preparation_scope
from bagelquant_core.update import UpdatePlan
from test_global_graph import finish, global_graph

# Operators also dispatch on Node; their kernel annotations describe DataFrames.
negate = cast(Callable[[Node], Node], _negate)
pct_change = cast(Callable[..., Node], _pct_change)
rolling_mean = cast(Callable[..., Node], _rolling_mean)


@pytest.fixture(autouse=True)
def isolate_operators(monkeypatch):
    monkeypatch.setattr(OPERATOR_REGISTRY, "_items", dict(OPERATOR_REGISTRY._items))


def source(count=101, *, case="reentry", revision=None, kind="numeric", datetime_trace=False):
    days = [date(2024, 1, 1) + timedelta(days=i + i // 5 * 2) for i in range(count)]
    rows = []
    for i, day in enumerate(days):
        assets = [] if case == "empty" and i >= 64 else ["A", "B"]
        if i < 7 or case == "reentry" and i >= 84:
            assets.append("REENTRY")
        if i >= 80 and case != "empty":
            assets.append("NEW")
        for asset in assets:
            number = (0.0, -0.0, None, float("nan"), 2.5, 7.0, float("inf"), -float("inf"))[i % 8]
            if revision == i:
                number = 19.0
            value = (None if i % 9 == 0 else asset.lower()) if kind == "category" else number
            observed = datetime.combine(day, datetime.min.time()) + timedelta(hours=17) if datetime_trace else day
            rows.append((day, asset, value, None if i % 13 == 0 else day + timedelta(days=1), observed))
    frame = pl.DataFrame(rows, schema={"time": pl.Date, "asset_id": pl.String,
        "value": pl.String if kind == "category" else pl.Float64,
        "available_date": pl.Date, "observed_at": pl.Datetime("us") if datetime_trace else pl.Date}, orient="row")
    domain = Domain(calendar=days, universe=(frame.select("time", "asset_id")
        .with_columns(pl.lit(True).alias("active"))) if case != "static" else ["A", "B", "REENTRY", "NEW"])
    return Node.from_domain(frame, domain, source_key="x", name="x", value_type=kind,
        trace_columns=("available_date", "observed_at"), trace_identity="fixture.traces.v1")


def legacy_windows(monkeypatch):
    original = UpdatePlan._window_reader
    def legacy(plan, context, guard, limits, **_kwargs):
        return original(plan, context, guard, limits)
    monkeypatch.setattr(UpdatePlan, "_window_reader", legacy)


def values(store, receipt, root):
    return store.read_values(receipt["results"]["main"][root], include_traces=True)


def prepared(tmp_path, *, count=70, datetime_trace=False):
    x = source(count, datetime_trace=datetime_trace)
    graph, store, root = global_graph(tmp_path, rolling_mean(negate(x), window=4))
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max())
    parent = plan.ready("main")[0]
    plan.execute("main", parent)
    expected = plan._values["main"][parent]
    return graph, store, root, plan, parent, expected


@pytest.mark.parametrize("case", ["static", "reentry", "empty"])
@pytest.mark.parametrize("window_loader", [False, True])
def test_multiblock_values_and_nonempty_traces_equal_legacy(tmp_path, monkeypatch, case, window_loader):
    x = source(case=case)
    def options():
        if not window_loader:
            return {}
        def window(_context, _key, start, end):
            return Node.from_domain(x.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
                x.domain.window(start, end), identity=x.identity, value_type=x.value_type,
                trace_columns=x.trace_columns, trace_identity=x.trace_identity)
        return {"input_window_loader": window}
    reference, old, old_root = global_graph(tmp_path / "old", rolling_mean(negate(x), window=4))
    with monkeypatch.context() as patch:
        legacy_windows(patch)
        before = finish(reference.plan_update({"main": {"x": x}}, through=x.domain.times.max(), **options()))
    graph, store, root = global_graph(tmp_path / "new", rolling_mean(negate(x), window=4))
    calls = []
    read = CoreStore.read_values
    def bounded(reader, identity, *, start=None, end=None, include_traces=False):
        if reader is store:
            calls.append((identity, start, end, include_traces, active_resource_limits()))
        return read(reader, identity, start=start, end=end, include_traces=include_traces)
    monkeypatch.setattr(CoreStore, "read_values", bounded)
    limits = ResourceLimits(total_threads=1, parallel_nodes=1, memory_target_mib=128, cache_mib=16)
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(), **options())
    after = finish(plan, limits=limits)
    # Shared preparation reads canonical slabs, including clipped overlap slabs.
    # It must not repeatedly decode an identical window or read an entire prefix.
    windows = [(call[1], call[2]) for call in calls]
    assert 4 <= len(calls) <= 7
    assert len(set(windows)) == len(windows)
    assert all(len(x.domain.times.filter(x.domain.times.is_between(first, last))) <= 32
        for first, last in windows)
    numerical_limits = replace(limits, cache_mib=limits.cache_mib - limits.cache_mib // 4)
    assert all(c[1] is not None and c[2] is not None and c[3] is True and c[4] == numerical_limits for c in calls)
    assert calls[-1][2] == x.domain.times.max()
    assert {c[0] for c in calls} == set(plan.results["main"].values()) - {after["results"]["main"][root]}
    actual, expected = values(store, after, root), values(old, before, old_root)
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual.schema["available_date"] == pl.Date
    assert actual.schema["observed_at"] == pl.Date
    assert after["results"] == before["results"]
    manifest = store.manifest(after["results"]["main"][root])
    old_manifest = old.manifest(before["results"]["main"][old_root])
    assert manifest is not None and old_manifest is not None
    assert manifest["trace_identity"] == old_manifest["trace_identity"]


@pytest.mark.parametrize("kind", ["category", "prediction", "weights"])
def test_typed_persisted_parents_keep_value_and_trace_channels(tmp_path, monkeypatch, kind):
    @operator(output_type=kind, input_types=(kind,), contract=OperationContract(
        execution=ExecutionMode.EAGER_BARRIER, density=InputDensity.SPARSE_OK,
        trace_rule=TraceRule.PASSTHROUGH, history=0))
    def typed(frame: pl.DataFrame, *, suffix: str):
        return (frame.with_columns((pl.col("value") + suffix).alias("value")) if kind == "category"
            else frame.with_columns((pl.col("value") + len(suffix)).alias("value")))
    x = source(65, kind=kind, datetime_trace=True)
    typed_node = cast(Callable[..., Node], typed)
    expression = typed_node(typed_node(x, suffix="a"), suffix="b")
    reference, old, old_root = global_graph(tmp_path / "old", expression)
    with monkeypatch.context() as patch:
        legacy_windows(patch)
        before = finish(reference.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    graph, store, root = global_graph(tmp_path / "new", expression)
    after = finish(graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    assert after["results"] == before["results"]
    assert_frame_equal(values(store, after, root), values(old, before, old_root), check_exact=True)
    assert values(store, after, root).schema["observed_at"] == pl.Datetime("us")
    assert store.read(after["results"]["main"][root]).panel.value_type == kind


def test_finite_without_window_loader_loads_lazy_source_once(tmp_path):
    x = source(70)
    placeholder = Node.from_domain(x.lazy(include_traces=True).filter(pl.lit(False)), x.domain,
        source_key="x", identity=x.identity, trace_columns=x.trace_columns, trace_identity=x.trace_identity)
    graph, store, root = global_graph(tmp_path, rolling_mean(negate(placeholder), window=4))
    loaded = []
    def load(context, key):
        loaded.append((context, key))
        return x
    plan = graph.plan_update({"main": {"x": placeholder}}, through=x.domain.times.max(), input_loader=load)
    result = finish(plan)
    assert loaded == [("main", "x")]
    reference, old, old_root = global_graph(tmp_path / "full", rolling_mean(negate(x), window=4))
    full = finish(reference.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    assert_frame_equal(values(store, result, root), values(old, full, old_root), check_exact=True)


@pytest.mark.parametrize("change", ["no_key", "no_completion", "memory_identity"])
def test_noneligible_node_keeps_legacy_path_without_value_read(tmp_path, monkeypatch, change):
    _graph, store, _root, plan, parent, expected = prepared(tmp_path)
    if change == "no_key":
        plan.keys["main"].pop(parent)
    elif change == "no_completion":
        plan.results["main"].pop(parent)
    else:
        expected = Node.from_domain(expected.lazy(include_traces=True), expected.domain,
            identity="legitimate.memory.value", value_type=expected.value_type,
            trace_columns=expected.trace_columns, trace_identity=expected.trace_identity)
    monkeypatch.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("noneligible parent read storage"))
    read = plan._window_reader("main", lambda: None, None, bounded_stored_parents=True)
    assert read is not None
    assert read(parent, expected, expected.domain.times[0], expected.domain.times[-1]) is expected


def test_contradictory_completed_authority_fails_before_read(tmp_path, monkeypatch):
    _graph, store, _root, plan, parent, expected = prepared(tmp_path)
    plan.results["main"][parent] = "different.immutable.receipt"
    monkeypatch.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("contradictory authority read values"))
    read = plan._window_reader("main", lambda: None, None, bounded_stored_parents=True)
    assert read is not None
    with pytest.raises(ValueError, match="authority differs"):
        read(parent, expected, expected.domain.times[0], expected.domain.times[-1])


def test_custom_store_keeps_original_parent_path(tmp_path):
    _graph, store, _root, plan, parent, expected = prepared(tmp_path)
    class CustomStore:
        def __getattr__(self, name):
            if name == "read_values":
                raise AssertionError("custom store gained a new read_values requirement")
            return getattr(store, name)
    plan.store = CustomStore()
    read = plan._window_reader("main", lambda: None, None, bounded_stored_parents=True)
    assert read is not None
    assert read(parent, expected, expected.domain.times[0], expected.domain.times[-1]) is expected


def test_missing_admitted_receipt_never_falls_back_to_cached_node(tmp_path):
    _graph, store, root, plan, parent, expected = prepared(tmp_path)
    with sqlite3.connect(store.meta_path) as connection:
        connection.execute("DELETE FROM materializations WHERE identity=?", (expected.identity,))
    read = plan._window_reader("main", plan.guard, None, bounded_stored_parents=True)
    assert read is not None
    with pytest.raises(KeyError, match="unknown Core materialization"):
        read(parent, expected, expected.domain.times[0], expected.domain.times[-1])
    assert root not in plan.results["main"]


@pytest.mark.parametrize("corruption", ["missing_value", "wrong_type", "missing_trace", "null_domain"])
def test_selected_parent_damage_is_rejected_before_child_kernel(tmp_path, monkeypatch, corruption):
    _graph, store, root, plan, parent, expected = prepared(tmp_path)
    manifest = store.manifest(expected.identity)
    assert manifest is not None
    path = store._file(manifest["domain"]["calendar"]["$file"] if corruption == "null_domain"
        else manifest["partitions"][0]["$file"])
    if corruption == "missing_value":
        path.unlink()
    else:
        frame = pl.read_parquet(path)
        if corruption == "wrong_type":
            frame = frame.with_columns(pl.col("value").cast(pl.String))
        elif corruption == "missing_trace":
            frame = frame.drop("observed_at")
        else:
            frame = pl.concat([frame, pl.DataFrame({"time": [None]}, schema={"time": pl.Date})])
        frame.write_parquet(path)
    from bagelquant_core import ExecutionRuntime
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("damaged parent ran child kernel"))
    with pytest.raises((ValueError, TypeError, OSError, pl.exceptions.PolarsError)):
        plan.execute("main", root)
    assert root not in plan.results["main"]
    assert store.graph_state("global")["current_update"] is None


def test_default_reader_keeps_checkpoint_none_and_parent_semantics(tmp_path, monkeypatch):
    _graph, store, _root, plan, parent, expected = prepared(tmp_path)
    assert plan._window_reader("main", lambda: None, None) is None
    calls = []
    plan._input_window_loader = lambda *_args: calls.append(_args)
    monkeypatch.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("checkpoint parent used finite read"))
    read = plan._window_reader("main", lambda: None, None)
    assert read is not None
    assert read(parent, expected, expected.domain.times[0], expected.domain.times[-1]) is expected
    assert calls == []


def test_same_cutoff_repeat_uses_no_values_or_kernels(tmp_path, monkeypatch):
    x = source(70)
    graph, store, _root = global_graph(tmp_path, rolling_mean(negate(x), window=4))
    before = finish(graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    def forbidden(*_args, **_kwargs):
        pytest.fail("same-cutoff repeat consumed values or ran kernel")
    monkeypatch.setattr(store, "read_values", forbidden)
    monkeypatch.setattr(store, "_value_plan", forbidden)
    from bagelquant_core import ExecutionRuntime
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(),
        input_loader=forbidden, input_window_loader=forbidden)
    after = finish(plan)
    assert after["results"] == before["results"]
    assert all(u["computed_blocks"] == u["input_sessions"] == 0 for u in plan.resource_usage["main"].values())


@pytest.mark.parametrize("revision", [None, 40])
def test_append_and_local_repair_preserve_originals_and_full_reference(tmp_path, monkeypatch, revision):
    first = source(65)
    graph, store, root = global_graph(tmp_path / "new", pct_change(negate(first), periods=4))
    before = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    old_manifest = store.manifest(before["results"]["main"][root])
    assert old_manifest is not None
    originals = {p["$file"]: (store._file(p["$file"]).read_bytes(), store._file(p["$file"]).stat().st_mtime_ns)
                 for p in old_manifest["partitions"]}
    current = source(67, revision=revision)
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max())
    after = finish(plan)
    assert any(u["reused_blocks"] for u in plan.resource_usage["main"].values())
    assert all(u["computed_output_sessions"] < 67 for u in plan.resource_usage["main"].values())
    for relative, (raw, mtime) in originals.items():
        assert store._file(relative).read_bytes() == raw
        assert store._file(relative).stat().st_mtime_ns == mtime
    assert store.manifest(before["results"]["main"][root]) == old_manifest
    reference, old, old_root = global_graph(tmp_path / "reference", pct_change(negate(current), periods=4))
    with monkeypatch.context() as patch:
        legacy_windows(patch)
        full = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    assert after["results"] == full["results"]
    assert_frame_equal(values(store, after, root), values(old, full, old_root), check_exact=True)


@pytest.mark.parametrize("cancel_after_read", [False, True])
def test_cancellation_guards_bound_read_and_preserve_child_completion(tmp_path, monkeypatch, cancel_after_read):
    _graph, store, root, plan, parent, expected = prepared(tmp_path)
    calls = []
    original = CoreStore.read_values
    def read(reader, *args, **kwargs):
        if reader is store:
            calls.append("read")
        return original(reader, *args, **kwargs)
    monkeypatch.setattr(CoreStore, "read_values", read)
    def guard():
        if not cancel_after_read or calls:
            raise RuntimeError("canceled synthetic read")
    acquire = plan._window_reader("main", guard, None, bounded_stored_parents=True)
    assert acquire is not None
    with pytest.raises(RuntimeError, match="canceled synthetic"):
        acquire(parent, expected, expected.domain.times[0], expected.domain.times[-1])
    assert calls == (["read"] if cancel_after_read else [])
    assert root not in plan.results["main"]


def test_revision_after_read_prevents_child_publication(tmp_path, monkeypatch):
    graph, store, root, plan, _parent, _expected = prepared(tmp_path)
    original = CoreStore.read_values
    def read(reader, *args, **kwargs):
        result = original(reader, *args, **kwargs)
        if reader is store:
            graph.sleep(root)
        return result
    monkeypatch.setattr(CoreStore, "read_values", read)
    with pytest.raises(RevisionConflict):
        plan.execute("main", root)
    assert root not in plan.results["main"]
    assert store.graph_state("global")["current_update"] is None


def test_parallel_parent_windows_keep_worker_limits_and_temporary_nodes_expire(tmp_path, monkeypatch):
    _graph, store, _root, plan, parent, expected = prepared(tmp_path)
    barrier, lock = Barrier(2), Lock()
    observed = []
    original = CoreStore.read_values
    def read(reader, *args, **kwargs):
        if reader is store:
            with lock:
                observed.append(active_resource_limits())
            barrier.wait(timeout=10)
        return original(reader, *args, **kwargs)
    monkeypatch.setattr(CoreStore, "read_values", read)
    acquire = plan._window_reader("main", plan.guard, None, bounded_stored_parents=True)
    assert acquire is not None
    limits = ResourceLimits(total_threads=1, memory_target_mib=64, cache_mib=8)
    def worker(offset):
        with resource_limits(limits):
            result = acquire(parent, expected, expected.domain.times[offset], expected.domain.times[offset + 2])
        output = result.collect(dense=False, include_traces=True)
        return weakref.ref(result), output
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, (0, 33)))
    gc.collect()
    assert observed == [limits, limits]
    assert all(ref() is None for ref, _frame in results)
    assert plan._values["main"][parent] is expected
    for (ref, actual), offset in zip(results, (0, 33), strict=True):
        assert ref() is None
        before = expected.lazy(include_traces=True).filter(pl.col("time").is_between(
            expected.domain.times[offset], expected.domain.times[offset + 2])).collect().sort("time", "asset_id")
        assert_frame_equal(actual, before, check_exact=True)


def test_scalar_float_bits_and_datetime_trace_survive_bounded_parent_acquisition(tmp_path):
    _graph, _store, _root, plan, parent, expected = prepared(tmp_path, datetime_trace=True)
    acquire = plan._window_reader("main", plan.guard, None, bounded_stored_parents=True)
    assert acquire is not None
    actual = acquire(parent, expected, expected.domain.times[0], expected.domain.times[-1]).collect(
        dense=False, include_traces=True)
    before = expected.collect(dense=False, include_traces=True)
    assert_frame_equal(actual, before, check_exact=True)
    assert actual.schema["observed_at"] == pl.Datetime("us")
    assert np.array_equal(actual["value"].is_null().to_numpy(), before["value"].is_null().to_numpy())
    assert actual["value"].drop_nulls().to_numpy().tobytes() == before["value"].drop_nulls().to_numpy().tobytes()


@pytest.mark.parametrize("case", ["static", "reentry", "empty"])
@pytest.mark.parametrize("kind", ["numeric", "category"])
def test_unaligned_stored_windows_share_complete_historical_slabs(case, kind):
    value = source(case=case, kind=kind, datetime_trace=True)
    days = value.domain.times.to_list()
    original = value.collect(dense=False, include_traces=True)
    preparation = ExecutionPreparation(1024 * 1024)
    reads = []
    def load(_parent, bound, first, last):
        assert bound is value
        reads.append((first, last))
        return value
    with preparation_scope(preparation):
        for first, last in ((16, 63), (48, 95), (80, 100)):
            actual = _parent_window("parent", value, days[first], days[last],
                input_window_loader=load, stored_parent=True, check_canceled=None)
            expected = original.filter(pl.col("time").is_between(days[first], days[last]))
            assert_frame_equal(actual, expected, check_exact=True)
            if kind == "numeric":
                assert actual["value"].drop_nulls().to_numpy().tobytes() == expected["value"].drop_nulls().to_numpy().tobytes()
        assert reads == [(days[0], days[31]), (days[32], days[63]),
                         (days[64], days[95]), (days[96], days[100])]
        assert preparation.counts["parent_slab_builds"] == 4
        assert preparation.counts["parent_slab_hits"] == 2
        assert preparation.bytes <= preparation.maximum_bytes


@pytest.mark.parametrize("stored,budget", [(False, 1024 * 1024), (True, 0), (True, 4096)])
def test_unaligned_external_or_nonadmitted_windows_keep_exact_reads(stored, budget):
    value = source(case="static")
    days = value.domain.times.to_list()
    preparation = ExecutionPreparation(budget)
    reads = []
    def load(_parent, _bound, first, last):
        reads.append((first, last))
        return value
    with preparation_scope(preparation):
        # A one-session request fits 4096 bytes; its complete 32-session
        # historical slab does not. Admission must include that extra prefix.
        actual = _parent_window("parent", value, days[31], days[31],
            input_window_loader=load, stored_parent=stored, check_canceled=None)
        assert reads == [(days[31], days[31])]
        expected = value.collect(dense=False, include_traces=True).filter(pl.col("time") == days[31])
        assert_frame_equal(actual, expected, check_exact=True)
        assert preparation.bytes <= budget
        if stored and budget:
            assert preparation.counts["parent_window_bulk_bypasses"] == 1
            assert preparation.counts["parent_slab_builds"] == 0


def test_stored_slab_alignment_never_extends_the_requested_end():
    value = source()
    days = value.domain.times.to_list()
    reads = []
    def load(_parent, _bound, first, last):
        reads.append((first, last))
        return value
    with preparation_scope(ExecutionPreparation(1024 * 1024)):
        actual = _parent_window("parent", value, days[17], days[36],
            input_window_loader=load, stored_parent=True, check_canceled=None)
    assert reads == [(days[0], days[31]), (days[32], days[36])]
    assert_frame_equal(actual, value.collect(dense=False, include_traces=True).filter(
        pl.col("time").is_between(days[17], days[36])), check_exact=True)


def test_dense_historical_prefix_is_charged_before_a_sparse_stored_request():
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(32)]
    rows = [(day, f"A{asset}", float(asset), day) for day in days[:-1] for asset in range(20)]
    rows.append((days[-1], "A0", 0.0, days[-1]))
    frame = pl.DataFrame(rows, schema={"time": pl.Date, "asset_id": pl.String,
        "value": pl.Float64, "available_date": pl.Date}, orient="row")
    domain = Domain(calendar=days, universe=frame.select("time", "asset_id").with_columns(
        pl.lit(True).alias("active")))
    value = Node.from_domain(frame, domain, source_key="x", trace_columns=("available_date",))
    preparation = ExecutionPreparation(8192)
    reads = []
    def load(_parent, _bound, first, last):
        reads.append((first, last))
        return value
    with preparation_scope(preparation):
        actual = _parent_window("parent", value, days[-1], days[-1],
            input_window_loader=load, stored_parent=True, check_canceled=None)
    assert reads == [(days[-1], days[-1])]
    assert actual.height == 1
    assert preparation.counts["parent_window_bulk_bypasses"] == 1
    assert preparation.counts["parent_window_bulk_estimated_bytes"] == domain.size * 48
    assert preparation.counts["parent_slab_builds"] == 0
    assert preparation.bytes <= preparation.maximum_bytes


def test_stored_window_uses_actual_sessions_for_noncalendar_endpoints():
    value = source()
    days = value.domain.times.to_list()
    start, end = days[4] + timedelta(days=1), days[34] + timedelta(days=1)
    assert start not in days and end not in days
    reads = []
    def load(_parent, _bound, first, last):
        reads.append((first, last))
        return value
    with preparation_scope(ExecutionPreparation(1024 * 1024)):
        actual = _parent_window("parent", value, start, end,
            input_window_loader=load, stored_parent=True, check_canceled=None)
        assert reads == [(days[0], days[31]), (days[32], days[34])]
        assert_frame_equal(actual, value.collect(dense=False, include_traces=True).filter(
            pl.col("time").is_between(start, end)), check_exact=True)
        for first, last in ((start, start), (days[-1] + timedelta(days=1), days[-1] + timedelta(days=2)),
                            (days[2], days[1])):
            with pytest.raises(ValueError, match="calendar must contain at least one time"):
                _parent_window("parent", value, first, last,
                    input_window_loader=load, stored_parent=True, check_canceled=None)
        assert len(reads) == 2
