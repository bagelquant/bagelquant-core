"""One admitted coordinate index preserves finite proofs and worker budgets."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
import gc
import hashlib
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Callable, cast
import weakref

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from bagelquant_core import Domain, Node
from bagelquant_core._finite_time_index import _FiniteTimeIndex, _FiniteTimeIndexLease
import bagelquant_core.incremental as incremental
from bagelquant_core.operator import negate as _negate, pct_change as _pct_change, smooth as _smooth
from bagelquant_core.resources import ResourceLimits, active_resource_limits, resource_limits
import bagelquant_core.update as update
from test_global_graph import finish, global_graph


negate = cast(Callable[[Node], Node], _negate)
pct_change = cast(Callable[..., Node], _pct_change)
smooth = cast(Callable[[Node], Node], _smooth)
PCT = "bagelquant_core.operator.basic.pct_change"
LIMITS = ResourceLimits(total_threads=1, parallel_nodes=1, memory_target_mib=128, cache_mib=16)


def source(count=98, *, revision=None, empty=False):
    days = [date(2024, 1, 1) + timedelta(days=i + i // 7 * 3) for i in range(count)]
    rows = []
    for i, day in enumerate(days):
        members = [] if empty else ["A"]
        if not empty:
            members += (["RETIRED"] if i < 3 else [])
            members += (["REENTRY"] if i < 7 or i >= 80 else [])
            members += (["NEW"] if i >= 75 else [])
        for asset in members:
            value = (None, -0.0, 0.0, float("nan"), float("inf"), 3.5, -2.5)[i % 7]
            rows.append((day, asset, 19.0 if revision == i else value,
                None if i % 13 == 0 else day + timedelta(days=1), day))
    frame = pl.DataFrame(rows, orient="row", schema={"time": pl.Date, "asset_id": pl.String,
        "value": pl.Float64, "available_date": pl.Date, "observed_at": pl.Date})
    domain = Domain(calendar=days, universe=frame.select("time", "asset_id")
        .with_columns(pl.lit(True).alias("active")))
    return Node.from_domain(frame, domain, name="x", source_key="x",
        trace_columns=("available_date", "observed_at"), trace_identity="time-index-fixture.v1")


def lease_for(domain, history=9, limits=LIMITS, operator=PCT):
    return incremental._finite_time_index_reservation(domain, history, operator, limits)


@pytest.mark.parametrize("history", [1, 9, 240])
def test_index_queries_match_original_for_every_missing_prefix(history):
    domain = source().domain
    days = domain.times.to_list()
    lease = lease_for(domain, history)
    expected = incremental._canonical_halos(domain, history)
    with resource_limits(lease.limits), lease:
        actual = incremental._canonical_halos(domain, history, _time_index_lease=lease)
        assert actual == expected
        assert lease.index is not None
        for first in range(len(days)):
            for width in (0, 1, 5):
                last = min(first + width, len(days) - 1)
                assert lease.index.numerical_read_start(days[first], days[last], history) == (
                    incremental._numerical_read_start(domain, days[first], days[last], history))
        assert lease.index.canonical_halos(0, incremental.CANONICAL_SESSIONS) == incremental._canonical_halos(domain, 0)
        with pytest.raises(ValueError, match="not in list"):
            lease.index.numerical_read_start(days[0] - timedelta(days=1), days[-1], history)
    assert lease.index is None and lease.closed


@pytest.mark.parametrize("case", ["history0", "static", "empty", "nonexact", "zero", "scratch"])
def test_nonadmission_does_not_build_or_expand_membership(monkeypatch, case):
    x = source(empty=case == "empty")
    domain, history, operator, limits = x.domain, 9, PCT, LIMITS
    if case == "history0":
        history = 0
    elif case == "static":
        domain = Domain(calendar=domain.times, universe=["A", "B"])
    elif case == "nonexact":
        operator = "bagelquant_core.operator.rolling_stats.rolling_mean"
    elif case == "zero":
        limits = replace(limits, cache_mib=0)
    elif case == "scratch":
        days = domain.times.to_list()
        membership = pl.DataFrame([(day, f"ASSET{asset}") for day in days for asset in range(32)],
            schema={"time": pl.Date, "asset_id": pl.String}, orient="row")
        domain = Domain(calendar=days, universe=membership.with_columns(pl.lit(True).alias("active")))
        limits = replace(limits, memory_target_mib=2, cache_mib=1)
    monkeypatch.setattr(Domain, "grid_lazy", lambda *_: pytest.fail("admission expanded membership"))
    lease = lease_for(domain, history, limits, operator)
    assert not lease.admitted and lease.index is None
    assert lease.limits is limits and lease.reserved_bytes == lease.actual_retained_bytes == 0


def test_owned_calendar_charge_and_scratch_headroom_are_exact():
    domain = source().domain
    lease = lease_for(domain)
    raw = 4 * domain.size + 8 * (len(domain.asset_ids) + 1) + 4 * len(domain.times) + 4096
    assert lease.raw_bytes == raw
    assert lease.reserved_bytes == ((raw + 1024**2 - 1) // 1024**2) * 1024**2
    assert lease.limits == replace(LIMITS, cache_mib=LIMITS.cache_mib - lease.reserved_bytes // 1024**2)
    with resource_limits(lease.limits), lease:
        incremental._canonical_halos(domain, 9, _time_index_lease=lease)
        assert lease.index is not None
        assert lease.actual_retained_bytes == raw - 4096
        for array in (lease.index.times, lease.index.offsets, lease.index.calendar):
            assert array is not None and array.flags.owndata and array.base is None
            assert not array.flags.writeable
        assert lease.scratch_bytes <= (LIMITS.memory_target_mib - LIMITS.cache_mib) * 1024**2 // 8


@pytest.mark.parametrize("extra_rows,admitted", [(0, True), (1, False)])
def test_exact_mib_admission_boundary_uses_metadata_only(extra_rows, admitted):
    fixed = 8 * (1 + 1) + 4 + 4096
    rows = (1024**2 - fixed) // 4 + extra_rows
    metadata = cast(Domain, SimpleNamespace(is_dynamic=True, size=rows,
        asset_ids=pl.Series(["A"]), times=pl.Series([date(2024, 1, 1)])))
    limits = replace(LIMITS, cache_mib=1)
    lease = lease_for(metadata, limits=limits)
    assert lease.raw_bytes == 1024**2 + extra_rows * 4
    assert lease.admitted is admitted
    assert lease.reserved_bytes == (1024**2 if admitted else 0)
    assert lease.index is None


def test_equal_opaque_domain_ids_do_not_share_index():
    first = source().domain
    second = Domain(calendar=first.times, universe=["OTHER"], identity=first.signature)
    dynamic = Domain(calendar=first.times, universe=second.grid_lazy().collect()
        .with_columns(pl.lit(True).alias("active")), identity=first.signature)
    with lease_for(first) as left, lease_for(dynamic) as right:
        with pytest.raises(ValueError, match="different or closed Domain"):
            incremental._canonical_halos(dynamic, 9, _time_index_lease=left)
        a = incremental._canonical_halos(first, 9, _time_index_lease=left)
        b = incremental._canonical_halos(dynamic, 9, _time_index_lease=right)
        assert a != b and left.index is not right.index


def test_constructor_failure_and_retained_traceback_release_arrays(monkeypatch):
    domain, refs = source().domain, []
    failure = MemoryError("owned allocation failed")
    def broken(index, *_):
        index.times = np.array([1, 2], dtype=np.int32)
        refs.append(weakref.ref(index.times))
        raise failure
    monkeypatch.setattr(_FiniteTimeIndex, "build", broken)
    lease = lease_for(domain)
    with pytest.raises(MemoryError) as caught:
        with resource_limits(lease.limits), lease:
            incremental._canonical_halos(domain, 9, _time_index_lease=lease)
    assert caught.value is failure and caught.tb is not None
    gc.collect()
    assert refs[0]() is None and lease.closed and lease.index is None


@pytest.mark.parametrize("partial", [True, False])
def test_build_tracebacks_release_argument_frames_and_calendar(monkeypatch, partial):
    domain = source().domain
    membership = domain.grid_lazy().sort("asset_id", "time").collect()
    calendar = domain.times.cast(pl.Int32).to_numpy()
    refs = [weakref.ref(membership), weakref.ref(calendar)]
    lease = lease_for(domain)
    if partial:
        failure = MemoryError("offset allocation failed")
        def broken(*args, **kwargs):
            try:
                assert lease.index is not None
                assert lease.index.times is not None and lease.index.offsets is None
                refs.append(weakref.ref(lease.index.times))
                raise failure
            finally:
                del args, kwargs
        monkeypatch.setattr(np, "empty", broken)
    else:
        size = _FiniteTimeIndex.retained_bytes.fget
        assert size is not None
        def invalid_charge(index):
            refs.extend(weakref.ref(value) for value in index._buffers())
            return size(index) + lease.raw_bytes
        monkeypatch.setattr(_FiniteTimeIndex, "retained_bytes", property(invalid_charge))
    with pytest.raises((MemoryError, ValueError)) as caught:
        with resource_limits(lease.limits), lease:
            lease.build_from_sorted(membership, calendar)
    del membership, calendar
    gc.collect()
    assert caught.tb is not None and lease.index is None and lease.closed
    assert all(ref() is None for ref in refs)


def test_query_failure_closes_before_resource_restoration(monkeypatch):
    domain = source().domain
    lease = lease_for(domain)
    refs, closing = [], []
    original = _FiniteTimeIndexLease.close
    def close(selected):
        closing.append(active_resource_limits())
        original(selected)
    monkeypatch.setattr(_FiniteTimeIndexLease, "close", close)
    failure = RuntimeError("query canceled")
    with resource_limits(LIMITS):
        with pytest.raises(RuntimeError) as caught:
            with resource_limits(lease.limits), lease:
                incremental._canonical_halos(domain, 9, _time_index_lease=lease)
                assert lease.index is not None
                class FailedSearch(np.ndarray):
                    def searchsorted(self, *args, **kwargs):
                        try:
                            raise failure
                        finally:
                            # This fake owns its foreign frame. Core does not
                            # clear foreign/caller tracebacks or rewrite errors.
                            del self, args, kwargs
                assert lease.index.calendar is not None
                refs.append(weakref.ref(lease.index.calendar))
                lease.index.calendar = lease.index.calendar.view(FailedSearch)
                refs.extend(weakref.ref(array) for array in lease.index._buffers())
                lease.index.numerical_read_start(domain.times[1], domain.times[-1], 9)
        assert active_resource_limits() is LIMITS
    assert caught.value is failure and closing == [lease.limits]
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_parallel_leases_use_only_already_divided_cache():
    domain, barrier = source().domain, Barrier(2)
    allocation = replace(LIMITS, total_threads=2, parallel_nodes=2, cache_mib=16)
    per_worker = allocation.for_workers(2)
    def worker(_):
        lease = lease_for(domain, limits=per_worker)
        with resource_limits(lease.limits), lease:
            incremental._canonical_halos(domain, 9, _time_index_lease=lease)
            barrier.wait(timeout=5)
            assert active_resource_limits() is lease.limits
            assert lease.index is not None
            return lease.reserved_bytes + lease.limits.cache_mib * 1024**2
    with ThreadPoolExecutor(max_workers=2) as pool:
        used = list(pool.map(worker, range(2)))
    assert sum(used) <= allocation.cache_mib * 1024**2


def assert_bits(left, right):
    assert_frame_equal(left, right, check_exact=True)
    for name, dtype in left.schema.items():
        if dtype == pl.Float64:
            assert left[name].drop_nulls().to_numpy().tobytes() == right[name].drop_nulls().to_numpy().tobytes()


@pytest.mark.parametrize("operation", ["pct_change", "smooth"])
def test_public_graph_cold_append_repair_and_history_match_old_queries(tmp_path, monkeypatch, operation):
    factory = (lambda x: pct_change(negate(x), periods=5)) if operation == "pct_change" else (lambda x: smooth(negate(x)))
    snapshots = [source(64), source(66), source(66, revision=40)]
    def build(path, disabled):
        graph, store, root = global_graph(path, factory(snapshots[0]))
        records, usage = [], []
        for x in snapshots:
            plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max())
            if disabled:
                with monkeypatch.context() as old:
                    old.setattr(update, "_finite_time_index_reservation", lambda domain, history, operator, limits:
                        _FiniteTimeIndexLease(domain, history, limits, eligible=False,
                            canonical_sessions=incremental.CANONICAL_SESSIONS))
                    result = finish(plan, limits=LIMITS)
            else:
                result = finish(plan, limits=LIMITS)
            records.append(result)
            usage.append(dict(plan.resource_usage["main"][root]))
        return graph, store, root, records, usage
    _, old, old_root, reference, old_usage = build(tmp_path / "old", True)
    graph, store, root, results, usage = build(tmp_path / "new", False)
    assert usage == old_usage
    assert [x["results"] for x in results] == [x["results"] for x in reference]
    for result, expected in zip(results, reference, strict=True):
        assert_bits(store.read_values(result["results"]["main"][root], include_traces=True),
            old.read_values(expected["results"]["main"][old_root], include_traces=True))
    # Same-boundary completed authority bypasses lease creation and all kernels.
    monkeypatch.setattr(update, "_finite_time_index_reservation", lambda *_: pytest.fail("completed work reserved index"))
    repeated = graph.plan_update({"main": {"x": snapshots[-1]}}, through=snapshots[-1].domain.times.max())
    assert repeated.complete and finish(repeated)["results"] == results[-1]["results"]
    assert usage[1]["computed_output_sessions"] == 2 and usage[2]["reused_blocks"] > 0


@pytest.mark.parametrize("windowed", [False, True])
def test_same_remainder_reaches_finite_loaders_and_runtime(tmp_path, monkeypatch, windowed):
    x, seen, runtimes, leases, views = source(66), [], [], [], []
    placeholder = Node.from_domain(x.lazy(include_traces=True).filter(pl.lit(False)), x.domain,
        source_key="x", identity=x.identity, trace_columns=x.trace_columns, trace_identity=x.trace_identity)
    graph, _, root = global_graph(tmp_path, pct_change(placeholder, periods=5))
    reserve = update._finite_time_index_reservation
    def capture(*args):
        lease = reserve(*args)
        leases.append(lease)
        return lease
    monkeypatch.setattr(update, "_finite_time_index_reservation", capture)
    view_factory = incremental._PctKernelView
    def capture_view(*args):
        view = view_factory(*args)
        views.append(view)
        return view
    monkeypatch.setattr(incremental, "_PctKernelView", capture_view)
    runtime_init = incremental.ExecutionRuntime.__init__
    def observe_runtime(runtime, *args, **kwargs):
        runtime_init(runtime, *args, **kwargs)
        if leases and leases[-1].admitted and not leases[-1].closed:
            runtimes.append(runtime._effective_limits)
    monkeypatch.setattr(incremental.ExecutionRuntime, "__init__", observe_runtime)
    def full(*_):
        seen.append(active_resource_limits())
        return x
    def window(_context, _key, start, end):
        seen.append(active_resource_limits())
        return Node.from_domain(x.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
            x.domain.window(start, end), identity=x.identity,
            trace_columns=x.trace_columns, trace_identity=x.trace_identity)
    options = {"input_loader": full,
        "input_proof": lambda _context, _key, start, end: x.interval_identity(start, end)}
    if windowed:
        options["input_window_loader"] = window
    plan = graph.plan_update({"main": {"x": placeholder}}, through=x.domain.times.max(), **options)
    finish(plan, limits=LIMITS)
    assert len(leases) == 1 and leases[0].admitted and leases[0].closed
    assert seen and all(item is leases[0].limits for item in seen)
    # The input reader keeps A's exact remainder. B separately reserves only
    # working headroom before its dense validation and the existing kernel.
    assert views and all(view.admitted and view.closed for view in views)
    assert runtimes == [view.limits for view in views]
    assert all(view.limits == replace(leases[0].limits,
        memory_target_mib=leases[0].limits.memory_target_mib - view.reserve_mib) for view in views)
    assert leases[0].limits.cache_mib < LIMITS.cache_mib
    assert plan.resource_usage["main"][root]["computed_blocks"] == 3


@pytest.mark.parametrize("failure_site", ["window", "cancel", "publish"])
def test_public_failure_releases_lease_and_retains_exception(tmp_path, monkeypatch, failure_site):
    x, leases, calls = source(66), [], []
    graph, store, _ = global_graph(tmp_path, pct_change(x, periods=5))
    failure = RuntimeError(f"failed at {failure_site}")
    reserve = update._finite_time_index_reservation
    def capture(*args):
        lease = reserve(*args)
        leases.append(lease)
        return lease
    monkeypatch.setattr(update, "_finite_time_index_reservation", capture)
    def window(_context, _key, start, end):
        calls.append((start, end))
        if failure_site == "window":
            raise failure
        return Node.from_domain(x.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
            x.domain.window(start, end), identity=x.identity,
            trace_columns=x.trace_columns, trace_identity=x.trace_identity)
    def cancel():
        if failure_site == "cancel" and leases and leases[0].index is not None:
            raise failure
    if failure_site == "publish":
        monkeypatch.setattr(store, "publish", lambda *_args, **_kwargs: (_ for _ in ()).throw(failure))
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(), input_window_loader=window)
    with resource_limits(LIMITS):
        with pytest.raises(RuntimeError) as caught:
            finish(plan, limits=LIMITS, check_canceled=cancel)
        assert active_resource_limits() is LIMITS
    assert caught.value is failure and caught.tb is not None
    assert leases and leases[0].closed and leases[0].index is None
    assert leases[0]._domain is None and not plan._in_flight
    assert bool(calls) is (failure_site != "cancel")


def test_interrupted_all_block_reassembly_runs_no_kernel_or_window_read(tmp_path, monkeypatch):
    x = source(66)
    graph, store, root = global_graph(tmp_path, pct_change(x, periods=5))
    windows, leases = [], []
    def window(_context, _key, start, end):
        windows.append((start, end))
        return Node.from_domain(x.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
            x.domain.window(start, end), identity=x.identity,
            trace_columns=x.trace_columns, trace_identity=x.trace_identity)
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(), input_window_loader=window)
    identity = plan.keys["main"][root].identity
    publish = store.publish
    failure = RuntimeError("stopped before enclosing receipt")
    def interrupt(value, **kwargs):
        if value.key.identity == identity:
            raise failure
        return publish(value, **kwargs)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(store, "publish", interrupt)
        with pytest.raises(RuntimeError, match="enclosing receipt"):
            finish(plan, limits=LIMITS)
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.artifact_path.rglob("*") if path.is_file()}
    windows.clear()
    reserve = update._finite_time_index_reservation
    def capture(*args):
        lease = reserve(*args)
        leases.append(lease)
        return lease
    monkeypatch.setattr(update, "_finite_time_index_reservation", capture)
    resumed = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(), input_window_loader=window)
    finish(resumed, limits=LIMITS)
    assert not windows and resumed.resource_usage["main"][root]["computed_blocks"] == 0
    assert resumed.resource_usage["main"][root]["reused_blocks"] == 3
    assert leases and leases[0].actual_retained_bytes > 0 and leases[0].closed
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum
        for path, checksum in before.items())
