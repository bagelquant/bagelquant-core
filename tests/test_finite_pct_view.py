"""The same percentage kernel consumes fewer keys with identical authority."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, timedelta
import gc
import hashlib
import struct
from types import SimpleNamespace
import weakref

import polars as pl
import pytest

from bagelquant_core import Domain, ExecutionRuntime, Node
import bagelquant_core._finite_pct_view as pct_view
import bagelquant_core.incremental as incremental
from bagelquant_core.operator import (
    canonicalize_values, constant, div, mul, pct_change, project_domain, rank, smooth,
)
from bagelquant_core.resources import ResourceLimits, active_resource_limits, resource_limits
from test_global_graph import finish, global_graph


PCT = "bagelquant_core.operator.basic.pct_change"
LIMITS = ResourceLimits(total_threads=1, parallel_nodes=1, memory_target_mib=128, cache_mib=16)


def bits(frame):
    return (tuple((name, str(dtype)) for name, dtype in frame.schema.items()),
        [tuple(struct.pack(">d", value).hex() if isinstance(value, float) else value for value in row)
         for row in frame.iter_rows()])


def source(count=640, *, revision=None, implicit=True, dtype=pl.Float64, extra=False):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]
    keys, rows = [], []
    special = (None, -0.0, 0.0, 1e-300, -1e-300, 1e300, -1e300,
        float("inf"), -float("inf"), float("nan"), 3.0, -7.0)
    for i, day in enumerate(days):
        assets = ["D"]
        if i < 240 or i >= 608:
            assets.append("S")
        if i < 400:
            assets.append("R")
        if extra and i >= 638:
            assets.append("NEW")
        for asset in assets:
            keys.append((day, asset))
            if implicit and asset == "D" and i in (399, 608):
                continue
            value = 29.0 if revision == i and asset == "D" else special[i % len(special)]
            rows.append((day, asset, value, None if i % 13 == 0 else day + timedelta(days=1),
                None if i % 17 == 0 else day))
    membership = pl.DataFrame(keys, schema={"time": pl.Date, "asset_id": pl.String}, orient="row")
    domain = Domain(calendar=days, universe=membership.with_columns(pl.lit(True).alias("active")))
    frame = pl.DataFrame(rows, schema={"time": pl.Date, "asset_id": pl.String, "value": dtype,
        "available_date": pl.Date, "observed_at": pl.Date}, orient="row")
    return Node.from_domain(frame, domain, source_key="x", name="x",
        trace_columns=("available_date", "observed_at"))


def logical(operator=PCT, *, auxiliary=False, inputs=("parent",)):
    return SimpleNamespace(operator=operator, inputs=inputs,
        panel_parameters={"other": ["aux"]} if auxiliary else {})


def indexed_source_proof(value, calls=None):
    """One genuine public proof family from first build through continuation.

    The eager Node's construction-time index binds scoped calendar/membership,
    values and both Date channels. UpdatePlan additionally binds this callback's
    token through source_interval_identity; use that same route in every phase.
    """
    def proof(context, source_key, start, end):
        assert context == "main" and source_key == "x"
        if calls is not None:
            calls.append(("proof", start, end))
        token = value.interval_identity(start, end)
        assert token is not None
        return token
    return proof


@contextmanager
def original_path(monkeypatch):
    """Disable only the new private admission; use the same original pipeline."""
    factory = incremental._PctKernelView
    with monkeypatch.context() as patch:
        patch.setattr(incremental, "_PctKernelView",
            lambda node, _scoped, output, history, limits: factory(node, {}, output, history, limits))
        yield


def kernel(frame_node, periods):
    return ExecutionRuntime().run(pct_change(frame_node, periods=periods), dense_output=False)


@pytest.mark.parametrize("periods", [1, 240])
@pytest.mark.parametrize("layout", ["contiguous", "reverse", "chunks"])
def test_actual_dense_kernel_gets_private_membership_and_exact_float_trace_bits(periods, layout):
    x = source()
    if layout != "contiguous":
        frame = x.collect(dense=False, include_traces=True)
        frame = frame.reverse() if layout == "reverse" else pl.concat(
            [frame.slice(offset, 83) for offset in range(0, frame.height, 83)], rechunk=False)
        x = Node.from_domain(frame, x.domain, source_key="x", trace_columns=x.trace_columns)
    first, last = x.domain.times[608], x.domain.times[-1]
    output = x.domain.window(first, last)
    scope = pct_view._PctKernelView(logical(), {"parent": x}, output, periods, LIMITS)
    assert scope.admitted
    before = bits(x.collect(dense=False, include_traces=True))
    with resource_limits(LIMITS):
        reference = kernel(x, periods).collect(dense=False, include_traces=True).filter(pl.col("time") >= first)
    with resource_limits(scope.limits), scope:
        values = scope.prepare({"parent": x}, output, first, last, periods, lambda: None)
        private = values["parent"]
        actual_dense = private.collect(dense=True, include_traces=True)
        expected_rows = 2 * (32 + periods)
        assert actual_dense.height == private.domain.size == expected_rows
        assert private.domain.times.equals(x.domain.times)
        assert private.domain.asset_ids.to_list() == ["D", "S"]
        assert not actual_dense.filter(pl.col("asset_id") == "R").height
        result = kernel(private, periods).collect(dense=False, include_traces=True)
        result = result.filter(pl.col("time") >= first)
        assert bits(result) == bits(reference)
        # The absent source row is still an admitted null at the correct shift
        # position; filtering to actual/non-null source keys would fail this.
        missing = actual_dense.filter((pl.col("asset_id") == "D") & (pl.col("time") == first))
        assert missing.height == 1 and missing["value"][0] is None
    assert scope.closed and scope._inputs is None
    assert bits(x.collect(dense=False, include_traces=True)) == before


def test_public_finite_runtime_spy_observes_1312_to_544_not_a_rectangle(tmp_path, monkeypatch):
    x, observed = source(), []
    run = incremental.ExecutionRuntime.run
    def spy(runtime, graph, **kwargs):
        inputs = [node for node in graph.nodes if node.node_type == "input"]
        assert len(inputs) == 1
        node = inputs[0]
        observed.append((len(node.domain.times), node.collect(dense=True).height,
            node.domain.asset_ids.to_list()))
        return run(runtime, graph, **kwargs)
    monkeypatch.setattr(incremental.ExecutionRuntime, "run", spy)
    graph, store, root = global_graph(tmp_path / "view", pct_change(x, periods=240))
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max())
    saved = finish(plan, limits=LIMITS)
    assert observed[-1] == (640, 544, ["D", "S"])
    view_bits = bits(store.read(saved["results"]["main"][root]).panel.collect(dense=False, include_traces=True))
    observed.clear()
    other, old_store, old_root = global_graph(tmp_path / "original", pct_change(x, periods=240))
    with original_path(monkeypatch):
        old_plan = other.plan_update({"main": {"x": x}}, through=x.domain.times.max())
        old = finish(old_plan, limits=LIMITS)
    assert observed[-1] == (640, 1312, ["D", "R", "S"])
    assert saved["results"] == old["results"]
    assert plan.resource_usage == old_plan.resource_usage
    assert view_bits == bits(old_store.read(old["results"]["main"][old_root]).panel.collect(
        dense=False, include_traces=True))


def test_charge_units_divided_budget_and_known_sparse_calendar_scratch(monkeypatch):
    x = source()
    divided = ResourceLimits(total_threads=6, parallel_nodes=3,
        memory_target_mib=768, cache_mib=96).for_workers(3)
    scope = pct_view._PctKernelView(logical(), {"parent": x}, x.domain, 240, divided)
    raw = 6 * x.domain.size * (64 + 1 + 8 * 2) + 8 * 3 + 256 * 640 + 65536
    assert scope.raw_bytes == raw
    assert scope.reserve_mib == (raw + 1024**2 - 1) // 1024**2
    assert scope.limits == replace(divided, memory_target_mib=divided.memory_target_mib - scope.reserve_mib)
    assert scope.limits.cache_mib == divided.cache_mib
    # N << S: allocating another calendar must not evade admission just because
    # there is one active key. This is metadata only; no large value fixture.
    huge = SimpleNamespace(is_dynamic=True, size=1, asset_ids=pl.Series(["A"]),
        times=range(1_000_000))
    fake = SimpleNamespace(domain=huge, value_type=x.value_type, trace_columns=(),
        lazy=lambda **_: pl.DataFrame(schema={"value": pl.Float64}).lazy())
    monkeypatch.setattr(Node, "collect", lambda *_a, **_k: pytest.fail("admission collected values"))
    rejected = pct_view._PctKernelView(logical(), {"parent": fake}, huge, 240, LIMITS)
    assert rejected.raw_bytes >= 256 * 1_000_000 and not rejected.admitted
    assert rejected.reserve_mib == 0 and rejected.limits is LIMITS


@pytest.mark.parametrize("memory,admitted", [(20, True), (19, False)])
def test_reservation_exact_mib_boundary(memory, admitted):
    x = source()
    limits = replace(LIMITS, memory_target_mib=memory)
    scope = pct_view._PctKernelView(logical(), {"parent": x}, x.domain, 240, limits)
    assert 0 < scope.raw_bytes <= 1024**2
    assert scope.admitted is admitted
    assert scope.reserve_mib == (1 if admitted else 0)
    assert scope.limits is limits if not admitted else scope.limits == replace(limits, memory_target_mib=memory - 1)


def test_dense_validation_and_public_domain_allocation_follow_reservation(monkeypatch):
    x, events = source(), []
    output = x.domain.window(x.domain.times[608], x.domain.times[-1])
    scope = pct_view._PctKernelView(logical(), {"parent": x}, output, 240, LIMITS)
    collect, constructor = Node.collect, pct_view.Domain
    def observe_collect(node, **kwargs):
        assert active_resource_limits() is scope.limits
        result = collect(node, **kwargs)
        events.append(("dense", result.height))
        return result
    def observe_domain(**kwargs):
        assert active_resource_limits() is scope.limits
        assert events == [("dense", 1312)]
        assert kwargs["universe"]["active"].all()
        events.append(("domain", kwargs["universe"].height))
        return constructor(**kwargs)
    monkeypatch.setattr(Node, "collect", observe_collect)
    monkeypatch.setattr(pct_view, "Domain", observe_domain)
    with resource_limits(scope.limits), scope:
        scope.prepare({"parent": x}, output, x.domain.times[608], x.domain.times[-1], 240, lambda: None)
    assert events == [("dense", 1312), ("domain", 544)]


@pytest.mark.parametrize("kind", ["static", "float32", "prediction", "trace", "smooth", "aux", "multi", "empty", "zero", "tight"])
def test_nonadmission_retains_original_node_without_collecting(monkeypatch, kind):
    x = source(dtype=pl.Float32 if kind == "float32" else pl.Float64)
    node, history, limits = logical(), 240, LIMITS
    if kind == "static":
        x = Node.from_domain(x.lazy(include_traces=True), Domain(calendar=x.domain.times, universe=["D", "S"]),
            identity="static", trace_columns=x.trace_columns)
    elif kind == "prediction":
        x = Node.from_domain(x.lazy(include_traces=True), x.domain, value_type="prediction",
            identity="prediction", trace_columns=x.trace_columns)
    elif kind == "trace":
        frame = x.collect(dense=False, include_traces=True).with_columns(pl.col("available_date").cast(pl.String))
        x = Node.from_domain(frame, x.domain, trace_columns=x.trace_columns)
    elif kind == "smooth":
        node = logical("bagelquant_core.operator.rolling_stats.smooth")
    elif kind == "aux":
        node = logical(auxiliary=True)
    elif kind == "multi":
        node = logical(inputs=("parent", "other"))
    elif kind == "empty":
        x = Node.from_domain(pl.DataFrame(schema={"time": pl.Date,"asset_id": pl.String,"value": pl.Float64}),
            Domain(calendar=x.domain.times, universe=pl.DataFrame(schema={"time": pl.Date,"asset_id": pl.String,"active": pl.Boolean})))
    elif kind == "zero":
        history = 0
    elif kind == "tight":
        limits = replace(LIMITS, memory_target_mib=17, cache_mib=16)
    monkeypatch.setattr(Node, "collect", lambda *_a, **_k: pytest.fail("nonadmitted view collected values"))
    original = {"parent": x}
    scope = pct_view._PctKernelView(node, original, x.domain, history, limits)
    assert not scope.admitted and scope.limits is limits
    with resource_limits(limits), scope:
        assert scope.prepare(original, x.domain, x.domain.times[0], x.domain.times[-1], history, lambda: None) is original


def test_custom_replacement_does_not_enter_builtin_view(monkeypatch):
    x = source()
    monkeypatch.setattr(pct_change, "_plan_operation", lambda *_: pytest.fail("foreign plan ran"))
    scope = pct_view._PctKernelView(logical(), {"parent": x}, x.domain, 240, LIMITS)
    assert not scope.admitted


def test_partial_prefix_empty_tail_uses_original_input_and_runtime():
    x = source(640)
    # The canonical block has active keys, while the exact tail contains none.
    cutoff = x.domain.times[620]
    keys = x.domain.grid_lazy().filter(pl.col("time") < cutoff).collect()
    domain = Domain(calendar=x.domain.times, universe=keys.with_columns(pl.lit(True).alias("active")))
    original_node = Node.from_domain(x.lazy(include_traces=True), domain, trace_columns=x.trace_columns)
    canonical = domain.window(domain.times[608], domain.times[-1])
    scope = pct_view._PctKernelView(logical(), {"parent": original_node}, canonical, 240, LIMITS)
    assert canonical.size > 0 and scope.admitted
    original = {"parent": original_node}
    with resource_limits(scope.limits), scope:
        prepared = scope.prepare(original, canonical, cutoff, domain.times[-1], 240, lambda: None)
        assert prepared is original and scope._inputs is None
        result = kernel(prepared["parent"], 240).collect(dense=False).filter(pl.col("time") >= cutoff)
        assert result.height == 0


@pytest.mark.parametrize("damage", ["duplicate", "invalid_key", "missing_trace", "nonnumeric"])
def test_original_eager_validation_rejects_discarded_bad_prefix_before_view(damage):
    x = source()
    frame = x.collect(dense=False, include_traces=True)
    discarded = frame.filter(pl.col("asset_id") == "R").head(1)
    if damage == "duplicate":
        frame = pl.concat([frame, discarded])
    elif damage == "invalid_key":
        frame = frame.with_columns(pl.when((pl.col("asset_id") == "R") & (pl.col("time") == x.domain.times[0]))
            .then(None).otherwise(pl.col("time")).alias("time"))
    elif damage == "missing_trace":
        frame = frame.drop("observed_at")
    else:
        frame = frame.with_columns(pl.col("value").cast(pl.String))
    # This is the unchanged construction preceding the private admission.
    with pytest.raises((ValueError, TypeError)):
        Node.from_domain(frame, x.domain, trace_columns=x.trace_columns)


def test_public_append_rejects_duplicate_consumed_retired_row_before_admission(tmp_path, monkeypatch):
    before, current = source(638), source(640)
    graph, _store, _root = global_graph(tmp_path, pct_change(before, periods=240))
    finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max(),
        input_proof=indexed_source_proof(before)), limits=LIMITS)
    bad = current.collect(dense=False, include_traces=True)
    duplicate = bad.filter((pl.col("asset_id") == "R") & (pl.col("time") == current.domain.times[100]))
    assert duplicate.height == 1
    bad = pl.concat([bad, duplicate])
    def loader(_context, _source, start, end):
        # The ordinary lazy source boundary defers the duplicate check until
        # execute_finite_node's untrimmed eager public Node construction.
        return Node.from_domain(bad.lazy().filter(pl.col("time").is_between(start, end)),
            current.domain.window(start, end), identity=current.identity,
            trace_columns=current.trace_columns, trace_identity=current.trace_identity)
    monkeypatch.setattr(incremental, "_PctKernelView",
        lambda *_: pytest.fail("duplicate prefix reached view admission"))
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
        input_window_loader=loader,
        input_proof=indexed_source_proof(current))
    with pytest.raises(ValueError, match="unique"):
        finish(plan, limits=LIMITS)


@pytest.mark.parametrize("site", ["dense", "domain", "cancel"])
def test_own_scope_cleanup_preserves_exception_and_foreign_frames(monkeypatch, site):
    x = source()
    output = x.domain.window(x.domain.times[608], x.domain.times[-1])
    scope = pct_view._PctKernelView(logical(), {"parent": x}, output, 240, LIMITS)
    failure = RuntimeError(f"failure at {site}")
    held = []
    if site == "dense":
        def broken_collect(source, **_):
            held.append(source)
            raise failure
        monkeypatch.setattr(Node, "collect", broken_collect)
    elif site == "domain":
        def broken_domain(**kwargs):
            held.append(kwargs)
            raise failure
        monkeypatch.setattr(pct_view, "Domain", broken_domain)
    checks = []
    def check():
        checks.append(active_resource_limits())
        if site == "cancel" and len(checks) == 3:
            raise failure
    with resource_limits(LIMITS):
        with pytest.raises(RuntimeError) as caught:
            with resource_limits(scope.limits), scope:
                scope.prepare({"parent": x}, output, x.domain.times[608], x.domain.times[-1], 240, check)
        assert active_resource_limits() is LIMITS
    assert caught.value is failure and caught.tb is not None and scope.closed and scope._inputs is None
    # Foreign frames/explicitly retained arguments are intentionally untouched.
    if site in ("dense", "domain"):
        assert held
    trace = caught.tb
    while trace is not None:
        if trace.tb_frame.f_code.co_name == "prepare":
            local = trace.tb_frame.f_locals
            for name in ("source", "dense", "output", "assets", "preceding", "keys", "domain", "frame", "value"):
                assert local.get(name) is None
            assert "scoped" not in local and "output_domain" not in local
        trace = trace.tb_next


def tree(root):
    result = {}
    for path in root.rglob("*.parquet"):
        d = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                d.update(block)
        result[path.relative_to(root)] = (d.hexdigest(), path.stat().st_mtime_ns)
    return result


@pytest.mark.parametrize("revision", [None, 620])
def test_append_repair_exact_receipts_read_proofs_and_old_bytes(tmp_path, monkeypatch, revision):
    before = source(638)
    current = source(640, revision=revision, extra=True)
    outputs, scopes = [], []
    for name, disabled in (("original", True), ("view", False)):
        graph, store, root = global_graph(tmp_path / name, pct_change(before, periods=240))
        with original_path(monkeypatch):
            saved = finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max(),
                input_proof=indexed_source_proof(before)), limits=LIMITS)
        prior_id = saved["results"]["main"][root]
        prior_manifest, old_files = store.manifest(prior_id), tree(tmp_path / name)
        calls = []
        def read(_context, _source, start, end):
            calls.append(("read", start, end))
            return Node.from_domain(current.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
                current.domain.window(start, end), identity=current.identity,
                trace_columns=current.trace_columns, trace_identity=current.trace_identity)
        plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
            input_window_loader=read, input_proof=indexed_source_proof(current, calls))
        with original_path(monkeypatch) if disabled else context_noop():
            saved = finish(plan, limits=LIMITS)
        outputs.append((saved["results"], bits(store.read(saved["results"]["main"][root]).panel.collect(
            dense=False, include_traces=True)), plan.resource_usage))
        scopes.append(calls)
        assert plan.resource_usage["main"][root]["computed_blocks"] == 1
        assert plan.resource_usage["main"][root]["reused_blocks"] == 19
        assert store.manifest(prior_id) == prior_manifest
        after = tree(tmp_path / name)
        assert all(after[path] == item for path, item in old_files.items())
        def forbidden(*_a, **_k):
            pytest.fail("completed same-boundary work consumed values or prepared a view")
        with monkeypatch.context() as repeat:
            repeat.setattr(incremental, "_PctKernelView", forbidden)
            repeat.setattr(incremental.ExecutionRuntime, "run", forbidden)
            repeat.setattr(store, "read_values", forbidden)
            repeated = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
                input_loader=forbidden, input_window_loader=forbidden, input_proof=forbidden)
            assert repeated.complete and finish(repeated)["results"] == saved["results"]
    assert outputs[0] == outputs[1] and scopes[0] == scopes[1]
    reference, store, root = global_graph(tmp_path / "fresh", pct_change(current, periods=240))
    with original_path(monkeypatch):
        fresh = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
            input_proof=indexed_source_proof(current)), limits=LIMITS)
    assert fresh["results"] == outputs[1][0]
    assert bits(store.read(fresh["results"]["main"][root]).panel.collect(dense=False, include_traces=True)) == outputs[1][1]


def test_scoped_fixture_proof_retains_values_coordinates_and_both_date_traces():
    before, appended, repaired = source(638), source(640, extra=True), source(640, revision=620, extra=True)
    start, early_end = before.domain.times[0], before.domain.times[31]
    first = indexed_source_proof(before)("main", "x", start, early_end)
    assert first == indexed_source_proof(appended)("main", "x", start, early_end)
    assert first == indexed_source_proof(repaired)("main", "x", start, early_end)
    late = appended.domain.times[608]
    end = appended.domain.times[637]
    assert indexed_source_proof(appended)("main", "x", late, end) != indexed_source_proof(repaired)("main", "x", late, end)
    for trace in appended.trace_columns:
        frame = appended.collect(dense=False, include_traces=True).with_columns(
            pl.when(pl.col("time") == appended.domain.times[620])
            .then(pl.col(trace) + timedelta(days=2)).otherwise(pl.col(trace)).alias(trace))
        changed = Node.from_domain(frame, appended.domain, source_key="x", trace_columns=appended.trace_columns)
        assert indexed_source_proof(appended)("main", "x", late, end) != indexed_source_proof(changed)("main", "x", late, end)
    keys = appended.domain.grid_lazy().filter(~((pl.col("asset_id") == "D")
        & (pl.col("time") == appended.domain.times[620]))).collect()
    membership = Domain(calendar=appended.domain.times, universe=keys.with_columns(pl.lit(True).alias("active")))
    changed = Node.from_domain(appended.lazy(include_traces=True).collect(), membership,
        source_key="x", trace_columns=appended.trace_columns)
    assert indexed_source_proof(appended)("main", "x", late, end) != indexed_source_proof(changed)("main", "x", late, end)


@pytest.mark.parametrize("proof_kind", ["unknown", "foreign"])
def test_unknown_or_changed_entire_proof_family_still_blocks_before_kernel(tmp_path, monkeypatch, proof_kind):
    before, current = source(66), source(68)
    graph, _store, _root = global_graph(tmp_path, pct_change(before, periods=240))
    finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max(),
        input_proof=indexed_source_proof(before)), limits=LIMITS)
    def changed(_context, _source, start, end):
        if proof_kind == "unknown":
            return None
        return "foreign-family:" + current.interval_identity(start, end)
    def forbidden(*_a, **_k):
        pytest.fail("unknown/full-history proof change reached values or kernel")
    monkeypatch.setattr(incremental, "_PctKernelView", forbidden)
    monkeypatch.setattr(incremental.ExecutionRuntime, "run", forbidden)
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
        input_loader=forbidden, input_window_loader=forbidden, input_proof=changed)
    with pytest.raises(ValueError, match="interval evidence|Input proof changes"):
        finish(plan, limits=LIMITS)


@contextmanager
def context_noop():
    yield


@pytest.mark.parametrize("site", ["kernel", "publish"])
def test_public_failure_closes_view_before_limits_restore(tmp_path, monkeypatch, site):
    x, views = source(66), []
    graph, store, _ = global_graph(tmp_path, pct_change(x, periods=240))
    factory = incremental._PctKernelView
    def capture(*args):
        scope = factory(*args)
        views.append(scope)
        return scope
    monkeypatch.setattr(incremental, "_PctKernelView", capture)
    failure = RuntimeError(f"failure at {site}")
    def broken(*args, **kwargs):
        assert active_resource_limits() == views[-1].limits
        raise failure
    if site == "kernel":
        monkeypatch.setattr(incremental.ExecutionRuntime, "run", broken)
    else:
        monkeypatch.setattr(store, "publish", broken)
    with resource_limits(LIMITS):
        with pytest.raises(RuntimeError) as caught:
            finish(graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()), limits=LIMITS)
        assert active_resource_limits() is LIMITS
    assert caught.value is failure and views and all(v.closed and v._inputs is None for v in views)
    gc.collect()  # Do not assert foreign traceback buffers have been destroyed.


def test_interrupted_publication_retry_reuses_original_block_without_view_work(tmp_path, monkeypatch):
    x = source(98)
    graph, store, root = global_graph(tmp_path / "retry", pct_change(x, periods=240))
    publish = store.publish
    failure = RuntimeError("interrupted after first durable block")
    def interrupt(value, **kwargs):
        publish(value, **kwargs)
        if value.panel.domain.times.max() == x.domain.times[31]:
            raise failure
    with monkeypatch.context() as broken:
        broken.setattr(store, "publish", interrupt)
        with pytest.raises(RuntimeError) as caught:
            finish(graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()), limits=LIMITS)
        assert caught.value is failure
    originals, starts = tree(tmp_path / "retry"), []
    prepare = pct_view._PctKernelView.prepare
    def observe(scope, scoped, output, first, end, history, check):
        starts.append(first)
        return prepare(scope, scoped, output, first, end, history, check)
    monkeypatch.setattr(pct_view._PctKernelView, "prepare", observe)
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max())
    saved = finish(plan, limits=LIMITS)
    assert starts and min(starts) == x.domain.times[32]
    assert plan.resource_usage["main"][root]["reused_blocks"] == 1
    assert plan.resource_usage["main"][root]["computed_blocks"] == 3
    after = tree(tmp_path / "retry")
    assert all(after[path] == item for path, item in originals.items())
    reference, old, old_root = global_graph(tmp_path / "reference", pct_change(x, periods=240))
    with original_path(monkeypatch):
        expected = finish(reference.plan_update({"main": {"x": x}}, through=x.domain.times.max()), limits=LIMITS)
    assert saved["results"] == expected["results"]
    assert bits(store.read(saved["results"]["main"][root]).panel.collect(dense=False, include_traces=True)) == (
        bits(old.read(expected["results"]["main"][old_root]).panel.collect(dense=False, include_traces=True)))


def test_closed_view_drops_own_nodes_before_resource_restore():
    x = source()
    output = x.domain.window(x.domain.times[608], x.domain.times[-1])
    scope = pct_view._PctKernelView(logical(), {"parent": x}, output, 240, LIMITS)
    with resource_limits(scope.limits), scope:
        values = scope.prepare({"parent": x}, output, x.domain.times[608], x.domain.times[-1], 240, lambda: None)
        ref = weakref.ref(values["parent"])
        del values
    gc.collect()
    assert ref() is None and scope._inputs is None


def test_ten_node_6488_session_chain_all_block_ids_and_bits(tmp_path, monkeypatch):
    x = source(6488, extra=True)
    y = Node.from_domain(x.lazy(include_traces=True).with_columns((pl.col("value") * 3.0).alias("value")),
        x.domain, identity="numerator.v1", source_key="y", trace_columns=x.trace_columns,
        trace_identity=x.trace_identity)
    members = Node.from_domain(x.domain.grid_lazy().with_columns(pl.lit(1.0).alias("value")),
        x.domain, identity="members.v1", source_key="members")
    price = project_domain(x, membership=members)
    numerator = project_domain(y, membership=members)
    ratio = div(numerator, price)
    change = pct_change(price, periods=240)
    scale = constant(change, value=100.0)
    denominator = mul(scale, change)
    combined = div(ratio, denominator)
    ranked = rank(combined)
    smoothed = smooth(ranked)
    expression = canonicalize_values(smoothed, significant_digits=14)
    results = []
    for name, disabled in (("original", True), ("view", False)):
        graph, store, _ = global_graph(tmp_path / name, expression)
        plan = graph.plan_update({"main": {"x": x, "y": y, "members": members}}, through=x.domain.times.max())
        with original_path(monkeypatch) if disabled else context_noop():
            saved = finish(plan, limits=LIMITS)
        assert len(saved["results"]["main"]) == 10
        per_node = {}
        for node_id, identity in saved["results"]["main"].items():
            usage = plan.resource_usage["main"][node_id]
            assert usage["computed_blocks"] == 203 and usage["computed_output_sessions"] == 6488
            manifest = store.manifest(identity)
            assert manifest is not None and len(manifest["value_sources"]) == 203
            blocks = [item["identity"] for item in manifest["value_sources"]]
            per_node[node_id] = (identity, blocks, [bits(store.read(block).panel.collect(
                dense=False, include_traces=True)) for block in blocks])
        results.append((saved["results"], per_node, plan.resource_usage))
    assert results[0] == results[1]
