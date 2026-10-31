"""Cold batches keep canonical receipts, while actual numerical work changes.

These are public Graph/CoreStore fixtures. The oracle disables only the new
private admission and uses the same original operators/read/publish pipeline.
"""
from contextlib import contextmanager
from collections import Counter
from dataclasses import replace
import gc
import hashlib
from threading import local
import weakref

import polars as pl
import pytest

from bagelquant_core import CoreStore, Domain, ExecutionRuntime, Node
import bagelquant_core._finite_cold_batch as cold_module
import bagelquant_core.incremental as incremental
from bagelquant_core.operator import (
    add, canonicalize_values, constant, div, mul, negate, pct_change,
    project_domain, rank, rolling_mean, smooth,
)
from bagelquant_core.resources import ResourceLimits, active_resource_limits
from test_finite_pct_view import bits, indexed_source_proof, source
from test_global_graph import finish, global_graph


LIMITS = ResourceLimits(total_threads=1, parallel_nodes=1, memory_target_mib=512,
    cache_mib=16, batch_rows=100_000)
NATIVE = pytest.mark.skipif(pl.__version__ != "1.42.1", reason="closed cold kernel whitelist")


@contextmanager
def canonical(monkeypatch):
    factory = incremental._ColdFiniteBatches
    def disabled(*args, **kwargs):
        controller = factory(*args, **kwargs)
        controller.eligible = False
        return controller
    with monkeypatch.context() as patch:
        patch.setattr(incremental, "_ColdFiniteBatches", disabled)
        yield


def scientific_inventory(store):
    result = {}
    for item in store.inventory():
        manifest = store.manifest(item["identity"])
        result[item["identity"]] = {
            name: manifest.get(name) for name in (
                "key", "value_type", "trace_columns", "trace_identity", "days", "logical_hash",
                "coverage_start", "through", "execution_policy", "content_identity", "value_sources")}
        result[item["identity"]]["partitions"] = [
            {name: part.get(name) for name in ("sha256", "start", "end", "rows")}
            for part in manifest["partitions"]]
    return result


def file_state(store):
    return {str(path.relative_to(store.artifact_path)):
        (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in store.artifact_path.rglob("*") if path.is_file()}


def assert_preserved(store, before):
    after = file_state(store)
    assert {name: after[name] for name in before} == before


def observe_value_reads(monkeypatch):
    """Keep public logical requests separate from the existing reader delegation.

    A plain CoreStore delegates read_values once through read_context, whose
    reader calls that same public method. Both method entries retain the exact
    window/budget; only the inner entry consumes the actual typed body.
    """
    read, nesting, entries, requests = CoreStore.read_values, local(), [], []
    def observed(store, identity, **kwargs):
        depth = getattr(nesting, "depth", 0)
        entry = {"depth": depth, "identity": identity, "start": kwargs["start"],
            "end": kwargs["end"], "limits": active_resource_limits()}
        entries.append(entry)
        if depth == 0:
            requests.append(entry)
        nesting.depth = depth + 1
        try:
            return read(store, identity, **kwargs)
        finally:
            nesting.depth = depth
    monkeypatch.setattr(CoreStore, "read_values", observed)
    return entries, requests


def assert_plain_store_read_delegation(entries, requests):
    # These fixtures create a plain store, not an already-open reader. Assert
    # its exact one-level delegation rather than counting it as extra work.
    assert len(entries) == 2 * len(requests)
    for index, request in enumerate(requests):
        pair = entries[index * 2:index * 2 + 2]
        assert [entry["depth"] for entry in pair] == [0, 1]
        assert all({name: value for name, value in entry.items() if name != "depth"}
            == {name: value for name, value in request.items() if name != "depth"} for entry in pair)


def chain(x, kind="pct"):
    parent = negate(x)
    if kind == "pct":
        return pct_change(parent, periods=240)
    if kind == "rank":
        return rank(parent)
    if kind == "smooth":
        return smooth(parent)
    if kind == "arithmetic":
        return add(parent, negate(parent))
    if kind == "generic":
        return rolling_mean(parent, window=5)
    raise AssertionError(kind)


def ten_nodes(x):
    y = Node.from_domain(x.collect(dense=False, include_traces=True).with_columns(pl.col("value") * 3),
        x.domain, source_key="y", trace_columns=x.trace_columns)
    members = Node.from_domain(x.domain.grid_lazy().collect().with_columns(pl.lit(1.0).alias("value")),
        x.domain, source_key="members")
    price, numerator = project_domain(x, membership=members), project_domain(y, membership=members)
    change = pct_change(price, periods=240)
    expression = canonicalize_values(smooth(rank(div(div(numerator, price),
        mul(constant(change, value=100.0), change)))), significant_digits=14)
    sources = {"x": x, "y": y, "members": members}
    # Omit explicit identities so the genuine eager public constructor creates
    # each content identity and its interval index. Never manufacture a token.
    assert all(value.interval_identity(value.domain.times.min(), value.domain.times.max()) is not None
        for value in sources.values())
    return expression, sources


def execute(tmp_path, x, *, expression=None, sources=None, limits=LIMITS):
    graph, store, root = global_graph(tmp_path, chain(x) if expression is None else expression)
    values = {"x": x} if sources is None else sources
    def proof(context, name, first, last):
        assert context == "main"
        token = values[name].interval_identity(first, last)
        assert token is not None
        return token
    # These tests isolate finite batching; region fusion has its own acceptance.
    # Preserve the same explicit durable boundaries in both comparison arms.
    durable = [node.node_id for node in graph.logical_spec().nodes if node.node_type == "operator"]
    plan = graph.plan_update({"main": values}, through=x.domain.times.max(), input_proof=proof,
        persist_nodes=durable)
    receipt = finish(plan, limits=limits)
    return graph, store, root, plan, receipt


def observe(monkeypatch):
    controllers, calls = [], []
    factory, run = incremental._ColdFiniteBatches, ExecutionRuntime.run
    def controller(*args, **kwargs):
        value = factory(*args, **kwargs)
        controllers.append(value)
        return value
    def runtime(value, graph, **kwargs):
        outputs = set(graph.logical_spec().outputs.values())
        calls.append({node.logical_id: len(node.domain.times) for node in graph.nodes
            if node.logical_id in outputs})
        return run(value, graph, **kwargs)
    monkeypatch.setattr(incremental, "_ColdFiniteBatches", controller)
    monkeypatch.setattr(ExecutionRuntime, "run", runtime)
    return controllers, calls


@NATIVE
@pytest.mark.parametrize("kind", ["pct", "rank", "smooth", "arithmetic"])
@pytest.mark.parametrize("count", [33, 257, 640])
def test_cold_bits_domains_traces_and_every_canonical_receipt_match(tmp_path, monkeypatch, kind, count):
    x = source(count, extra=True)
    controllers, calls = observe(monkeypatch)
    _, store, root, plan, receipt = execute(tmp_path / "new", x, expression=chain(x, kind))
    future_calls = list(calls)
    with canonical(monkeypatch):
        _, old, old_root, baseline, original = execute(tmp_path / "old", x, expression=chain(x, kind))
    assert receipt["results"] == original["results"]
    assert scientific_inventory(store) == scientific_inventory(old)
    for identity in receipt["results"]["main"].values():
        actual, expected = store.read(identity).panel, old.read(identity).panel
        assert actual.domain.signature == expected.domain.signature
        assert actual.trace_identity == expected.trace_identity
        assert bits(actual.collect(dense=False, include_traces=True)) == bits(
            expected.collect(dense=False, include_traces=True))
    assert root == old_root
    current, prior = plan.resource_usage["main"][root], baseline.resource_usage["main"][root]
    assert current["computed_blocks"] == prior["computed_blocks"] == (count + 31) // 32
    assert current["computed_output_sessions"] == prior["computed_output_sessions"] == count
    assert current["canonical_sessions"] == prior["canonical_sessions"] == 32
    assert current["kernel_sessions"] <= prior["kernel_sessions"]
    assert current["input_sessions"] <= prior["input_sessions"]
    assert any(value.schema_reads for value in controllers)
    # The first operator consumes the external input and remains individual32.
    inputs = {node.node_id for node in plan.spec.nodes if node.node_type == "input"}
    external = {node.node_id for node in plan.spec.nodes if node.node_type == "operator"
        and set(node.inputs) <= inputs}
    assert len(external) == 1
    assert all(sum(first in call for call in future_calls) == (count + 31) // 32 for first in external)
    assert all(value._store is None and not value._schemas for value in controllers)


@NATIVE
def test_all_ten_operators_partial_calendar_and_original_partition_checksums(tmp_path, monkeypatch):
    x = source(513, extra=True)
    expression, sources = ten_nodes(x)
    _, store, _, plan, receipt = execute(tmp_path / "new", x, expression=expression, sources=sources)
    with canonical(monkeypatch):
        _, old, _, prior, original = execute(tmp_path / "old", x, expression=expression, sources=sources)
    operator_ids = {node.node_id for node in plan.spec.nodes if node.node_type == "operator"}
    assert len(operator_ids) == 10
    assert set(receipt["results"]["main"]) == set(original["results"]["main"]) == operator_ids
    assert set(plan.resource_usage["main"]) == set(prior.resource_usage["main"]) == operator_ids
    assert receipt["results"] == original["results"]
    assert scientific_inventory(store) == scientific_inventory(old)
    for node_id in operator_ids:
        identity = receipt["results"]["main"][node_id]
        assert bits(store.read(identity).panel.collect(dense=False, include_traces=True)) == bits(
            old.read(identity).panel.collect(dense=False, include_traces=True))
        assert plan.resource_usage["main"][node_id]["computed_blocks"] == 17
        assert plan.resource_usage["main"][node_id]["computed_output_sessions"] == 513


@NATIVE
def test_schema_preflight_is_once_per_parent_under_P_before_bodies_and_runtime(tmp_path, monkeypatch):
    x = source(640)
    events, scopes = [], []
    preflight, read, run = cold_module._ColdFiniteBatches.preflight_schema, CoreStore.read_values, ExecutionRuntime.run
    def schema(controller, canceled):
        assert active_resource_limits().memory_target_mib < controller._limits.memory_target_mib
        before = controller.schema_reads
        result = preflight(controller, canceled)
        events.append(("schema", controller._node_id, controller.schema_reads - before))
        scopes.append(controller)
        return result
    def body(store, identity, **kwargs):
        assert events and events[-1][0] in {"schema", "body", "runtime"}
        events.append(("body", identity, kwargs))
        return read(store, identity, **kwargs)
    def runtime(value, graph, **kwargs):
        events.append(("runtime", graph._single_output().logical_id, None))
        return run(value, graph, **kwargs)
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "preflight_schema", schema)
    monkeypatch.setattr(CoreStore, "read_values", body)
    monkeypatch.setattr(ExecutionRuntime, "run", runtime)
    _, _, root, _, _ = execute(tmp_path, x)
    resolutions = [entry[2] for entry in events if entry[:2] == ("schema", root)]
    assert resolutions[0] == 1 and sum(resolutions) == 1
    assert all(scope._store is None and not scope._schemas for scope in scopes)


@NATIVE
def test_schema_P_is_released_before_unbatched_tail_and_limits_restore(tmp_path, monkeypatch):
    x, controllers, closes = source(257), [], []
    factory, close = incremental._ColdFiniteBatches, cold_module._ColdFiniteBatches.close
    outer = active_resource_limits()
    def capture(*args, **kwargs):
        controller = factory(*args, **kwargs)
        controllers.append(controller)
        return controller
    def observe_close(controller):
        if controller._schemas:
            limits = active_resource_limits()
            assert limits == controller.planning_limits
            closes.append((limits, len(requests)))
            result = close(controller)
            assert not controller._schemas and controller._parents is None and controller._store is None
            assert active_resource_limits() == limits
            return result
        return close(controller)
    monkeypatch.setattr(incremental, "_ColdFiniteBatches", capture)
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "close", observe_close)
    entries, requests = observe_value_reads(monkeypatch)
    execute(tmp_path, x, expression=chain(x, "rank"))
    holders = [controller for controller in controllers if controller.schema_reads]
    assert len(holders) == 1
    assert_plain_store_read_delegation(entries, requests)
    # Preparation retains canonical slabs even when the kernel coalesces them.
    assert [(entry["start"], entry["end"]) for entry in requests] == [
        (x.domain.times[first], x.domain.times[min(first + 31, 256)])
        for first in range(0, 257, 32)]
    assert all(entry["limits"].memory_target_mib <= holders[0].planning_limits.memory_target_mib
        for entry in entries[:-2])
    # Release held schema/parent/store state under P after the eight batched
    # reads, before restoring the ordinary final block's original budget.
    assert closes == [(holders[0].planning_limits, 8)]
    assert len(requests) == 9 and requests[-1]["limits"] == holders[0]._limits
    assert active_resource_limits() == outer


@NATIVE
@pytest.mark.parametrize("kind", ["version", "budget", "rows", "generic", "integer", "missing_index", "custom"])
def test_unsupported_admission_uses_original_before_body_and_no_new_footer(tmp_path, monkeypatch, kind):
    x = source(96)
    if kind == "integer":
        # Integer fixture values must be representable; use the same public
        # Domain/trace constructor without NaN/Inf conversion shortcuts.
        frame = source(96).collect(dense=True, include_traces=True).with_columns(pl.lit(2).cast(pl.Int64).alias("value"))
        x = Node.from_domain(frame, x.domain, source_key="x", trace_columns=x.trace_columns)
    limits, expression = LIMITS, chain(x, "generic" if kind == "generic" else "rank")
    if kind == "version":
        monkeypatch.setattr(pl, "__version__", "1.41.2")
    elif kind == "budget":
        # Leave insufficient scratch after preparation has debited the cache.
        limits = replace(LIMITS, memory_target_mib=4, cache_mib=3)
    elif kind == "rows":
        limits = replace(LIMITS, batch_rows=1)
    elif kind == "missing_index":
        describe = CoreStore.describe
        def missing(store, identity):
            value = describe(store, identity)
            return None if value is None else {**value, "calendar": None}
        monkeypatch.setattr(CoreStore, "describe", missing)
    elif kind == "custom":
        operation = rank.operation
        monkeypatch.setattr(rank, "operation", lambda *args, **kwargs: operation(*args, **kwargs))
    original = cold_module._ColdFiniteBatches.preflight_schema
    resolutions = []
    def preflight(controller, canceled):
        resolutions.append(controller)
        return original(controller, canceled)
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "preflight_schema", preflight)
    _, _, root, plan, _ = execute(tmp_path, x, limits=limits, expression=expression)
    assert plan.resource_usage["main"][root]["computed_blocks"] == 3
    if kind == "integer":
        assert len(resolutions) == 1 and resolutions[0].schema_reads == 1
    else:
        assert resolutions == []


@NATIVE
def test_repeat_noop_has_no_footer_values_runtime_or_publication(tmp_path, monkeypatch):
    x = source(96)
    graph, store, _, _, first = execute(tmp_path, x)
    before = file_state(store)
    def forbidden(*_args, **_kwargs):
        pytest.fail("metadata-only repeat consumed a value or executed")
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "preflight_schema", forbidden)
    monkeypatch.setattr(CoreStore, "read_values", forbidden)
    monkeypatch.setattr(Node, "collect", forbidden)
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    monkeypatch.setattr(CoreStore, "publish", forbidden)
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(), input_proof=indexed_source_proof(x))
    assert plan.complete
    second = plan.publish()
    assert second["results"] == first["results"]
    operators = {node.node_id for node in plan.spec.nodes if node.node_type == "operator"}
    assert len(operators) == 2
    assert set(plan.resource_usage) == {"main"}
    assert set(plan.resource_usage["main"]) == set(second["results"]["main"]) == operators
    counters = {"computed_blocks", "reused_blocks", "input_sessions", "output_sessions"}
    for row in plan.resource_usage["main"].values():
        assert set(row) == counters | {"outcome"} and row["outcome"] == "up_to_date"
        assert all(type(row[name]) is int and row[name] == 0 for name in counters)
    assert file_state(store) == before


@NATIVE
@pytest.mark.parametrize("change", ["append", "repair"])
def test_own_history_vetoes_batches_preserves_all_old_files_and_matches_fresh(tmp_path, monkeypatch, change):
    old_x = source(96)
    graph, store, root, _, _ = execute(tmp_path / "history", old_x)
    before = file_state(store)
    x = source(160) if change == "append" else source(96, revision=70)
    graph.merge(chain(x))
    with monkeypatch.context() as patch:
        patch.setattr(cold_module._ColdFiniteBatches, "preflight_schema",
            lambda *_args, **_kwargs: pytest.fail("history entered new schema preflight"))
        plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(), input_proof=indexed_source_proof(x))
        result = finish(plan, limits=LIMITS)
    _, reference, expected_root, _, expected = execute(tmp_path / "fresh", x)
    assert root == expected_root and result["results"] == expected["results"]
    assert bits(store.read(result["results"]["main"][root]).panel.collect(dense=False, include_traces=True)) == bits(
        reference.read(expected["results"]["main"][root]).panel.collect(dense=False, include_traces=True))
    assert_preserved(store, before)
    assert plan.resource_usage["main"][root]["reused_blocks"] > 0
    assert plan.resource_usage["main"][root]["computed_blocks"] <= 2


@NATIVE
def test_canceled_batch_preserves_actual_durable_block_then_retry_is_original(tmp_path, monkeypatch):
    x = source(96)
    graph, store, root = global_graph(tmp_path, chain(x))
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(), input_proof=indexed_source_proof(x))
    cancel, committed = False, []
    publish = CoreStore.publish
    def durable(owner, value, **kwargs):
        nonlocal cancel
        result = publish(owner, value, **kwargs)
        if value.key.node_id == root and kwargs.get("value_references") is None:
            committed.append(value.key.identity)
            cancel = True
        return result
    def canceled():
        if cancel:
            raise RuntimeError("stop after durable canonical receipt")
    with monkeypatch.context() as patch:
        patch.setattr(CoreStore, "publish", durable)
        with pytest.raises(RuntimeError, match="durable canonical"):
            finish(plan, limits=LIMITS, check_canceled=canceled)
    assert len(committed) == 1
    assert store.describe(committed[0]) is not None
    before = file_state(store)
    with monkeypatch.context() as patch:
        patch.setattr(cold_module._ColdFiniteBatches, "preflight_schema",
            lambda *_args, **_kwargs: pytest.fail("retry must use original bounded path"))
        retry = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(), input_proof=indexed_source_proof(x))
        result = finish(retry, limits=LIMITS)
    assert retry.resource_usage["main"][root]["reused_blocks"] == 1
    assert retry.resource_usage["main"][root]["computed_blocks"] == 2
    assert_preserved(store, before)
    assert store.manifest(result["results"]["main"][root])["value_sources"][0]["identity"] == committed[0]


@NATIVE
@pytest.mark.parametrize("phase", ["footer", "body", "kernel", "publication"])
def test_admitted_failure_propagates_once_without_fallback_and_closes_owned_scope(tmp_path, monkeypatch, phase):
    x, scopes, failures, frames = source(96), [], [], []
    outer = active_resource_limits()
    factory = incremental._ColdFiniteBatches
    def capture(*args, **kwargs):
        controller = factory(*args, **kwargs)
        scopes.append(controller)
        return controller
    monkeypatch.setattr(incremental, "_ColdFiniteBatches", capture)
    def reject(*args, **kwargs):
        try:
            failures.append(active_resource_limits())
            raise MemoryError("admitted failure is not replayable")
        finally:
            del args, kwargs
    if phase == "footer":
        monkeypatch.setattr(cold_module._ColdFiniteBatches, "preflight_schema", reject)
    elif phase == "body":
        validate = cold_module._ColdFiniteBatches.validate_frame
        def reject_frame(controller, parent, frame):
            try:
                frames.append(weakref.ref(frame))
                validate(controller, parent, frame)
                reject()
            finally:
                del frame
        monkeypatch.setattr(cold_module._ColdFiniteBatches, "validate_frame", reject_frame)
    elif phase == "kernel":
        run = ExecutionRuntime.run
        def reject_kernel(runtime, graph, **kwargs):
            if any(scope.eligible and scope._schemas for scope in scopes):
                reject()
            return run(runtime, graph, **kwargs)
        monkeypatch.setattr(ExecutionRuntime, "run", reject_kernel)
    else:
        publish = CoreStore.publish
        def reject_publish(store, value, **kwargs):
            try:
                if any(scope.eligible and scope._schemas for scope in scopes):
                    reject()
                return publish(store, value, **kwargs)
            finally:
                del value
        monkeypatch.setattr(CoreStore, "publish", reject_publish)
    with pytest.raises(MemoryError, match="not replayable") as caught:
        execute(tmp_path, x)
    assert len(failures) == 1
    assert all(scope._store is None and scope._parents is None and not scope._schemas for scope in scopes)
    assert active_resource_limits() == outer
    # Only Core-owned aliases are promised released. Injected callees clear
    # their own arguments; foreign/library tracebacks are never rewritten.
    assert caught.value.__traceback__ is not None
    gc.collect()
    assert all(reference() is None for reference in frames)


@NATIVE
def test_unexpected_new_own_candidate_is_not_admitted_as_history(tmp_path, monkeypatch):
    x = source(96)
    guard = cold_module._ColdFiniteBatches.guard
    entered = []
    def conflict(controller):
        if controller.eligible and controller._schemas:
            entered.append(True)
            raise ValueError("competing cold frontier; replan")
        return guard(controller)
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "guard", conflict)
    with pytest.raises(ValueError, match="competing cold frontier"):
        execute(tmp_path, x)
    assert entered == [True]


@NATIVE
@pytest.mark.parametrize("cache", [0, 48])
def test_dense_rectangle_and_already_divided_limits_bound_every_batch(tmp_path, monkeypatch, cache):
    x, observed = source(640), []
    plan_batch = cold_module._ColdFiniteBatches.plan
    def capture(controller, *args):
        result = plan_batch(controller, *args)
        if result is not None:
            observed.append(result)
            assert result.limits.cache_mib == controller._limits.cache_mib
            assert result.limits.total_threads == controller._limits.total_threads
            assert result.limits.memory_target_mib == controller._limits.memory_target_mib - result.reserve_mib
            assert result.reserve_mib <= (controller._limits.memory_target_mib - controller._limits.cache_mib) // 4
            assert result.reserve_mib == (result.raw_bytes + 1024**2 - 1) // 1024**2
            assert 1 < (result.stop - args[0] + 31) // 32 <= 8
        return result
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "plan", capture)
    divided = ResourceLimits(total_threads=6, parallel_nodes=3, memory_target_mib=1536,
        cache_mib=cache, batch_rows=100_000).for_workers(3)
    execute(tmp_path, x, limits=divided)
    assert observed


@NATIVE
def test_original_per32_proof_scopes_are_not_replaced_by_batch_union(tmp_path, monkeypatch):
    x = source(257)
    interval = CoreStore.interval_identity
    calls = []
    def proof(store, identity, **kwargs):
        calls.append((identity, str(kwargs["start"]), str(kwargs["end"])))
        return interval(store, identity, **kwargs)
    monkeypatch.setattr(CoreStore, "interval_identity", proof)
    execute(tmp_path / "new", x)
    future = Counter(calls)
    calls.clear()
    with canonical(monkeypatch):
        execute(tmp_path / "old", x)
    assert future == Counter(calls)
    # Each canonical endpoint remains present; no union-only proof namespace.
    assert {str(x.domain.times[index]) for index in (31, 63, 95, 127, 159, 191, 223, 255, 256)} <= {
        last for _, _, last in calls}


@NATIVE
def test_external_window_loader_keeps_original32_scope_and_source_authority(tmp_path):
    x, reads = source(257), []
    graph, _, root = global_graph(tmp_path, chain(x, "rank"))
    def read(context, name, first, last):
        assert context == "main" and name == "x"
        reads.append((first, last))
        return Node.from_domain(x.lazy(include_traces=True).filter(pl.col("time").is_between(first, last)),
            x.domain.window(first, last), identity=x.identity, source_key="x",
            trace_columns=x.trace_columns, trace_identity=x.trace_identity)
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(),
        input_window_loader=read, input_proof=indexed_source_proof(x))
    finish(plan, limits=LIMITS)
    assert reads == [(x.domain.times[start], x.domain.times[min(start + 31, 256)])
        for start in range(0, 257, 32)]
    assert plan.resource_usage["main"][root]["computed_blocks"] == 9


@NATIVE
@pytest.mark.parametrize("count", [31, 32])
def test_single_canonical_block_never_resolves_new_footer(tmp_path, monkeypatch, count):
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "preflight_schema",
        lambda *_args, **_kwargs: pytest.fail("one block cannot be coalesced"))
    _, _, root, plan, _ = execute(tmp_path, source(count))
    assert plan.resource_usage["main"][root]["computed_blocks"] == 1


@NATIVE
@pytest.mark.parametrize("layout", ["static", "empty"])
def test_ineligible_domain_has_no_new_footer_and_retains_original_execution(tmp_path, monkeypatch, layout):
    original = source(96)
    frame = original.collect(dense=False, include_traces=True)
    universe = original.domain.asset_ids
    if layout == "empty":
        frame = frame.clear()
        universe = original.domain.grid_lazy().collect().clear().with_columns(pl.lit(True).alias("active"))
    x = Node.from_domain(frame, Domain(calendar=original.domain.times, universe=universe),
        source_key="x", trace_columns=original.trace_columns)
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "preflight_schema",
        lambda *_args, **_kwargs: pytest.fail("ineligible Domain resolved a footer"))
    _, _, root, plan, _ = execute(tmp_path, x, expression=chain(x, "rank"))
    assert plan.resource_usage["main"][root]["computed_blocks"] == 3


@NATIVE
def test_long_utf8_keys_increase_conservative_charge_without_identity_shortcuts(tmp_path, monkeypatch):
    original, charges = source(96), []
    frame = original.collect(dense=False, include_traces=True).with_columns(
        (pl.lit("键" * 256) + pl.col("asset_id")).alias("asset_id"))
    membership = original.domain.grid_lazy().collect().with_columns(
        (pl.lit("键" * 256) + pl.col("asset_id")).alias("asset_id"), pl.lit(True).alias("active"))
    long = Node.from_domain(frame, Domain(calendar=original.domain.times, universe=membership),
        source_key="x", trace_columns=original.trace_columns)
    planner = cold_module._ColdFiniteBatches.plan
    def observe_charge(controller, *args):
        batch = planner(controller, *args)
        if batch is not None:
            charges.append((controller.planning_bytes, batch.raw_bytes))
        return batch
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "plan", observe_charge)
    execute(tmp_path / "short", original)
    short = charges[0]
    charges.clear()
    execute(tmp_path / "long", long)
    assert charges and charges[0][0] > short[0] and charges[0][1] > short[1]


@NATIVE
def test_unsupported_schema_preflight_returns_original_before_any_batch_body(tmp_path, monkeypatch):
    x, schemas = source(96), []
    preflight = cold_module._ColdFiniteBatches.preflight_schema
    def unsupported(controller, canceled):
        # Exercise the real preflight and then reject its supported result as
        # an unsupported-schema choice, without swallowing an exception.
        assert preflight(controller, canceled)
        schemas.append(True)
        controller.eligible = False
        controller._schemas.clear()
        return False
    monkeypatch.setattr(cold_module._ColdFiniteBatches, "preflight_schema", unsupported)
    entries, requests = observe_value_reads(monkeypatch)
    _, _, root, plan, _ = execute(tmp_path, x, expression=chain(x, "rank"))
    assert schemas == [True]
    assert_plain_store_read_delegation(entries, requests)
    assert [(entry["start"], entry["end"]) for entry in requests] == [
        (x.domain.times[start], x.domain.times[min(start + 31, 95)])
        for start in range(0, 96, 32)]
    numerical_limits = replace(LIMITS, cache_mib=LIMITS.cache_mib - LIMITS.cache_mib // 4)
    assert all(entry["limits"] == numerical_limits for entry in entries)
    assert plan.resource_usage["main"][root]["computed_blocks"] == 3
