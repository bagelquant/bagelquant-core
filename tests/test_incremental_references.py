from dataclasses import replace
from datetime import timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Node, NodeMaterialization
from bagelquant_core import Domain
from bagelquant_core import ExecutionRuntime
from bagelquant_core.operator import negate, rolling_mean

from test_global_graph import finish, global_graph, source


@pytest.mark.parametrize("current_authority", ["matching", "missing", "other_parameters", "missing_inputs"])
def test_old_policy_does_not_veto_proven_current_prefix(tmp_path, monkeypatch, current_authority):
    first = source(32)
    graph, store, root = global_graph(tmp_path / "incremental", negate(first))
    prior = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    identity = prior["results"]["main"][root]
    materialization = store.read(identity)
    policy = store.manifest(identity)["execution_policy"]
    legacy_key = replace(materialization.key, context_identity="retained-v2-policy")
    store.publish(NodeMaterialization(legacy_key, materialization.panel),
        execution_policy={**policy, "name": "finite_causal_blocks.v2"})
    candidates = [legacy_key.identity]
    if current_authority in {"matching", "missing_inputs"}:
        candidates.append(identity)
    elif current_authority == "other_parameters":
        unrelated_key = replace(materialization.key, context_identity="unrelated-parameters")
        store.publish(NodeMaterialization(unrelated_key, materialization.panel),
            execution_policy={**policy, "parameter_bindings": {"$other": 1}})
        candidates.append(unrelated_key.identity)
    monkeypatch.setattr(store, "completed_materializations", lambda *_: tuple(candidates))
    current = source(33)
    options = ({"input_proof": lambda *_: None} if current_authority == "missing_inputs" else {})
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(), **options)
    if current_authority != "matching":
        with monkeypatch.context() as patch:
            patch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("unproven prefix ran a kernel"))
            with pytest.raises(ValueError, match="explicit"):
                finish(plan)
        assert store.graph_state("global")["current_update"] == prior["identity"]
        return
    updated = finish(plan)
    assert plan.resource_usage["main"][root]["computed_output_sessions"] == 1
    assert plan.resource_usage["main"][root]["kernel_sessions"] == 1
    before, after = store.manifest(identity), store.manifest(updated["results"]["main"][root])
    assert {part["$file"] for part in before["partitions"]} <= {part["$file"] for part in after["partitions"]}
    assert store.manifest(legacy_key.identity)["execution_policy"]["name"] == "finite_causal_blocks.v2"
    reference, fresh, reference_root = global_graph(tmp_path / "reference", negate(current))
    full = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    assert_frame_equal(store.read(after["identity"]).panel.collect(include_traces=True),
        fresh.read(full["results"]["main"][reference_root]).panel.collect(include_traces=True), check_exact=True)


def test_broadcast_append_has_no_historical_read_or_kernel_and_preserves_traces(tmp_path):
    from bagelquant_core.operator import broadcast_by_time
    from bagelquant_core import causal_history_requirements

    def inputs(count):
        like = source(count, name="like", dynamic=True)
        frame = source(count).lazy(include_traces=True).filter(pl.col("asset_id") == "a")
        index = Node.from_domain(frame.collect(), like.domain,
            source_key="index", name="index", trace_columns=("available_date",))
        return {"index": index, "like": like}

    previous_inputs = inputs(31)
    expression = broadcast_by_time(previous_inputs["index"], previous_inputs["like"])
    graph, store, root = global_graph(tmp_path / "incremental", expression)
    assert causal_history_requirements(expression.logical_spec())[root] == 0
    previous = finish(graph.plan_update({"main": previous_inputs},
        through=previous_inputs["like"].domain.times.max()))
    current = inputs(32)
    reads = []
    def window(_context, key, first, last):
        reads.append((key, first, last))
        value = current[key]
        return Node.from_domain(value.lazy(include_traces=True).filter(pl.col("time").is_between(first, last)),
            value.domain.window(first, last), identity=value.identity,
            trace_columns=value.trace_columns, trace_identity=value.trace_identity)
    day = current["like"].domain.times.max()
    plan = graph.plan_update({"main": current}, through=day, input_window_loader=window,
        input_loader=lambda *_: pytest.fail("broadcast append read the historical source"))
    updated = finish(plan)
    assert sorted(reads) == [("index", day, day), ("like", day, day)]
    assert plan.resource_usage["main"][root]["computed_output_sessions"] == 1
    assert plan.resource_usage["main"][root]["kernel_sessions"] == 1
    reference, fresh, reference_root = global_graph(tmp_path / "reference",
        broadcast_by_time(current["index"], current["like"]))
    full = finish(reference.plan_update({"main": current}, through=day))
    assert_frame_equal(store.read(updated["results"]["main"][root]).panel.collect(include_traces=True),
        fresh.read(full["results"]["main"][reference_root]).panel.collect(include_traces=True))
    original = store.manifest(previous["results"]["main"][root])
    combined = store.manifest(updated["results"]["main"][root])
    assert {part["$file"] for part in original["partitions"]} <= {part["$file"] for part in combined["partitions"]}


@pytest.mark.parametrize("completed", [31, 32, 63, 64])
def test_pointwise_append_reads_and_writes_only_new_output(tmp_path, completed):
    first = source(completed)
    graph, store, root = global_graph(tmp_path / "incremental", negate(first))
    previous = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    old = store.manifest(previous["results"]["main"][root])
    old_files = {part["$file"]: store._file(part["$file"]).read_bytes() for part in old["partitions"]}
    appended = source(completed + 1)
    reads = []

    def window(_context, key, start, end):
        assert key == "x"
        reads.append((start, end))
        own = appended.domain.window(start, end)
        return Node.from_domain(appended.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
            own, identity=appended.identity, value_type=appended.value_type,
            trace_columns=appended.trace_columns, trace_identity=appended.trace_identity)

    def whole(*_args):
        pytest.fail("append decoded the complete source")

    plan = graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max(),
        input_loader=whole, input_window_loader=window)
    updated = finish(plan)
    day = first.domain.times.max() + timedelta(days=1)
    assert reads == [(day, day)]
    assert plan.resource_usage["main"][root]["computed_output_sessions"] == 1
    assert plan.resource_usage["main"][root]["kernel_sessions"] == 1
    manifest = store.manifest(updated["results"]["main"][root])
    assert all(store._file(path).read_bytes() == contents for path, contents in old_files.items())
    assert set(old_files) <= {part["$file"] for part in manifest["partitions"]}
    added = [part for part in manifest["partitions"] if part["$file"] not in old_files]
    assert len(added) == 1
    assert added[0]["start"] == added[0]["end"] == str(day)
    assert added[0]["rows"] == 2
    reference, fresh, reference_root = global_graph(tmp_path / "reference", negate(appended))
    full = finish(reference.plan_update({"main": {"x": appended}}, through=day))
    assert_frame_equal(store.read(manifest["identity"]).panel.collect(include_traces=True),
        fresh.read(full["results"]["main"][reference_root]).panel.collect(include_traces=True), check_exact=True)
    store.verify_update(updated["identity"])


def test_rolling_append_preserves_original_partitions_and_exact_canonical_values(tmp_path):
    first = source(63, dynamic=True)
    graph, store, root = global_graph(tmp_path / "incremental", rolling_mean(first, window=4))
    previous = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    old = store.manifest(previous["results"]["main"][root])
    appended = source(64, dynamic=True)
    plan = graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max())
    updated = finish(plan)
    manifest = store.manifest(updated["results"]["main"][root])
    assert set(part["$file"] for part in old["partitions"]) <= set(part["$file"] for part in manifest["partitions"])
    assert plan.resource_usage["main"][root]["computed_output_sessions"] == 1
    assert plan.resource_usage["main"][root]["kernel_sessions"] == 35
    reference, fresh, reference_root = global_graph(tmp_path / "reference", rolling_mean(appended, window=4))
    full = finish(reference.plan_update({"main": {"x": appended}}, through=appended.domain.times.max()))
    assert_frame_equal(store.read(manifest["identity"]).panel.collect(include_traces=True),
        fresh.read(full["results"]["main"][reference_root]).panel.collect(include_traces=True), check_exact=True)


def test_reference_publication_rejects_missing_overlapping_and_wrong_membership(tmp_path):
    x = source(3)
    graph, store, root = global_graph(tmp_path, negate(x))
    update = finish(graph.plan_update({"main": {"x": x}}, through=x.domain.times.max()))
    record = store.read(update["results"]["main"][root])
    for refs in (("missing",), (record.key.identity, record.key.identity)):
        with pytest.raises(ValueError, match="registered|exactly once"):
            store.publish(record, value_references=refs)
    wrong = Node.from_domain(record.panel.lazy(include_traces=True), source(3, dynamic=True).domain,
        value_type=record.panel.value_type, trace_columns=record.panel.trace_columns)
    from dataclasses import replace
    key = replace(record.key, domain_identity=wrong.domain.signature)
    with pytest.raises(ValueError, match="membership differs"):
        store.publish(NodeMaterialization(key, wrong), value_references=(record.key.identity,))


def test_window_loader_rejects_forged_domain_signature(tmp_path):
    first = source(31)
    graph, _store, _root = global_graph(tmp_path, negate(first))
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    appended = source(32)

    def window(_context, _key, start, end):
        expected = appended.domain.window(start, end)
        forged = Domain(calendar=expected.times, universe=["a"], identity=expected.signature)
        frame = appended.lazy(include_traces=True).filter(pl.col("time").is_between(start, end) & (pl.col("asset_id") == "a"))
        return Node.from_domain(frame, forged, identity=appended.identity, trace_columns=appended.trace_columns,
            trace_identity=appended.trace_identity)

    plan = graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max(), input_window_loader=window)
    with pytest.raises(ValueError, match="bounded input differs"):
        finish(plan)


def opaque_source(count):
    value = source(count)
    return Node.from_domain(value.lazy(include_traces=True), value.domain, source_key="x", name="x",
        identity=f"frozen:{count}", trace_columns=value.trace_columns, trace_identity=value.trace_identity)


def test_completed_unknown_interval_blocks_without_kernel_execution(tmp_path, monkeypatch):
    first = opaque_source(63)
    graph, store, _root = global_graph(tmp_path, negate(first))
    completed = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    appended = opaque_source(64)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("unknown proof replayed history"))
    with pytest.raises(ValueError, match="explicit owner index maintenance"):
        finish(graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max()))
    assert store.graph_state("global")["current_update"] == completed["identity"]


def test_checkpoint_missing_prefix_proof_blocks_before_historical_kernel(tmp_path, monkeypatch):
    from bagelquant_core.operator import ewm_mean
    first = opaque_source(35)
    graph, store, _root = global_graph(tmp_path, ewm_mean(first, alpha=0.4))
    completed = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    appended = opaque_source(36)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("unknown checkpoint replayed history"))
    with pytest.raises(ValueError, match="proven preceding state"):
        finish(graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max()))
    assert store.graph_state("global")["current_update"] == completed["identity"]


def test_new_input_proof_namespace_does_not_replay_completed_finite_history(tmp_path, monkeypatch):
    first = opaque_source(63)
    graph, store, _root = global_graph(tmp_path, negate(first))
    def proof(namespace):
        return lambda _context, _key, start, end: f"{namespace}:{start}:{end}"
    previous = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max(),
        input_proof=proof("provenance")))
    current = opaque_source(64)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_args, **_kwargs: pytest.fail("proof format replayed old kernels"))
    with pytest.raises(ValueError, match="proof changes cover the completed finite history"):
        finish(graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
            input_proof=proof("scalar_projection")))
    assert store.graph_state("global")["current_update"] == previous["identity"]


def test_unknown_initialization_retry_keeps_completed_units(tmp_path, monkeypatch):
    value = opaque_source(64)
    graph, store, _root = global_graph(tmp_path, negate(value))
    original_run = ExecutionRuntime.run
    calls = []

    def stop(runtime, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted initialization")
        return original_run(runtime, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", stop)
        with pytest.raises(RuntimeError, match="interrupted"):
            finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    assert store.graph_state("global")["current_update"] is None
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    finished = finish(plan)
    assert next(iter(plan.resource_usage["main"].values()))["reused_blocks"] == 1
    store.verify_update(finished["identity"])


def test_completed_other_context_does_not_block_new_initialization(tmp_path):
    value = opaque_source(31)
    graph, store, root = global_graph(tmp_path, negate(value))
    finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    graph.register_context("new", anchor="2024-01-01")
    appended = opaque_source(32)
    plan = graph.plan_update({"new": {"x": appended}}, through=appended.domain.times.max(), roots=[root])
    update = finish(plan)
    store.verify_update(update["identity"])


def test_missing_domain_index_cannot_authorize_coordinate_collision(tmp_path):
    from dataclasses import replace
    value = source(3)
    graph, store, root = global_graph(tmp_path, negate(value))
    update = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    old = store.read(update["results"]["main"][root])
    with store.transaction() as connection:
        connection.execute("DELETE FROM computation_index")
    domain = Domain(calendar=old.panel.domain.times, universe=["a"], identity=old.panel.domain.signature)
    wrong = Node.from_domain(old.panel.lazy(include_traces=True).filter(pl.col("asset_id") == "a"), domain,
        trace_columns=old.panel.trace_columns)
    record = NodeMaterialization(replace(old.key, context_identity="new-context"), wrong)
    with pytest.raises(ValueError, match="Domain identity needs explicit"):
        store.publish(record)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_unused_context_parameter_preserves_completed_numerical_work(tmp_path, checkpoint):
    from bagelquant_core.operator import ewm_mean
    first = source(35 if checkpoint else 63)
    expression = ewm_mean(first, alpha=0.4) if checkpoint else negate(first)
    graph, store, root = global_graph(tmp_path / "incremental", expression)
    graph.register_context("main", anchor="2024-01-01", parameters={"unused": 1})
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    graph.register_context("main", anchor="2024-01-01", parameters={"unused": 2})
    appended = source(len(first.domain.times) + 1)
    plan = graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max())
    updated = finish(plan)
    usage = plan.resource_usage["main"][root]
    assert usage["replayed_sessions" if checkpoint else "computed_output_sessions"] == 1
    if checkpoint:
        assert usage["checkpoint_through"] == str(first.domain.times.max())
    else:
        assert usage["kernel_sessions"] == 1
    reference, fresh, reference_root = global_graph(tmp_path / "reference",
        ewm_mean(appended, alpha=0.4) if checkpoint else negate(appended))
    full = finish(reference.plan_update({"main": {"x": appended}}, through=appended.domain.times.max()))
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True),
        fresh.read_values(full["results"]["main"][reference_root], include_traces=True), check_exact=True)


def test_checkpoint_same_boundary_new_envelope_reads_only_proof_records(tmp_path, monkeypatch):
    from bagelquant_core.operator import ewm_mean
    first = opaque_source(35)
    graph, store, root = global_graph(tmp_path, ewm_mean(first, alpha=0.4))
    def proof(_context, _input, _start, _end):
        return "unchanged-complete-prefix"
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max(), input_proof=proof))
    changed = Node.from_domain(first.lazy(include_traces=True), first.domain, source_key="x",
        identity="new-enclosing-snapshot", trace_columns=first.trace_columns, trace_identity=first.trace_identity)
    def forbidden(*_args, **_kwargs):
        pytest.fail("same-boundary continuation loaded values or executed a kernel")
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    plan = graph.plan_update({"main": {"x": changed}}, through=changed.domain.times.max(),
        input_proof=proof, input_loader=forbidden)
    updated = finish(plan)
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 0
    assert store.manifest(updated["results"]["main"][root])["partitions"]


def test_checkpoint_append_reads_only_new_source_and_references_original_values(tmp_path):
    from bagelquant_core.operator import ewm_mean
    def traced(count):
        value = source(count)
        frame = value.collect(include_traces=True).with_columns(
            pl.when(pl.col("time") == value.domain.times[10]).then(pl.lit(value.domain.times[10] + timedelta(days=100)))
            .otherwise(pl.col("available_date")).alias("available_date"))
        return Node.from_domain(frame, value.domain, source_key="x", trace_columns=value.trace_columns)
    first = traced(35)
    graph, store, root = global_graph(tmp_path / "incremental", ewm_mean(first, alpha=0.4))
    previous = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    old = store.manifest(previous["results"]["main"][root])
    old_files = {part["$file"]: store._file(part["$file"]).read_bytes() for part in old["partitions"]}
    appended = traced(36)
    reads = []
    def window(_context, _key, start, end):
        reads.append((start, end))
        return Node.from_domain(appended.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
            appended.domain.window(start, end), identity=appended.identity, trace_columns=appended.trace_columns,
            trace_identity=appended.trace_identity)
    def forbidden(*_args, **_kwargs):
        pytest.fail("checkpoint append read the complete source")
    plan = graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max(),
        input_loader=forbidden, input_window_loader=window)
    updated = finish(plan)
    day = appended.domain.times.max()
    assert reads == [(day, day)]
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 1
    manifest = store.manifest(updated["results"]["main"][root])
    assert set(old_files) <= {part["$file"] for part in manifest["partitions"]}
    assert all(store._file(path).read_bytes() == contents for path, contents in old_files.items())
    reference, fresh, reference_root = global_graph(tmp_path / "reference", ewm_mean(appended, alpha=0.4))
    full = finish(reference.plan_update({"main": {"x": appended}}, through=day))
    assert_frame_equal(store.read_values(manifest["identity"], include_traces=True),
        fresh.read_values(full["results"]["main"][reference_root], include_traces=True), check_exact=True)


def test_malformed_ewm_asset_state_uses_valid_preceding_checkpoint(tmp_path, monkeypatch):
    from dataclasses import replace
    from bagelquant_core.operator import ewm_mean
    first = source(5)
    graph, store, root = global_graph(tmp_path, ewm_mean(first, alpha=0.4))
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    second = source(35)
    completed = finish(graph.plan_update({"main": {"x": second}}, through=second.domain.times.max()))
    identity = completed["results"]["main"][root]
    read = store.read
    def damaged(key, **kwargs):
        result = read(key, **kwargs)
        if key == identity:
            return replace(result, checkpoint={**result.checkpoint, "state": {"through": str(second.domain.times.max()), "assets": None}})
        return result
    monkeypatch.setattr(store, "read", damaged)
    appended = source(36)
    plan = graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max())
    finish(plan)
    assert plan.resource_usage["main"][root]["checkpoint_through"] == str(first.domain.times.max())
