"""Original-bound certification supplies state without historical calculation."""
from dataclasses import replace
from copy import deepcopy

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from bagelquant_core import ExecutionRuntime, MaterializationKey, Node
from bagelquant_core.operation_contract import ExecutionMode
from bagelquant_core.operator import negate, streak_count
from bagelquant_core.resources import ResourceLimits, resource_limits
from bagelquant_core.streak_state import valid_streak_state
from test_global_graph import global_graph, finish
from test_streak_checkpoint import source, expected


def legacy(root, monkeypatch, *, traces=False, dynamic=False):
    value = source(36, traces=traces, dynamic=dynamic)
    graph, store, result = global_graph(root, streak_count(negate(value)))
    with monkeypatch.context() as old:
        old.setattr(streak_count, "contract", replace(streak_count.contract,
            execution=ExecutionMode.LAZY, checkpoint_replay=False, checkpoint_history=None, checkpoint_validator=None))
        update = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    return value, graph, store, result, update


def certify(graph, store, result, update):
    return store.plan_input_proof_rebind(graph._state()["spec"], update["results"]["main"][result],
        mode="streak_checkpoint.v1", original_update=update["identity"], context_id="main",
        original_context={"anchor": "2024-01-01", "parameters": {}},
        report_id="streak-maintenance-test", report_digest="c" * 64)


@pytest.mark.parametrize("traces", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_certification_preserves_originals_and_only_new_streak_sessions_run(tmp_path, monkeypatch, traces, dynamic):
    before, graph, store, result, old = legacy(tmp_path, monkeypatch, traces=traces, dynamic=dynamic)
    original_id = old["results"]["main"][result]
    original = store.manifest(original_id)
    original_files = {part["$file"]: (store.artifact_path / part["$file"]).read_bytes()
        for part in original["partitions"]}
    reads = []
    read = type(store).read_values
    def observed(owner, identity, **bounds):
        assert bounds.get("start") is not None and bounds.get("end") is not None
        assert (bounds["end"] - bounds["start"]).days < 32
        reads.append((identity, bounds["start"], bounds["end"]))
        return read(owner, identity, **bounds)
    with monkeypatch.context() as guard:
        guard.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("certification ran a numerical kernel"))
        guard.setattr(type(store), "read_values", observed)
        plan = certify(graph, store, result, old)
        assert reads
        guard.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("certification apply consumed values"))
        applied = store.apply_input_proof_rebind(plan)
        repeated = store.apply_input_proof_rebind(plan)
        assert repeated["outcome"] == "up_to_date"
        assert applied["computed_sessions"] == 0
        checkpoint = store.read_checkpoint(applied["identity"])
        assert valid_streak_state(checkpoint["state"])
    assert graph._state()["current_update"] == old["identity"]
    assert store.manifest(original_id) == original
    for name, content in original_files.items():
        assert (store.artifact_path / name).read_bytes() == content
    current = source(37, traces=traces, dynamic=dynamic)
    next_plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max())
    final = finish(next_plan)
    assert next_plan.resource_usage["main"][result]["replayed_sessions"] == 1
    parent = negate(current).compute()
    assert_frame_equal(store.read(final["results"]["main"][result]).panel.collect(dense=True, include_traces=True),
        expected(parent, True), check_exact=True)
    store.verify_identity(applied["identity"])


@pytest.mark.parametrize("problem", ["context", "update", "report", "holding", "external", "missing_index"])
def test_bad_original_authority_never_certifies_or_runs_history(tmp_path, monkeypatch, problem):
    before, graph, store, result, old = legacy(tmp_path, monkeypatch)
    identity = old["results"]["main"][result]
    bindings = dict(original_update=old["identity"], context_id="main",
        original_context={"anchor": "2024-01-01", "parameters": {}},
        report_id="test", report_digest="c" * 64)
    spec = graph._state()["spec"]
    if problem == "context":
        bindings["original_context"]["information_cutoff"] = "2024-02-01"
    elif problem == "update":
        bindings["context_id"] = "other"
    elif problem == "report":
        bindings["report_digest"] = "bad"
    elif problem in {"holding", "external"}:
        value = source(36, traces=False)
        expression = streak_count(negate(value), reset_on_equal=False) if problem == "holding" else streak_count(value)
        graph, store, result = global_graph(tmp_path / "other", expression)
        with monkeypatch.context() as patch:
            patch.setattr(streak_count, "contract", replace(streak_count.contract, execution=ExecutionMode.LAZY,
                checkpoint_replay=False, checkpoint_history=None, checkpoint_validator=None))
            old = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
        identity = old["results"]["main"][result]
        bindings["original_update"] = old["identity"]
        spec = graph._state()["spec"]
    else:
        monkeypatch.setattr(store, "interval_identity", lambda *_a, **_k: None)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("bad certification replayed history"))
    monkeypatch.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("bad certification read values"))
    with pytest.raises((KeyError, ValueError), match="original|Original|report"):
        store.plan_input_proof_rebind(spec, identity, mode="streak_checkpoint.v1", **bindings)


def test_budget_admission_precedes_original_value_reads(tmp_path, monkeypatch):
    _, graph, store, result, old = legacy(tmp_path, monkeypatch)
    monkeypatch.setattr(store, "value_read_rows", lambda *_a, **_k: 1_000_000)
    monkeypatch.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("budget denial consumed original values"))
    with resource_limits(ResourceLimits(memory_target_mib=1, cache_mib=0)):
        with pytest.raises(MemoryError, match="before value reads"):
            certify(graph, store, result, old)


def test_wide_trace_state_is_admitted_before_saved_value_reads(tmp_path, monkeypatch):
    raw = source(4, traces=False)
    names = tuple(f"trace_{index}" for index in range(512))
    frame = raw.collect().with_columns(pl.lit(raw.domain.times.min()).alias(name) for name in names)
    value = Node.from_domain(frame, raw.domain, source_key="x", trace_columns=names)
    graph, store, result = global_graph(tmp_path, streak_count(negate(value)))
    with monkeypatch.context() as legacy_contract:
        legacy_contract.setattr(streak_count, "contract", replace(streak_count.contract, execution=ExecutionMode.LAZY,
            checkpoint_replay=False, checkpoint_history=None, checkpoint_validator=None))
        old = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    monkeypatch.setattr(store, "read_values", lambda *_a, **_k: pytest.fail("wide trace denial consumed originals"))
    with resource_limits(ResourceLimits(memory_target_mib=1, cache_mib=0)):
        with pytest.raises(MemoryError, match="before value reads"):
            certify(graph, store, result, old)


def test_tampered_or_lost_admission_cannot_publish_or_reuse(tmp_path, monkeypatch):
    _, graph, store, result, old = legacy(tmp_path, monkeypatch)
    plan = certify(graph, store, result, old)
    bad = deepcopy(plan)
    bad["checkpoint"]["state"]["assets"]["a"]["count"] += 1
    with pytest.raises(ValueError, match="admission changed"):
        store.apply_input_proof_rebind(bad)
    applied = store.apply_input_proof_rebind(plan)
    with store.transaction() as db:
        db.execute("DELETE FROM input_proof_rebinding_plans WHERE identity=?", (plan["identity"],))
    with pytest.raises(ValueError, match="immutable admission"):
        store.read_checkpoint(applied["identity"])


def test_cancellation_leaves_original_and_retry_is_safe(tmp_path, monkeypatch):
    _, graph, store, result, old = legacy(tmp_path, monkeypatch)
    plan = certify(graph, store, result, old)
    def cancel():
        raise RuntimeError("operator canceled")
    with pytest.raises(RuntimeError, match="operator canceled"):
        store.apply_input_proof_rebind(plan, check_canceled=cancel)
    assert store.describe(MaterializationKey.from_dict(plan["replacement_key"]).identity) is None
    assert store.graph_state("global")["current_update"] == old["identity"]
    assert store.apply_input_proof_rebind(plan)["outcome"] == "certified"
