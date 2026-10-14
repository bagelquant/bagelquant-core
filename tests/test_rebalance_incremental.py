"""Rebalance retains its explicit date axis across bounded checkpoint work."""
from datetime import date, timedelta
from contextlib import contextmanager
from dataclasses import replace

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, ExecutionRuntime, Graph, Node, NodeMaterialization
from bagelquant_core.operator import equal_weight, rebalance, top_n
from test_global_graph import finish, global_graph


def _inputs(count, *, empty=(), changed_calendar=False):
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(count)]
    members = pl.DataFrame({"time": [day for index, day in enumerate(dates) if index not in empty],
                            "asset_id": ["a"] * sum(index not in empty for index in range(count))},
                           schema={"time": pl.Date, "asset_id": pl.String})
    domain = Domain(calendar=dates, universe=members.with_columns(pl.lit(True).alias("active")))
    frame = members.with_columns(pl.lit(1.0).alias("value"), pl.col("time").alias("available_date"))
    weights = Node.from_domain(frame, domain, source_key="x", trace_columns=("available_date",))
    calendar_dates = [day for index, day in enumerate(dates) if not changed_calendar or index != 10]
    calendar_domain = (Domain(calendar=calendar_dates, universe=members.with_columns(pl.lit(True).alias("active")),
                              identity=domain.signature) if changed_calendar else domain)
    calendar = Node.from_domain(frame.filter(pl.col("time").is_in(calendar_dates)), calendar_domain,
                                source_key="calendar", trace_columns=("available_date",))
    return {"x": weights, "calendar": calendar}


def _expression(inputs):
    return rebalance(equal_weight(top_n(inputs["x"], count=1)), calendar=inputs["calendar"],
                     data_start="2024-01-01", every=5)


@pytest.mark.parametrize("old_count,new_count,empty", [
    (31, 32, ()), (35, 36, ()), (31, 32, (0, 30, 31)),
    (35, 36, (35,)), (31, 32, tuple(range(32))),
])
def test_rebalance_append_complete_axis_and_reference(tmp_path, old_count, new_count, empty):
    first = _inputs(old_count, empty=empty)
    graph, store, root = global_graph(tmp_path / "incremental", _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    manifest = store.manifest(old["results"]["main"][root])
    original = {part["$file"]: store._file(part["$file"]).read_bytes() for part in manifest["partitions"]}
    original_decisions = manifest["artifacts"]["decisions"]
    decision_bytes = store._file(original_decisions["$file"]).read_bytes()
    current = _inputs(new_count, empty=empty)
    reads = []

    def whole(*_):
        pytest.fail("rebalance append loaded complete source history")

    def window(_context, key, start, end):
        reads.append((key, start, end))
        value = current[key]
        return Node.from_domain(value.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
                                value.domain.window(start, end), identity=value.identity,
                                value_type=value.value_type, trace_columns=value.trace_columns,
                                trace_identity=value.trace_identity)

    plan = graph.plan_update({"main": current}, through=current["x"].domain.times.max(),
                             input_loader=whole, input_window_loader=window)
    updated = finish(plan)
    full, full_store, full_root = global_graph(tmp_path / "reference", _expression(current))
    expected = finish(full.plan_update({"main": current}, through=current["x"].domain.times.max()))
    identity = updated["results"]["main"][root]
    reference = expected["results"]["main"][full_root]
    assert_frame_equal(store.read(identity).panel.collect(include_traces=True),
                       full_store.read(reference).panel.collect(include_traces=True), check_exact=True)
    decisions = store.evidence(identity, channels=("decisions",))["artifacts"]["decisions"]
    assert_frame_equal(decisions, full_store.evidence(reference, channels=("decisions",))["artifacts"]["decisions"],
                       check_exact=True)
    assert decisions.height == new_count
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 1
    assert reads and all(start == end == current["x"].domain.times.max() for _, start, end in reads)
    assert all(store._file(path).read_bytes() == content for path, content in original.items())
    assert set(original) <= {part["$file"] for part in store.manifest(identity)["partitions"]}
    assert store.manifest(identity)["artifacts"]["decisions"]["$artifact_concat"][0] == original_decisions
    assert store._file(original_decisions["$file"]).read_bytes() == decision_bytes
    assert store.read(identity).checkpoint["state"]["through"] == str(current["x"].domain.times.max())
    assert finish(graph.plan_update({"main": current}, through=current["x"].domain.times.max()))["results"] == updated["results"]


def test_rebalance_forged_equal_domain_calendar_blocks_before_math(tmp_path, monkeypatch):
    first = _inputs(31)
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    current = _inputs(32, changed_calendar=True)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_args, **_kwargs: pytest.fail("changed calendar ran math"))
    with pytest.raises(ValueError, match="calendar|Domain"):
        finish(graph.plan_update({"main": current}, through=current["x"].domain.times.max()))
    assert store.graph_state("global")["current_update"] == old["identity"]


@pytest.mark.parametrize("empty", [(0, 5, 9), tuple(range(10))])
def test_public_node_explicit_axis_includes_empty_sessions(empty):
    inputs = _inputs(10, empty=empty)
    runtime = ExecutionRuntime()
    node = _expression(inputs)
    result = runtime.run(node.graph, dense_output=False)
    decisions = runtime.node_artifacts[node.logical_id]["decisions"]
    assert decisions.height == 10
    assert decisions["time"].to_list() == inputs["calendar"].domain.times.to_list()
    if 0 in empty:
        assert decisions.row(0, named=True)["status"] == "unavailable"
        assert decisions.row(0, named=True)["reason"] == "empty_universe"
    assert result.collect().schema["value"] == pl.Float64


def test_symbolic_rebalance_defers_axis_until_actual_binding(monkeypatch):
    symbolic = _expression({"x": Node.symbolic("x"), "calendar": Node.symbolic("calendar")})
    inputs = _inputs(10, empty=(0, 5, 9))
    bound = Graph.from_logical_spec(symbolic.logical_spec(), inputs=inputs)
    runtime = ExecutionRuntime()
    result = bound.compute(runtime=runtime, dense_output=False)
    expected = _expression(inputs).compute(dense_output=False)
    assert_frame_equal(result.collect(include_traces=True), expected.collect(include_traces=True), check_exact=True)
    invalid = _inputs(32, changed_calendar=True)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("unmatched symbolic calendar ran math"))
    with pytest.raises(ValueError, match="calendar axis"):
        Graph.from_logical_spec(symbolic.logical_spec(), inputs=invalid)


def test_frame_rebalance_uses_supplied_calendar_only():
    from bagelquant_core.operator.portfolio import rebalance_value
    inputs = _inputs(10, empty=(0, 5))
    frame = inputs["x"].collect()
    axis = pl.DataFrame({"time": inputs["calendar"].domain.times})
    result = rebalance_value(frame, calendar=axis, data_start="2024-01-01", coverage_calendar=axis)
    assert result.decisions.height == 10
    assert result.decisions.filter(pl.col("status") == "unavailable")["time"].to_list() == [date(2024, 1, 1), date(2024, 1, 6)]
    with pytest.raises(ValueError, match="absent"):
        rebalance_value(frame, calendar=axis.filter(pl.col("time") != date(2024, 1, 6)),
                        data_start="2024-01-01", coverage_calendar=axis)


def test_second_append_reuses_original_decisions_without_decode(tmp_path, monkeypatch):
    first = _inputs(31, empty=(30, 35))
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    original_file = store.manifest(old["results"]["main"][root])["artifacts"]["decisions"]["$file"]
    original_decode = store._decode

    def reject_original(value):
        if isinstance(value, dict) and (value.get("$file") == original_file or "$artifact_concat" in value):
            pytest.fail("continuation decoded old decision table")
        return original_decode(value)

    monkeypatch.setattr(store, "_decode", reject_original)
    for count in (32, 36):
        current = _inputs(count, empty=(30, 35))
        updated = finish(graph.plan_update({"main": current}, through=current["x"].domain.times.max()))
        refs = store.manifest(updated["results"]["main"][root])["artifacts"]["decisions"]["$artifact_concat"]
        assert refs[0]["$file"] == original_file
    monkeypatch.setattr(store, "_decode", original_decode)
    full, full_store, full_root = global_graph(tmp_path / "reference", _expression(current))
    expected = finish(full.plan_update({"main": current}, through=current["x"].domain.times.max()))
    assert_frame_equal(store.evidence(updated["results"]["main"][root], channels=("decisions",))["artifacts"]["decisions"],
        full_store.evidence(expected["results"]["main"][full_root], channels=("decisions",))["artifacts"]["decisions"], check_exact=True)


def test_checkpoint_candidate_never_decodes_original_domain(tmp_path, monkeypatch):
    first = _inputs(31, empty=(0, 30))
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    saved_domain = store.manifest(old["results"]["main"][root])["domain"]
    forbidden = {saved_domain[name]["$file"] for name in ("calendar", "membership")}
    original_decode = store._decode

    def reject_domain(value):
        if isinstance(value, dict) and value.get("$file") in forbidden:
            pytest.fail("checkpoint eligibility decoded original Domain")
        return original_decode(value)

    monkeypatch.setattr(store, "_decode", reject_domain)
    current = _inputs(32, empty=(0, 30))
    plan = graph.plan_update({"main": current}, through=current["x"].domain.times.max())
    finish(plan)
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 1


@pytest.mark.parametrize("change", ["membership", "calendar"])
def test_domain_matches_actual_coordinates_not_opaque_id(tmp_path, monkeypatch, change):
    first = _inputs(31)
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    identity = old["results"]["main"][root]
    key = store.manifest(identity)["key"]["domain_identity"]
    actual = first["x"].domain
    calendar = actual.times if change == "membership" else actual.times.head(30)
    members = actual.grid_lazy().collect()
    if change == "membership":
        members = members.filter(pl.col("time") != date(2024, 1, 10))
    forged = Domain(calendar=calendar, universe=members.with_columns(pl.lit(True).alias("active")), identity=key)
    monkeypatch.setattr(store, "_decode", lambda *_a, **_k: pytest.fail("coordinate proof decoded original"))
    assert store.domain_matches(identity, actual) is True
    assert store.domain_matches(identity, forged) is False


@pytest.mark.parametrize("damage", ["missing", "old_version", "crc"])
def test_checkpoint_domain_proof_unknown_or_corrupt_blocks_before_math(tmp_path, monkeypatch, damage):
    first = _inputs(31)
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    identity = old["results"]["main"][root]
    with store.transaction() as connection:
        if damage == "missing":
            connection.execute("DELETE FROM computation_index WHERE identity=?", (identity,))
        elif damage == "old_version":
            connection.execute("UPDATE computation_index SET version='old' WHERE identity=?", (identity,))
        else:
            connection.execute("UPDATE computation_index SET payload_identity='wrong' WHERE identity=?", (identity,))
    current = _inputs(32)
    plan = graph.plan_update({"main": current}, through=current["x"].domain.times.max())
    while root not in plan.ready("main"):
        for node in plan.ready("main"):
            if node != root:
                plan.execute("main", node)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("unproven Domain ran checkpoint math"))
    with pytest.raises(ValueError, match="interval-index maintenance|corrupt Core computation index"):
        finish(plan)
    assert store.graph_state("global")["current_update"] == old["identity"]


@pytest.mark.parametrize("problem", ["schema", "overlap", "missing_channel"])
def test_artifact_composition_admission(tmp_path, problem):
    first = _inputs(31)
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    identity = old["results"]["main"][root]
    tail = pl.DataFrame({"time": [date(2024, 2, 1)], "status": ["hold"], "reason": [None]},
                        schema={"time": pl.Date, "status": pl.String, "reason": pl.String})
    if problem == "schema":
        tail = tail.with_columns(pl.col("time").cast(pl.String))
    elif problem == "overlap":
        tail = tail.with_columns(pl.lit(date(2024, 1, 31)).alias("time"))
    with pytest.raises(ValueError, match="schema|overlap|channel"):
        store.compose_artifact(identity, "missing" if problem == "missing_channel" else "decisions", tail)


def test_damaged_rebalance_checkpoint_blocks_without_replay(tmp_path, monkeypatch):
    first = _inputs(31)
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    read_checkpoint = store.read_checkpoint

    def damaged(identity):
        original = read_checkpoint(identity)
        return {**original, "state": {"through": "not-a-date"}} if original is not None else None

    monkeypatch.setattr(store, "read_checkpoint", damaged)
    current = _inputs(32)
    plan = graph.plan_update({"main": current}, through=current["x"].domain.times.max())
    # Upstream finite new-session work is allowed. The rebalance state itself
    # must never be replaced by a cold historical calculation.
    while root not in plan.ready("main"):
        for node in plan.ready("main"):
            if node != root:
                plan.execute("main", node)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("damaged state replayed history"))
    with pytest.raises(ValueError, match="no proven preceding state"):
        finish(plan)
    assert store.graph_state("global")["current_update"] == old["identity"]


@pytest.mark.parametrize("problem", ["authority", "file"])
def test_artifact_original_authority_fenced_in_closing_writer(tmp_path, monkeypatch, problem):
    first = _inputs(31)
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    original_id = old["results"]["main"][root]
    record = store.read(original_id)
    reference = store.compose_artifact(original_id, "decisions")
    original_file = store._file(store.manifest(original_id)["artifacts"]["decisions"]["$file"])
    key = replace(record.key, context_identity="artifact-closing-fence")
    transaction = store.transaction
    calls = 0

    @contextmanager
    def remove_before_writer():
        nonlocal calls
        calls += 1
        # publish's first transaction is its existing initialize() metadata
        # check. Mutate only after preparation, immediately before final writer.
        if calls == 2:
            if problem == "authority":
                with transaction() as connection:
                    connection.execute("DELETE FROM node_completions WHERE identity=?", (original_id,))
                    connection.execute("DELETE FROM materializations WHERE identity=?", (original_id,))
            else:
                original_file.unlink()
        with transaction() as connection:
            yield connection

    monkeypatch.setattr(store, "transaction", remove_before_writer)
    with pytest.raises(ValueError, match="original artifact (authority|file) missing"):
        store.publish(NodeMaterialization(key, record.panel, artifacts={"decisions": reference}))
    assert store.manifest(key.identity) is None


def test_rebalance_cancellation_retains_original_global_pointer(tmp_path):
    first = _inputs(31)
    graph, store, _ = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    current = _inputs(32)
    plan = graph.plan_update({"main": current}, through=current["x"].domain.times.max())

    def canceled():
        raise RuntimeError("isolated cooperative cancellation")

    with pytest.raises(RuntimeError, match="cooperative cancellation"):
        finish(plan, check_canceled=canceled)
    assert store.graph_state("global")["current_update"] == old["identity"]


def test_explicit_context_cannot_override_actual_calendar_axis():
    from bagelquant_core import capture_operator_checkpoints
    from bagelquant_core.operator_state import current_operator_checkpoints
    inputs = _inputs(10)
    node = _expression(inputs)
    days = inputs["calendar"].domain.times.to_list()
    with capture_operator_checkpoints(calendar=days, anchor_calendars={node.logical_id: days[:5] + days[6:]}):
        with pytest.raises(ValueError, match="retained explicit calendar differs"):
            ExecutionRuntime().run(node.graph)
    assert current_operator_checkpoints() is None


def test_legacy_generic_rebalance_completion_requires_maintenance(tmp_path, monkeypatch):
    operation = rebalance
    contract = operation.contract
    monkeypatch.setattr(operation, "contract", replace(contract, checkpoint_replay=False, checkpoint_calendar_role=None))
    first = _inputs(31)
    graph, store, root = global_graph(tmp_path, _expression(first))
    old = finish(graph.plan_update({"main": first}, through=first["x"].domain.times.max()))
    monkeypatch.setattr(operation, "contract", contract)
    current = _inputs(32)
    plan = graph.plan_update({"main": current}, through=current["x"].domain.times.max())
    while root not in plan.ready("main"):
        for node in plan.ready("main"):
            plan.execute("main", node)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("legacy generic result replayed history"))
    with pytest.raises(ValueError, match="explicit checkpoint/input-proof maintenance"):
        finish(plan)
    assert store.graph_state("global")["current_update"] == old["identity"]
