"""Explicit bounded cold integration acceptance; synthetic only, no Data owner.

Run after the focused cold cases. This is a fresh ten-operator 6488-session
pair, not a claim about a captured Alpha or its elapsed time.
"""
from bagelquant_core import ExecutionRuntime
from test_finite_cold_batches import (
    NATIVE, LIMITS, bits, canonical, execute, scientific_inventory, source, ten_nodes,
)


@NATIVE
def test_6488_sessions_every203_block_all_ten_ids_bits_and_truthful_runtime_calls(tmp_path, monkeypatch):
    x = source(6488, extra=True)
    expression, sources = ten_nodes(x)
    run = ExecutionRuntime.run
    calls = []
    def observe(runtime, graph, **kwargs):
        calls.append(frozenset(graph.logical_spec().outputs.values()))
        return run(runtime, graph, **kwargs)
    monkeypatch.setattr(ExecutionRuntime, "run", observe)
    _, store, _, plan, receipt = execute(tmp_path / "new", x, expression=expression,
        sources=sources, limits=LIMITS)
    future = list(calls)
    calls.clear()
    with canonical(monkeypatch):
        _, old, _, reference, expected = execute(tmp_path / "old", x,
            expression=expression, sources=sources, limits=LIMITS)
    operators = {node.node_id for node in plan.spec.nodes if node.node_type == "operator"}
    assert len(operators) == 10
    assert set(plan.resource_usage["main"]) == set(reference.resource_usage["main"]) == operators
    assert set(receipt["results"]["main"]) == operators
    assert receipt["results"] == expected["results"]
    assert scientific_inventory(store) == scientific_inventory(old)
    assert len(future) < len(calls) == 2030
    for node in plan.spec.nodes:
        if node.node_type != "operator":
            continue
        identity = receipt["results"]["main"][node.node_id]
        current, prior = plan.resource_usage["main"][node.node_id], reference.resource_usage["main"][node.node_id]
        assert current["computed_blocks"] == prior["computed_blocks"] == 203
        assert current["computed_output_sessions"] == prior["computed_output_sessions"] == 6488
        assert current["reused_blocks"] == prior["reused_blocks"] == 0
        assert current["kernel_sessions"] <= prior["kernel_sessions"]
        assert current["input_sessions"] <= prior["input_sessions"]
        assert bits(store.read(identity).panel.collect(dense=False, include_traces=True)) == bits(
            old.read(identity).panel.collect(dense=False, include_traces=True))
        # External/membership inputs keep the exact original 203 calls.
        if node.operator.endswith("project_domain"):
            assert sum(node.node_id in call for call in future) == sum(node.node_id in call for call in calls) == 203
