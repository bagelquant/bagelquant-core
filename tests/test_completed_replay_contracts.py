"""Completed history requires an incremental contract before a cache miss runs."""
from dataclasses import replace

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import ExecutionRuntime, IdentityPredictionOperator, EqualWeightPredictionOperator, Node, OPERATOR_REGISTRY
from bagelquant_core.operator import top_n, equal_weight, negate
from test_global_graph import source, global_graph, finish


@pytest.mark.parametrize("factory", [IdentityPredictionOperator, EqualWeightPredictionOperator])
def test_public_prediction_contract_matches_registered_execution(factory):
    assert factory().contract == OPERATOR_REGISTRY.get("prediction:" + factory.kind).contract
    assert factory().contract.history == 0


@pytest.mark.parametrize("case", ["append", "same_boundary_revision", "missing_authority",
                                  "new_context", "changed_implementation", "exact_hit"])
def test_unknown_history_never_silently_replays_completion(tmp_path, monkeypatch, case):
    operation = OPERATOR_REGISTRY.get("prediction:identity")
    monkeypatch.setattr(operation, "contract", replace(operation.contract, history=None))
    first = source(31)
    graph, store, root = global_graph(tmp_path, IdentityPredictionOperator()(first))
    old = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    current = (source(31, revision=10) if case == "same_boundary_revision"
               else source(31) if case == "exact_hit" else source(32))
    context = "main"
    if case == "new_context":
        graph.register_context("new", anchor="2024-01-01")
        context = "new"
    if case == "missing_authority":
        original_describe = store.describe
        identity = old["results"]["main"][root]
        monkeypatch.setattr(store, "describe", lambda key: None if key == identity else original_describe(key))
    if case == "changed_implementation":
        monkeypatch.setattr(operation, "version", "explicit-new-numerical-implementation")
    calls = []
    original_run = ExecutionRuntime.run

    def spy(self, bound, *args, **kwargs):
        calls.append(max(len(node.domain.times) for node in bound.nodes))
        return original_run(self, bound, *args, **kwargs)

    monkeypatch.setattr(ExecutionRuntime, "run", spy)
    plan = graph.plan_update({context: {"x": current}}, through=current.domain.times.max(),
                            **({"roots": [root]} if case == "new_context" else {}))
    blocked = case in {"append", "same_boundary_revision", "missing_authority"}
    if blocked:
        with pytest.raises(ValueError, match="explicit|plan affected intervals"):
            finish(plan)
        assert store.graph_state("global")["current_update"] == old["identity"]
    else:
        finish(plan)
    assert calls == ([] if blocked or case == "exact_hit" else [32])


@pytest.mark.parametrize("kind", ["top_n", "equal_weight", "identity_prediction", "equal_prediction"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_pointwise_append_reads_and_computes_only_new_session(tmp_path, kind, dynamic):
    def expression(node):
        if kind == "top_n":
            return top_n(node, count=1)
        if kind == "equal_weight":
            return equal_weight(top_n(node, count=1))
        if kind == "identity_prediction":
            return IdentityPredictionOperator()(node)
        return EqualWeightPredictionOperator()(node, negate(node))

    first = source(31, dynamic=dynamic)
    graph, store, root = global_graph(tmp_path / "incremental", expression(first))
    old = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    manifest = store.manifest(old["results"]["main"][root])
    old_files = {part["$file"]: store._file(part["$file"]).read_bytes() for part in manifest["partitions"]}
    current = source(32, dynamic=dynamic)
    reads = []

    def whole(*_):
        pytest.fail("pointwise append loaded historical source")

    def window(_context, _key, start, end):
        reads.append((start, end))
        return Node.from_domain(current.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
                                current.domain.window(start, end), identity=current.identity,
                                value_type=current.value_type, trace_columns=current.trace_columns,
                                trace_identity=current.trace_identity)

    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
                             input_loader=whole, input_window_loader=window)
    result = finish(plan)
    full, reference_store, full_root = global_graph(tmp_path / "reference", expression(current))
    reference = finish(full.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    assert_frame_equal(store.read(result["results"]["main"][root]).panel.collect(include_traces=True),
                       reference_store.read(reference["results"]["main"][full_root]).panel.collect(include_traces=True),
                       check_exact=True)
    assert reads and all(start == end == current.domain.times.max() for start, end in reads)
    assert all(usage["kernel_sessions"] == usage["computed_output_sessions"] == 1
               for usage in plan.resource_usage["main"].values())
    assert all(store._file(path).read_bytes() == original for path, original in old_files.items())
    assert set(old_files) <= {part["$file"] for part in store.manifest(result["results"]["main"][root])["partitions"]}
