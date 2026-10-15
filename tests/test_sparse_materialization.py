"""Sparse outputs share physical collection without sharing positional traces."""
from datetime import date, timedelta

import polars as pl
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, ExecutionRuntime, Node
from bagelquant_core.execution import PlanValue
from bagelquant_core.operation_contract import InputDensity
from bagelquant_core.operator import ceil, negate


def fixture():
    days = [date(2024, 1, day) for day in (1, 2, 3)]
    keys = pl.DataFrame({"time": [days[0], days[0], days[1], days[2], days[2]],
        "asset_id": ["A", "B", "A", "A", "B"]})
    domain = Domain(calendar=days, universe=keys.with_columns(pl.lit(True).alias("active")))
    frame = keys.with_columns(pl.Series("value", [-0., None, float("inf"), 1. / 3, float("-inf")]))
    traces = keys.with_columns(pl.Series("available", [day + timedelta(days=index)
        for index, day in enumerate(keys["time"])]))
    return domain, frame, traces


def test_sparse_outputs_batch_key_joined_traces_and_reuse_validated_frames(monkeypatch):
    domain, frame, traces = fixture()
    source = Node.from_domain(frame.join(traces, on=["time", "asset_id"]), domain,
        source_key="x", trace_columns=("available",))
    definitions = (negate(source), ceil(source))
    selections = (frame[[4, 0, 2]], frame[[3, 1]])
    # Each output has trace-only coordinates; the first also has a value-only
    # coordinate whose trace must remain null after the original left join.
    traces = traces.filter(~((pl.col("time") == frame["time"][0]) & (pl.col("asset_id") == "A")))
    plans = [PlanValue(frame=selected.lazy(), domain=domain, density=InputDensity.SPARSE_OK,
        identity=f"output-{index}", trace_columns=("available",), trace_frame=traces.reverse().lazy(),
        trace_identity="shared-original-traces") for index, selected in enumerate(selections)]
    reference = [ExecutionRuntime()._materialize_node(node, plan, dense_output=False)
        .collect(dense=False, include_traces=True)
        for node, plan in zip(definitions, plans, strict=True)]
    original = pl.collect_all
    calls = []
    def collect_all(values, **kwargs):
        calls.append(len(values))
        return original(values, **kwargs)
    monkeypatch.setattr(pl, "collect_all", collect_all)
    runtime = ExecutionRuntime()
    outputs = runtime._materialize_many(list(zip(definitions, plans, strict=True)), dense_output=False)
    assert calls == [2]
    expected = [selected.join(traces, on=["time", "asset_id"], how="left").sort("time", "asset_id")
        for selected in selections]
    for original_frame, wanted in zip(reference, expected, strict=True):
        assert_frame_equal(original_frame, wanted, check_exact=True)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("validated sparse output was physically collected again")
    with monkeypatch.context() as cached:
        cached.setattr(pl.LazyFrame, "collect", forbidden)
        for node, wanted in zip(definitions, expected, strict=True):
            actual = outputs[node.name].collect(dense=False, include_traces=True)
            assert_frame_equal(actual, wanted, check_exact=True)
            assert [None if value is None else value.hex() for value in actual["value"]] == [
                None if value is None else value.hex() for value in wanted["value"]]
    returned = outputs[definitions[0].name].collect(dense=False, include_traces=True)
    returned.replace_column(returned.columns.index("value"), pl.Series("value", [99.] * returned.height))
    assert_frame_equal(outputs[definitions[0].name].collect(dense=False, include_traces=True), expected[0], check_exact=True)
    assert outputs[definitions[0].name].collect(dense=True, include_traces=True).height == domain.size


def test_dense_then_sparse_runtime_keeps_original_sparse_rows_and_dense_padding():
    domain, frame, traces = fixture()
    selected = frame[[4, 0, 2]].join(traces, on=["time", "asset_id"])
    value = Node.from_domain(selected, domain, source_key="x", trace_columns=("available",))
    graph = negate(value).graph
    runtime = ExecutionRuntime()
    dense = runtime.run(graph, dense_output=True).collect(include_traces=True)
    sparse_node = runtime.run(graph, dense_output=False)
    sparse = sparse_node.collect(dense=False, include_traces=True)
    fresh = ExecutionRuntime().run(negate(value).graph, dense_output=False).collect(dense=False, include_traces=True)
    assert sparse.height == selected.height and dense.height == domain.size
    assert_frame_equal(sparse, fresh, check_exact=True)
    assert_frame_equal(sparse_node.collect(dense=True, include_traces=True), dense, check_exact=True)
