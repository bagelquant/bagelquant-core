from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_core import (
    Domain,
    ExecutionRuntime,
    Graph,
    OperationContract,
    Panel,
    TraceRule,
)
from bagelquant_core.composer import add, mul
from bagelquant_core.transformer import constant, pct_change, rolling_rank
from bagelquant_core.transformer.core import transformer


def _traced_panel(*, identity: str | None = None) -> Panel:
    days = [date(2024, 1, day) for day in (2, 3, 4)]
    domain = Domain(calendar=days, universe=["A"])
    frame = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["A"] * 3,
            "value": [10.0, 11.0, 12.0],
            "observation_date": days,
            "base_available_date": days,
        }
    )
    return Panel.from_domain(
        frame.lazy(),
        domain,
        name="input",
        identity=identity,
        trace_columns=("observation_date", "base_available_date"),
    )


def test_simple_lazy_graph_materializes_only_the_final_output() -> None:
    source = _traced_panel()
    graph = mul(source, constant(source, value=100.0), name="scaled")
    runtime = ExecutionRuntime()

    output = graph.compute(runtime=runtime)

    assert output.data["value"].to_list() == [1000.0, 1100.0, 1200.0]
    assert runtime.eager_barriers == 0
    assert runtime.materializations == 1


def test_eager_operation_is_one_barrier_plus_final_materialization() -> None:
    runtime = ExecutionRuntime()

    rolling_rank(_traced_panel(), window=2).compute(runtime=runtime)

    assert runtime.eager_barriers == 1
    assert runtime.materializations == 2


def test_compiled_graph_rebinds_inputs_without_revalidating_topology() -> None:
    source = _traced_panel(identity="month:2024-01")
    specification = add(source, source, name="double").spec()
    compiled = Graph.compile(specification)
    runtime = ExecutionRuntime()

    first = compiled.compute({"input": source}, runtime=runtime)
    second_source = _traced_panel(identity="month:2024-02")
    second = compiled.compute({"input": second_source}, runtime=runtime)

    assert first.data["value"].to_list() == [20.0, 22.0, 24.0]
    assert second.data.equals(first.data)
    assert runtime.materializations == 2


def test_trace_rules_cover_window_and_parent_max() -> None:
    source = _traced_panel()
    changed = pct_change(source, periods=2, name="changed")
    combined = add(changed, source, name="combined")

    output = combined.compute().collect(include_traces=True)

    assert output["base_available_date"].to_list() == [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]


def test_custom_operator_with_traces_must_declare_trace_rule() -> None:
    @transformer
    def unsafe(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    with pytest.raises(ValueError, match="no trace rule"):
        unsafe(_traced_panel()).compute()


def test_custom_trace_contract_can_opt_in_explicitly() -> None:
    @transformer(
        contract=OperationContract(trace_rule=TraceRule.PASSTHROUGH)
    )
    def safe(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame

    output = safe(_traced_panel()).compute().collect(include_traces=True)

    assert output["observation_date"].to_list() == output["time"].to_list()


def test_input_instance_tokens_prevent_accidental_cross_input_cache_hits() -> None:
    runtime = ExecutionRuntime()
    first = _traced_panel()
    second = _traced_panel()

    add(first, first, name="sum").compute(runtime=runtime)
    add(second, second, name="sum").compute(runtime=runtime)

    assert runtime.materializations == 2


def test_multiple_outputs_share_one_final_collection() -> None:
    source = _traced_panel()
    first = add(source, source, name="sum")
    second = mul(source, source, name="product")
    runtime = ExecutionRuntime()

    result = Graph(outputs=[first, second]).compute(runtime=runtime)

    assert list(result) == ["sum", "product"]
    assert runtime.materializations == 1


def test_graph_execution_never_hashes_full_frames(monkeypatch) -> None:
    source = _traced_panel()

    def fail_if_called(_frame):
        raise AssertionError("graph execution must not hash a full DataFrame")

    monkeypatch.setattr(
        "bagelquant_core.hashing.hash_dataframe",
        fail_if_called,
    )

    add(source, source).compute()


def test_explicit_stable_identity_reuses_equivalent_input_results() -> None:
    runtime = ExecutionRuntime()
    first = _traced_panel(identity="immutable:monthly-snapshot")
    second = _traced_panel(identity="immutable:monthly-snapshot")

    add(first, first, name="sum").compute(runtime=runtime)
    add(second, second, name="sum").compute(runtime=runtime)

    assert runtime.materializations == 1
