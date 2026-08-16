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
from bagelquant_core.composer import add, mul, sum_frames
from bagelquant_core.transformer import (
    bfill,
    constant,
    ffill,
    group_demean,
    identity,
    lag,
    pct_change,
    rolling_ols,
    rolling_mean,
    rolling_percentile,
    rolling_rank,
)
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

    assert output.collect(dense=True)["value"].to_list() == [1000.0, 1100.0, 1200.0]
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

    assert first.collect(dense=True)["value"].to_list() == [20.0, 22.0, 24.0]
    assert second.collect(dense=True).equals(first.collect(dense=True))
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


def test_transformer_panel_parameters_participate_in_availability_trace() -> None:
    source = _traced_panel()
    delayed = Panel.from_domain(
        source.collect(include_traces=True).with_columns(
            pl.Series(
                "base_available_date",
                [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 4)],
            )
        ).lazy(),
        source.domain,
        name="group",
        trace_columns=("observation_date", "base_available_date"),
    )

    output = group_demean(source, group=delayed).compute().collect(
        include_traces=True
    )

    assert output["base_available_date"].to_list() == [
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 4),
    ]


def test_trace_rules_preserve_exact_shift_fill_and_rolling_dates() -> None:
    days = [date(2024, 2, day) for day in (1, 2, 3)]
    domain = Domain(calendar=days, universe=["A"])
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["A"] * 3,
                "value": [1.0, None, 3.0],
                "observation_date": days,
                "base_available_date": [days[0], None, days[2]],
            }
        ),
        domain,
        trace_columns=("observation_date", "base_available_date"),
    )

    outputs = {
        "passthrough": identity(source),
        "shift": lag(source, periods=1),
        "forward": ffill(source),
        "backward": bfill(source),
        "rolling": rolling_mean(source, window=2, min_periods=1),
    }
    actual = {
        name: graph.compute()
        .collect(include_traces=True)
        .get_column("base_available_date")
        .to_list()
        for name, graph in outputs.items()
    }

    assert actual == {
        "passthrough": [days[0], None, days[2]],
        "shift": [None, days[0], None],
        "forward": [days[0], days[0], days[2]],
        "backward": [days[0], days[2], days[2]],
        "rolling": [days[0], days[0], days[2]],
    }


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


def test_sibling_eager_outputs_reuse_materialized_input() -> None:
    source = _traced_panel()
    runtime = ExecutionRuntime()

    Graph(
        outputs=[
            rolling_rank(source, window=2, name="rank"),
            rolling_percentile(source, window=2, name="percentile"),
        ]
    ).compute(runtime=runtime)

    assert runtime.eager_barriers == 2
    assert runtime.materializations == 2


def test_multiple_outputs_reuse_exact_trace_plan() -> None:
    source = _traced_panel()
    changed = pct_change(source, periods=1, name="changed")
    first = add(changed, source, name="first")
    second = mul(changed, source, name="second")
    runtime = ExecutionRuntime()

    outputs = Graph(outputs=[first, second]).compute(runtime=runtime)

    assert runtime.materializations == 1
    for output in outputs.values():
        traced = output.collect(include_traces=True)
        assert traced.get_column("base_available_date").to_list() == [
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
        ]


def test_semantically_equal_named_nodes_share_physical_plan() -> None:
    source = _traced_panel()
    runtime = ExecutionRuntime()
    graph = Graph(
        outputs=[
            rolling_mean(
                source,
                window=2,
                min_periods=1,
                name=f"mean_{index}",
                metadata={"index": index},
            )
            for index in range(5)
        ]
    )

    outputs = graph.compute(runtime=runtime)

    assert list(outputs) == [f"mean_{index}" for index in range(5)]
    assert len({panel.identity for panel in outputs.values()}) == 5
    assert [
        panel.metadata["index"] for panel in outputs.values()
    ] == list(range(5))
    expected = outputs["mean_0"].collect(include_traces=True)
    assert all(
        panel.collect(include_traces=True).equals(expected)
        for panel in outputs.values()
    )
    assert runtime._diagnostics["semantic_cse_hits"] == 4
    assert runtime.materializations == 1


def test_semantically_equal_eager_nodes_share_physical_result() -> None:
    source = _traced_panel()
    runtime = ExecutionRuntime()
    outputs = Graph(
        outputs=[
            rolling_ols(
                source,
                factors=(source,),
                window=2,
                name=f"ols_{index}",
                metadata={"index": index},
            )
            for index in range(5)
        ]
    ).compute(runtime=runtime)

    assert len({panel.identity for panel in outputs.values()}) == 5
    assert [panel.metadata["index"] for panel in outputs.values()] == list(
        range(5)
    )
    assert runtime.eager_barriers == 5
    assert runtime.materializations == 2
    assert runtime._diagnostics["eager_cse_hits"] == 4


def test_same_key_composer_uses_positional_plan() -> None:
    source = Panel.from_domain(_traced_panel().collect(dense=True), _traced_panel().domain)
    runtime = ExecutionRuntime()

    output = sum_frames(*([source] * 10)).compute(runtime=runtime)

    assert output.collect(dense=True)["value"].to_list() == [100.0, 110.0, 120.0]
    assert runtime._diagnostics["positional_composer_hits"] == 1
    assert "SORT BY" not in output.lazy(dense=False).explain().upper()


def test_unvalidated_composer_input_keeps_conservative_join() -> None:
    source = _traced_panel()
    runtime = ExecutionRuntime()

    add(source, source).compute(runtime=runtime)

    assert runtime._diagnostics["positional_composer_hits"] == 0


def test_custom_operation_remains_scoped_sorted_and_conservative() -> None:
    days = [date(2024, 3, day) for day in (1, 2)]
    domain = Domain(calendar=days, universe=["A"])
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["A", "A"],
                "value": [1.0, 2.0],
            }
        ),
        domain,
    )

    @transformer
    def disorder(frame: pl.DataFrame) -> pl.DataFrame:
        return pl.concat(
            [
                frame.reverse(),
                pl.DataFrame(
                    {
                        "time": [date(2024, 3, 3)],
                        "asset_id": ["OUTSIDE"],
                        "value": [99.0],
                    }
                ),
            ]
        )

    output = disorder(source).compute().collect(dense=True)

    assert output.select("time", "asset_id").rows() == [
        (days[0], "A"),
        (days[1], "A"),
    ]
    assert output["value"].to_list() == [1.0, 2.0]


def test_direct_rolling_operation_preserves_public_sorting() -> None:
    frame = pl.DataFrame(
        {
            "time": [date(2024, 4, 2), date(2024, 4, 1)],
            "asset_id": ["A", "A"],
            "value": [2.0, 1.0],
        }
    )

    output = rolling_mean.operation(
        frame,
        window=2,
        min_periods=1,
    )

    assert output["time"].to_list() == sorted(output["time"].to_list())


def test_direct_composer_operation_preserves_public_sorting() -> None:
    frame = pl.DataFrame(
        {
            "time": [date(2024, 4, 2), date(2024, 4, 1)],
            "asset_id": ["A", "A"],
            "value": [2.0, 1.0],
        }
    )

    output = sum_frames.operation(frame, frame)

    assert output["time"].to_list() == sorted(output["time"].to_list())


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
