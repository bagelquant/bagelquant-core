"""Paired session operations require current trace support, not nonnull values."""
from datetime import date

import polars as pl
import pytest

from bagelquant_core import Domain, ExecutionRuntime, Graph, Node
from bagelquant_core import execution
from bagelquant_core.operator import diff, pct_change


def source(*, known_null=False):
    days = [date(2024, 1, day) for day in (2, 3, 4, 5)]
    indices = [0, 1, 2, 3] if known_null else [0, 1, 3]
    frame = pl.DataFrame({"time": [days[i] for i in indices], "asset_id": ["A"] * len(indices),
        "value": [10.0, 11.0, None, 12.1] if known_null else [10.0, 11.0, 12.1],
        "observation_date": [days[i] for i in indices],
        "available_date": [days[i] for i in indices]})
    return days, Node.from_domain(frame, Domain(calendar=days, universe=["A"]),
        source_key="prices", identity="prices.v1", trace_columns=("observation_date", "available_date"))


@pytest.mark.parametrize("operation", [diff, pct_change])
@pytest.mark.parametrize("known_null", [False, True])
def test_paired_traces_preserve_known_null_and_reject_prior_only_support(operation, known_null):
    days, prices = source(known_null=known_null)
    actual = operation(prices, periods=1).compute().collect(dense=True, include_traces=True)
    assert actual["value"][0] is None and actual["value"][2] is None and actual["value"][3] is None
    assert actual["value"][1] == pytest.approx(1.0 if operation is diff else 0.1)
    for trace in ("observation_date", "available_date"):
        assert actual[trace].to_list() == [days[0], days[1], days[2] if known_null else None, days[3]]


def test_execution_kernel_version_separates_materialization_without_changing_logical_identity(monkeypatch):
    _, prices = source()
    output = pct_change(prices, periods=1, name="rate")
    graph = Graph(outputs=[output])
    current = ExecutionRuntime().plan_materialization_keys(graph)
    monkeypatch.setattr(execution, "EXECUTION_KERNEL_VERSION", "logical_runtime.v1")
    previous = ExecutionRuntime().plan_materialization_keys(graph)
    assert current.keys() == previous.keys()
    assert current[output.logical_id].implementation_id != previous[output.logical_id].implementation_id
