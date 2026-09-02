from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import broadcast_by_time
from bagelquant_core.transformer import ceil


def test_ceil_preserves_missing_values_and_rounds_finite_values() -> None:
    domain = Domain(
        calendar=[date(2024, 1, 2)], universe=["a", "b", "c", "d", "e"]
    )
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [date(2024, 1, 2)] * 5,
                "asset_id": ["a", "b", "c", "d", "e"],
                "value": [1.1, -1.1, None, float("inf"), float("nan")],
            }
        ),
        domain,
    )

    values = ceil(source).compute().collect(dense=True)["value"].to_list()
    assert values[:4] == [
        2.0,
        -1.0,
        None,
        float("inf"),
    ]
    assert values[4] is None


def test_broadcast_by_time_uses_like_keys_and_combines_traces() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    domain = Domain(calendar=days, universe=["a", "b", "benchmark"])
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["benchmark", "benchmark"],
                "value": [0.1, 0.2],
                "available": [days[0], days[1]],
            }
        ),
        domain,
        trace_columns=("available",),
    )
    like = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [days[0], days[0], days[1]],
                "asset_id": ["a", "b", "b"],
                "value": [1.0, 2.0, 3.0],
                "available": [days[0], days[1], days[1]],
            }
        ),
        domain,
        trace_columns=("available",),
    )

    result = broadcast_by_time(source, like).compute().collect(
        dense=False, include_traces=True
    )

    assert result.select("time", "asset_id", "value").to_dicts() == [
        {"time": days[0], "asset_id": "a", "value": 0.1},
        {"time": days[0], "asset_id": "b", "value": 0.1},
        {"time": days[1], "asset_id": "b", "value": 0.2},
    ]
    assert result["available"].to_list() == [days[0], days[1], days[1]]


def test_broadcast_by_time_rejects_multiple_source_rows_on_one_date() -> None:
    day = date(2024, 1, 2)
    domain = Domain(calendar=[day], universe=["a", "b"])
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [day, day],
                "asset_id": ["a", "b"],
                "value": [1.0, 2.0],
            }
        ),
        domain,
    )
    like = Panel.from_domain(
        pl.DataFrame({"time": [day], "asset_id": ["a"], "value": [0.0]}),
        domain,
    )

    with pytest.raises(ValueError, match="exactly one source row per date"):
        broadcast_by_time(source, like).compute()


def test_broadcast_by_time_omits_like_dates_without_a_source_value() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    domain = Domain(calendar=days, universe=["a", "benchmark"])
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [days[0]],
                "asset_id": ["benchmark"],
                "value": [0.1],
            }
        ),
        domain,
    )
    like = Panel.from_domain(
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a", "a"],
                "value": [1.0, 2.0],
            }
        ),
        domain,
    )

    result = broadcast_by_time(source, like).compute().collect(dense=False)

    assert result.select("time", "asset_id", "value").to_dicts() == [
        {"time": days[0], "asset_id": "a", "value": 0.1}
    ]
