from __future__ import annotations

from datetime import date

import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import (
    add,
    sum_frames,
    weighted_sum,
)
from bagelquant_core.transformer import group_mean
from bagelquant_core.composer.core import _horizontal_value_plan

from helpers import panel, values


def test_horizontal_value_plan_preserves_positionally_aligned_lazy_rows() -> None:
    keys = pl.DataFrame(
        {
            "time": [date(2024, 1, 1), date(2024, 1, 2)],
            "asset_id": ["a", "b"],
        }
    )
    left = keys.with_columns(pl.Series("value", [1.0, 2.0])).lazy()
    right = keys.with_columns(pl.Series("value", [10.0, 20.0])).lazy()

    combined, values = _horizontal_value_plan((left, right))

    assert combined.select(*values).collect().to_dicts() == [
        {"__value_0": 1.0, "__value_1": 10.0},
        {"__value_0": 2.0, "__value_1": 20.0},
    ]


def test_add_joins_on_time_asset_id() -> None:
    left = panel([("2024-01-01", "a", 1.0), ("2024-01-01", "b", 2.0)], name="left")
    right = panel([("2024-01-01", "a", 10.0), ("2024-01-01", "b", 20.0)], name="right")

    graph = add(left, right)
    graph.compute()

    assert values(graph.output.collect(dense=True)) == {
        ("2024-01-01", "a"): 11.0,
        ("2024-01-01", "b"): 22.0,
    }


def test_group_mean_uses_group_panel() -> None:
    frame = panel([("2024-01-01", "a", 1.0), ("2024-01-01", "b", 3.0)], name="x")
    group = panel([("2024-01-01", "a", 1.0), ("2024-01-01", "b", 1.0)], name="g")

    graph = group_mean(frame, group=group)
    graph.compute()

    assert values(graph.output.collect(dense=True)) == {
        ("2024-01-01", "a"): 2.0,
        ("2024-01-01", "b"): 2.0,
    }


def test_wide_aggregation_preserves_sparse_inner_alignment() -> None:
    domain = Domain(
        calendar=["2024-01-01", "2024-01-02"],
        universe=["a", "b"],
    )
    panels = []
    for index in range(7):
        frame = pl.DataFrame(
            {
                "time": ["2024-01-02", "2024-01-01"],
                "asset_id": ["b", "a" if index % 2 == 0 else "b"],
                "value": [float(index + 1), float(index + 10)],
            }
        )
        panels.append(
            Panel.from_domain(frame, domain, name=f"input_{index}")
        )

    summed = sum_frames(*panels)
    weighted = weighted_sum(*panels, weights=[1.0] * len(panels))
    summed.compute()
    weighted.compute()

    expected = {
        ("2024-01-01", "a"): None,
        ("2024-01-01", "b"): None,
        ("2024-01-02", "a"): None,
        ("2024-01-02", "b"): 28.0,
    }
    assert values(summed.output.collect(dense=True)) == expected
    assert values(weighted.output.collect(dense=True)) == expected


def test_wide_aggregation_direct_output_remains_sorted() -> None:
    frames = [
        pl.DataFrame(
            {
                "time": ["2024-01-02", "2024-01-01"],
                "asset_id": ["b", "a"],
                "value": [float(index), float(index + 1)],
            }
        )
        for index in range(10)
    ]

    result = sum_frames.operation(*frames)

    assert result.get_column("time").to_list() == sorted(
        result.get_column("time").to_list()
    )
    assert result.get_column("value").to_list() == [55.0, 45.0]
