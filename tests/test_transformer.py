from __future__ import annotations

from datetime import date

import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.transformer import lag, rank, rolling_mean, zscore

from helpers import panel, values


def test_cross_sectional_rank_uses_time_groups() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 2.0),
            ("2024-01-01", "b", 1.0),
        ]
    )

    graph = rank(source)
    graph.compute()

    assert values(graph.output.data) == {
        ("2024-01-01", "a"): 1.0,
        ("2024-01-01", "b"): 0.5,
    }


def test_rolling_mean_uses_asset_id_groups() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 3.0),
            ("2024-01-01", "b", 10.0),
            ("2024-01-02", "b", 20.0),
        ]
    )

    graph = rolling_mean(source, window=2, min_periods=1)
    graph.compute()

    assert values(graph.output.data)[("2024-01-02", "a")] == 2.0
    assert values(graph.output.data)[("2024-01-02", "b")] == 15.0


def test_lag_uses_daily_domain_sessions_for_sparse_monthly_inputs() -> None:
    domain = Domain(
        calendar=["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"],
        universe=["a"],
    )
    source = Panel.from_domain(
        pl.DataFrame(
            {"time": ["2024-01-31"], "asset_id": ["a"], "value": [1.0]}
        ),
        domain,
        name="monthly_input",
    )

    delayed = lag(source, periods=1)
    delayed.compute()

    assert delayed.output.data.to_dicts() == [
        {"time": date(2024, 1, 31), "asset_id": "a", "value": None},
        {"time": date(2024, 2, 1), "asset_id": "a", "value": 1.0},
        {"time": date(2024, 2, 2), "asset_id": "a", "value": None},
        {"time": date(2024, 2, 5), "asset_id": "a", "value": None},
    ]


def test_zscore_constant_cross_section_returns_null_or_nan() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 1.0),
        ]
    )

    graph = zscore(source)
    graph.compute()

    assert (
        graph.output.data["value"].null_count()
        + graph.output.data["value"].is_nan().sum()
        == 2
    )
