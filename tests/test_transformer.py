from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, Panel, pct_change_frame
from bagelquant_core.transformer import (
    kelly,
    kelly_nonan_standardize,
    kelly_rank_boxcox,
    kelly_rescaling_weight,
    lag,
    rank,
    rolling_mean,
    zscore,
)

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


def test_pct_change_frame_supports_runtime_consumers_without_a_graph() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 2.0),
            ("2024-01-02", "a", 3.0),
            ("2024-01-01", "b", 10.0),
            ("2024-01-02", "b", 8.0),
        ]
    ).compute()

    result = pct_change_frame(source)
    result_values = values(result)

    assert result_values[("2024-01-01", "a")] is None
    assert result_values[("2024-01-01", "b")] is None
    assert result_values[("2024-01-02", "a")] == pytest.approx(0.5)
    assert result_values[("2024-01-02", "b")] == pytest.approx(-0.2)


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


@pytest.mark.parametrize(
    ("operation", "parameters"),
    [
        (kelly, {"window": 3}),
        (kelly_nonan_standardize, {"window": 3}),
        (kelly_rank_boxcox, {"window": 3, "lambda_": 0.5}),
        (kelly_rescaling_weight, {"window": 3}),
    ],
)
def test_kelly_private_plan_matches_public_operation(
    operation,
    parameters: dict[str, float | int],
) -> None:
    times = [f"2024-01-{day:02d}" for day in range(1, 8)]
    assets = ["a", "b", "c"]
    domain = Domain(calendar=times, universe=assets)
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [time for time in times for _ in assets],
                "asset_id": assets * len(times),
                "value": [
                    1.0,
                    2.0,
                    None,
                    2.0,
                    -1.0,
                    3.0,
                    4.0,
                    0.0,
                    2.0,
                    None,
                    5.0,
                    1.0,
                    3.0,
                    2.0,
                    6.0,
                    5.0,
                    4.0,
                    -2.0,
                    7.0,
                    6.0,
                    8.0,
                ],
            }
        ),
        domain,
    )

    expected = operation.operation(source.data, **parameters)
    actual = operation(source, **parameters).compute().data

    assert_frame_equal(actual, expected)
