from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, ExecutionRuntime, Graph, Panel, pct_change_frame
from bagelquant_core.transformer import (
    diff_from_last_change,
    kelly,
    kelly_nonan_standardize,
    kelly_rank_boxcox,
    kelly_rescaling_weight,
    lag,
    pct_change_from_last_change,
    rank,
    repeat_count,
    rolling_ewm_fw,
    rolling_mean,
    smooth,
    streak_count,
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

    assert values(graph.output.collect(dense=True)) == {
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

    assert values(graph.output.collect(dense=True))[("2024-01-02", "a")] == 2.0
    assert values(graph.output.collect(dense=True))[("2024-01-02", "b")] == 15.0


def test_rolling_ewm_fw_uses_only_the_configured_trailing_window() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", 4.0),
            ("2024-01-04", "a", 8.0),
        ]
    )

    graph = rolling_ewm_fw(
        source,
        window=2,
        halflife=1.0,
        min_periods=2,
    )
    actual = graph.compute().collect(dense=True)
    eager = rolling_ewm_fw.operation(
        source.collect(dense=True),
        window=2,
        halflife=1.0,
        min_periods=2,
    )

    assert_frame_equal(actual, eager)
    result = values(actual)
    assert result[("2024-01-01", "a")] is None
    assert result[("2024-01-03", "a")] == pytest.approx(10.0 / 3.0)
    assert result[("2024-01-04", "a")] == pytest.approx(20.0 / 3.0)

    default_periods = rolling_ewm_fw(
        source,
        window=2,
        halflife=1.0,
    ).compute()
    assert values(default_periods.collect(dense=True))[("2024-01-01", "a")] == 1.0


def test_smooth_is_the_fixed_daily_rolling_ewm_preset() -> None:
    rows = [
        (f"2024-01-{day:02d}", asset, value)
        for asset, series in (
            (
                "a",
                [1000.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ),
            (
                "b",
                [2.0, 4.0, None, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            ),
        )
        for day, value in enumerate(series, start=1)
    ]
    source = panel(rows)
    runtime = ExecutionRuntime()

    actual = smooth(source).compute(runtime=runtime).collect(dense=True)
    direct = smooth.operation(source.collect(dense=True))
    expected = rolling_ewm_fw.operation(
        source.collect(dense=True),
        window=10,
        halflife=3.0,
        min_periods=5,
    )

    assert_frame_equal(actual, direct)
    assert_frame_equal(actual, expected)
    assert runtime.eager_barriers == 0
    result = values(actual)
    assert result[("2024-01-04", "a")] is None
    assert result[("2024-01-05", "a")] is not None
    assert result[("2024-01-05", "b")] is None
    assert result[("2024-01-06", "b")] is not None

    changed = panel(
        [
            (
                time,
                asset,
                1.0 if time == "2024-01-01" and asset == "a" else value,
            )
            for time, asset, value in rows
        ]
    )
    changed_result = values(smooth(changed).compute().collect(dense=True))
    assert result[("2024-01-10", "a")] != changed_result[("2024-01-10", "a")]
    assert result[("2024-01-11", "a")] == pytest.approx(
        changed_result[("2024-01-11", "a")]
    )

    node = smooth(source).spec().to_dict()["nodes"][-1]
    assert node["config"] == {
        "transformer": "bagelquant_core.transformer.rolling.smooth"
    }


def test_smooth_window_counts_missing_domain_sessions() -> None:
    times = [f"2024-02-{day:02d}" for day in range(1, 12)]
    domain = Domain(calendar=times, universe=["a"])
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [times[0], *times[2:]],
                "asset_id": ["a"] * 10,
                "value": [100.0, *([1.0] * 9)],
            }
        ),
        domain,
        name="source",
    )

    result = smooth(source).compute().collect(dense=True)

    assert result[-1, "value"] == pytest.approx(1.0)


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

    assert delayed.output.collect(dense=True).to_dicts() == [
        {"time": date(2024, 1, 31), "asset_id": "a", "value": None},
        {"time": date(2024, 2, 1), "asset_id": "a", "value": 1.0},
        {"time": date(2024, 2, 2), "asset_id": "a", "value": None},
        {"time": date(2024, 2, 5), "asset_id": "a", "value": None},
    ]


def test_repeat_and_change_transformers_reset_at_missing_values() -> None:
    times = [f"2024-01-{day:02d}" for day in range(1, 9)]
    domain = Domain(calendar=times, universe=["a", "b"])
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": times * 2,
                "asset_id": ["a"] * 8 + ["b"] * 8,
                "value": [
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    None,
                    2.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    2.0,
                ],
            }
        ),
        domain,
        name="input",
    )

    repeated = repeat_count(source).compute().collect(dense=True)
    changed = diff_from_last_change(source).compute().collect(dense=True)
    pct_changed = pct_change_from_last_change(source).compute().collect(dense=True)

    assert repeated.filter(pl.col("asset_id") == "a")["value"].to_list() == [
        1,
        2,
        3,
        1,
        None,
        1,
        2,
        1,
    ]
    assert changed.filter(pl.col("asset_id") == "a")["value"].to_list() == [
        None,
        None,
        None,
        1.0,
        None,
        None,
        None,
        1.0,
    ]
    pct_b = pct_changed.filter(pl.col("asset_id") == "b")["value"].to_list()
    assert pct_b[:2] == [None, None]
    assert pct_b[2] == float("inf")
    assert pct_b[3:6] == [None, -1.0, None]
    assert pct_b[6] == float("inf")
    assert pct_b[7] is None
    assert repeated.schema["value"] == pl.Int64


def test_streak_count_reversals_equal_values_and_missing_values() -> None:
    times = [f"2024-01-{day:02d}" for day in range(1, 10)]
    domain = Domain(calendar=times, universe=["a"])
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": times,
                "asset_id": ["a"] * 9,
                "value": [1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 0.0, None, -1.0],
            }
        ),
        domain,
        name="input",
    )

    resetting = streak_count(source).compute().collect(dense=True)
    holding = streak_count(
        source, reset_on_equal=False, name="holding"
    ).compute().collect(dense=True)

    assert resetting["value"].to_list() == [0, 1, 2, 0, -1, 0, -1, None, 0]
    assert holding["value"].to_list() == [0, 1, 2, 0, -1, -1, -2, None, 0]
    assert resetting.schema["value"] == pl.Int64


def test_streak_plan_matches_direct_execution_and_compiled_graph() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", 3.0),
            ("2024-01-04", "a", 3.0),
            ("2024-01-05", "a", 4.0),
        ],
        name="input",
    )
    dense = source.collect(dense=True).sort(["asset_id", "time"], descending=True)
    expected = streak_count.operation(dense, reset_on_equal=False)
    graph = streak_count(source, reset_on_equal=False, name="streak")
    actual = graph.compute().collect(dense=True)
    compiled = Graph.compile(graph.spec())
    restored = compiled.compute({"input": source}).collect(dense=True)

    assert_frame_equal(actual, expected)
    assert_frame_equal(restored, expected)
    assert actual["value"].to_list() == [0, 1, 2, 2, 3]


def test_streak_count_requires_a_boolean_reset_flag() -> None:
    source = panel([("2024-01-01", "a", 1.0)])

    with pytest.raises(TypeError, match="must be a boolean"):
        streak_count(source, reset_on_equal=1).compute()


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
        graph.output.collect(dense=True)["value"].null_count()
        + graph.output.collect(dense=True)["value"].is_nan().sum()
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

    expected = operation.operation(source.collect(dense=True), **parameters)
    actual = operation(source, **parameters).compute().collect(dense=True)

    assert_frame_equal(actual, expected)
