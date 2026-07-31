from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import (
    rolling_elastic_net,
    rolling_lasso,
    rolling_ols,
    rolling_ridge,
)
from bagelquant_core.transformer import (
    rolling_percentile,
    rolling_rank,
    rolling_zscore,
)


def _static_panel(
    values: list[tuple[str, str, float | None]],
    *,
    name: str,
) -> Panel:
    calendar = sorted({time for time, _, _ in values})
    assets = sorted({asset for _, asset, _ in values})
    present = [row for row in values if row[2] is not None]
    return Panel.from_domain(
        pl.DataFrame(
            present,
            schema=["time", "asset_id", "value"],
            orient="row",
        ),
        Domain(calendar=calendar, universe=assets),
        name=name,
    )


def _dense_values(panel: Panel) -> np.ndarray:
    return (
        panel.data.sort(["asset_id", "time"])
        .get_column("value")
        .to_numpy()
        .astype(float, copy=False)
    )


def _rolling_rank_reference(
    values: np.ndarray,
    *,
    group_size: int,
    window: int,
    min_periods: int,
    pct: bool,
) -> np.ndarray:
    output = np.full(len(values), np.nan)
    for group_start in range(0, len(values), group_size):
        group = values[group_start : group_start + group_size]
        for index in range(len(group)):
            sample = group[max(0, index - window + 1) : index + 1]
            sample = sample[~np.isnan(sample)]
            if len(sample) < min_periods or len(sample) == 0:
                continue
            less = np.sum(sample < sample[-1])
            equal = np.sum(sample == sample[-1])
            rank = float(less + (equal + 1.0) / 2.0)
            output[group_start + index] = rank / len(sample) if pct else rank
    return output


def _rolling_zscore_reference(
    values: np.ndarray,
    *,
    group_size: int,
    window: int,
    min_periods: int,
    ddof: int,
) -> np.ndarray:
    output = np.full(len(values), np.nan)
    for group_start in range(0, len(values), group_size):
        group = values[group_start : group_start + group_size]
        for index in range(len(group)):
            sample = group[max(0, index - window + 1) : index + 1]
            sample = sample[~np.isnan(sample)]
            if len(sample) < min_periods or len(sample) <= ddof:
                continue
            std = sample.std(ddof=ddof)
            if std != 0:
                output[group_start + index] = (
                    sample[-1] - sample.mean()
                ) / std
    return output


@pytest.mark.parametrize("min_periods", [0, 1, 3])
def test_rolling_rank_fast_paths_match_reference(min_periods: int) -> None:
    source = _static_panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", None),
            ("2024-01-04", "a", 2.0),
            ("2024-01-05", "a", 5.0),
            ("2024-01-01", "b", None),
            ("2024-01-02", "b", None),
            ("2024-01-03", "b", 4.0),
            ("2024-01-04", "b", 4.0),
            ("2024-01-05", "b", 3.0),
        ],
        name="source",
    )
    values = _dense_values(source)

    ranked = rolling_rank(
        source,
        window=3,
        min_periods=min_periods,
    ).compute()
    percentile = rolling_percentile(
        source,
        window=3,
        min_periods=min_periods,
    ).compute()

    np.testing.assert_array_equal(
        _dense_values(ranked),
        _rolling_rank_reference(
            values,
            group_size=5,
            window=3,
            min_periods=min_periods,
            pct=False,
        ),
    )
    np.testing.assert_array_equal(
        _dense_values(percentile),
        _rolling_rank_reference(
            values,
            group_size=5,
            window=3,
            min_periods=min_periods,
            pct=True,
        ),
    )


def test_rolling_rank_randomized_groups_match_reference_exactly() -> None:
    rng = np.random.default_rng(90210)
    group_size = 70
    assets = ["a", "b", "c", "d"]
    dates = [
        str(date(2024, 3, 1) + timedelta(days=offset))
        for offset in range(group_size)
    ]
    raw = rng.integers(-3, 4, size=group_size * len(assets)).astype(float)
    raw[rng.choice(len(raw), size=35, replace=False)] = np.nan
    source = _static_panel(
        [
            (time, asset, None if np.isnan(value) else float(value))
            for (asset, time), value in zip(
                (
                    (asset, time)
                    for asset in assets
                    for time in dates
                ),
                raw,
                strict=True,
            )
        ],
        name="random_rank",
    )
    dense = _dense_values(source)

    actual_rank = rolling_rank(
        source,
        window=17,
        min_periods=5,
    ).compute()
    actual_pct = rolling_percentile(
        source,
        window=17,
        min_periods=5,
    ).compute()

    np.testing.assert_array_equal(
        _dense_values(actual_rank),
        _rolling_rank_reference(
            dense,
            group_size=group_size,
            window=17,
            min_periods=5,
            pct=False,
        ),
    )
    np.testing.assert_array_equal(
        _dense_values(actual_pct),
        _rolling_rank_reference(
            dense,
            group_size=group_size,
            window=17,
            min_periods=5,
            pct=True,
        ),
    )


@pytest.mark.parametrize(("min_periods", "ddof"), [(0, 0), (1, 1), (3, 1)])
def test_rolling_zscore_native_path_matches_reference(
    min_periods: int,
    ddof: int,
) -> None:
    source = _static_panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", None),
            ("2024-01-04", "a", 4.0),
            ("2024-01-05", "a", 4.0),
            ("2024-01-01", "b", None),
            ("2024-01-02", "b", None),
            ("2024-01-03", "b", 3.0),
            ("2024-01-04", "b", 3.0),
            ("2024-01-05", "b", 3.0),
        ],
        name="source",
    )
    expected = _rolling_zscore_reference(
        _dense_values(source),
        group_size=5,
        window=3,
        min_periods=min_periods,
        ddof=ddof,
    )

    actual = rolling_zscore(
        source,
        window=3,
        min_periods=min_periods,
        ddof=ddof,
    ).compute()

    np.testing.assert_allclose(
        _dense_values(actual),
        expected,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def _rolling_ols_reference(
    target: np.ndarray,
    factors: np.ndarray,
    *,
    group_size: int,
    window: int,
) -> np.ndarray:
    output = np.full(len(target), np.nan)
    for group_start in range(0, len(target), group_size):
        group_y = target[group_start : group_start + group_size]
        group_x = factors[group_start : group_start + group_size]
        for current in range(window, group_size):
            if not np.isfinite(group_x[current]).all():
                continue
            train_y = group_y[current - window : current]
            train_x = group_x[current - window : current]
            valid = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
            if not valid.any():
                continue
            design = np.column_stack(
                [np.ones(valid.sum()), train_x[valid]]
            )
            coefficients = np.linalg.lstsq(
                design,
                train_y[valid],
                rcond=None,
            )[0]
            output[group_start + current] = (
                np.r_[1.0, group_x[current]] @ coefficients
            )
    return output


def _rolling_regularized_reference(
    target: np.ndarray,
    factors: np.ndarray,
    *,
    group_size: int,
    window: int,
    method: str,
    alpha: float,
    l1_ratio: float = 0.0,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    output = np.full(len(target), np.nan)
    for group_start in range(0, len(target), group_size):
        group_y = target[group_start : group_start + group_size]
        group_x = factors[group_start : group_start + group_size]
        for current in range(window, group_size):
            if not np.isfinite(group_x[current]).all():
                continue
            train_y = group_y[current - window : current]
            train_x = group_x[current - window : current]
            valid = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
            if not valid.any():
                continue
            design = np.column_stack(
                [np.ones(valid.sum()), train_x[valid]]
            )
            selected_y = train_y[valid]
            if method == "ridge":
                penalty = np.eye(design.shape[1])
                penalty[0, 0] = 0.0
                coefficients = (
                    np.linalg.pinv(
                        design.T @ design + alpha * penalty
                    )
                    @ design.T
                    @ selected_y
                )
            else:
                coefficients = np.zeros(design.shape[1])
                coefficients[0] = selected_y.mean()
                for _ in range(max_iter):
                    previous = coefficients.copy()
                    for index in range(design.shape[1]):
                        residual = (
                            selected_y
                            - design @ coefficients
                            + design[:, index] * coefficients[index]
                        )
                        numerator = design[:, index] @ residual
                        if index == 0:
                            coefficients[index] = numerator / (
                                design[:, index] @ design[:, index]
                            )
                            continue
                        denominator = (
                            design[:, index] @ design[:, index]
                            + alpha * (1.0 - l1_ratio)
                        )
                        coefficients[index] = (
                            np.sign(numerator)
                            * max(
                                abs(numerator) - alpha * l1_ratio,
                                0.0,
                            )
                            / denominator
                        )
                    if (
                        np.max(np.abs(coefficients - previous))
                        <= tolerance
                    ):
                        break
            output[group_start + current] = (
                np.r_[1.0, group_x[current]] @ coefficients
            )
    return output


def test_rolling_ols_batched_path_matches_lstsq_reference() -> None:
    rng = np.random.default_rng(42)
    group_size = 18
    dates = [
        str(date(2024, 1, 1) + timedelta(days=offset))
        for offset in range(group_size)
    ]
    first_values = rng.normal(size=group_size * 2)
    second_values = first_values * (1.0 + 1e-14)
    second_values += rng.normal(scale=1e-15, size=group_size * 2)
    target_values = (
        1.0
        + 2.0 * first_values
        - 0.5 * second_values
        + rng.normal(scale=0.01, size=group_size * 2)
    )
    target_values[[1, 4, 20, 21]] = np.nan
    first_values[[3, 11, 24]] = np.nan
    second_values[[3, 14, 24]] = np.nan
    rows = [
        (dates[index], asset)
        for asset in ("a", "b")
        for index in range(group_size)
    ]

    def build(values: np.ndarray, name: str) -> Panel:
        return _static_panel(
            [
                (time, asset, None if np.isnan(value) else float(value))
                for (time, asset), value in zip(rows, values, strict=True)
            ],
            name=name,
        )

    target = build(target_values, "target")
    first = build(first_values, "first")
    second = build(second_values, "second")
    expected = _rolling_ols_reference(
        _dense_values(target),
        np.column_stack(
            [_dense_values(first), _dense_values(second)]
        ),
        group_size=group_size,
        window=5,
    )

    actual = rolling_ols(target, first, second, window=5).compute()

    np.testing.assert_allclose(
        _dense_values(actual),
        expected,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize("factor_count", [2, 4, 8])
def test_rolling_ols_multi_factor_gram_path_matches_reference(
    factor_count: int,
) -> None:
    rng = np.random.default_rng(100 + factor_count)
    group_size = 32
    dates = [
        str(date(2024, 1, 1) + timedelta(days=offset))
        for offset in range(group_size)
    ]
    rows = [
        (time, asset)
        for asset in ("a", "b")
        for time in dates
    ]
    factor_values = rng.normal(
        size=(group_size * 2, factor_count)
    )
    target_values = (
        2.0
        + factor_values
        @ np.linspace(-1.0, 1.0, factor_count)
        + rng.normal(scale=0.01, size=group_size * 2)
    )
    target_values[[3, 35]] = np.nan
    factor_values[7, 0] = np.nan
    factor_values[44, -1] = np.nan

    def build(values: np.ndarray, name: str) -> Panel:
        return _static_panel(
            [
                (time, asset, None if np.isnan(value) else float(value))
                for (time, asset), value in zip(rows, values, strict=True)
            ],
            name=name,
        )

    target = build(target_values, "target")
    factors = [
        build(factor_values[:, index], f"factor_{index}")
        for index in range(factor_count)
    ]
    expected = _rolling_ols_reference(
        _dense_values(target),
        np.column_stack([_dense_values(factor) for factor in factors]),
        group_size=group_size,
        window=12,
    )

    actual = rolling_ols(target, *factors, window=12).compute()

    np.testing.assert_allclose(
        _dense_values(actual),
        expected,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    (
        "operation",
        "method",
        "alpha",
        "l1_ratio",
        "max_iter",
        "tolerance",
    ),
    [
        (rolling_ridge, "ridge", 0.0, 0.0, 80, 1e-10),
        (rolling_ridge, "ridge", 10.0, 0.0, 80, 1e-10),
        (rolling_lasso, "elastic", 0.7, 1.0, 1, 0.0),
        (rolling_elastic_net, "elastic", 0.7, 0.35, 80, 1e-10),
    ],
)
@pytest.mark.parametrize("factor_count", [1, 2, 8])
def test_regularized_rolling_paths_match_row_reference(
    operation,
    method: str,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
    factor_count: int,
) -> None:
    rng = np.random.default_rng(86)
    group_size = 28
    dates = [
        str(date(2024, 2, 1) + timedelta(days=offset))
        for offset in range(group_size)
    ]
    rows = [
        (time, asset)
        for asset in ("a", "b")
        for time in dates
    ]
    factors_raw = rng.normal(size=(group_size * 2, factor_count))
    target_raw = (
        1.5
        + factors_raw
        @ np.linspace(0.8, -0.3, factor_count)
        + rng.normal(scale=0.03, size=group_size * 2)
    )
    target_raw[[2, 31]] = np.nan
    factors_raw[8, 0] = np.nan
    factors_raw[46, -1] = np.nan

    def build(values: np.ndarray, name: str) -> Panel:
        return _static_panel(
            [
                (time, asset, None if np.isnan(value) else float(value))
                for (time, asset), value in zip(rows, values, strict=True)
            ],
            name=name,
        )

    target = build(target_raw, "target")
    factors = [
        build(factors_raw[:, index], f"factor_{index}")
        for index in range(factor_count)
    ]
    expected = _rolling_regularized_reference(
        _dense_values(target),
        np.column_stack([_dense_values(factor) for factor in factors]),
        group_size=group_size,
        window=10,
        method=method,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    config = {
        "window": 10,
        "alpha": alpha,
    }
    if method == "elastic":
        config.update(max_iter=max_iter, tolerance=tolerance)
        if operation is rolling_elastic_net:
            config["l1_ratio"] = l1_ratio

    actual = operation(target, *factors, **config).compute()

    np.testing.assert_allclose(
        _dense_values(actual),
        expected,
        rtol=1e-9,
        atol=1e-11,
        equal_nan=True,
    )


@pytest.mark.parametrize("seed", [7, 42, 2024])
def test_rolling_ols_single_factor_path_matches_lstsq_reference(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    group_size = 40
    dates = [
        str(date(2024, 1, 1) + timedelta(days=offset))
        for offset in range(group_size)
    ]
    factor_values = rng.normal(size=group_size * 2)
    target_values = (
        3.0
        - 1.25 * factor_values
        + rng.normal(scale=0.05, size=group_size * 2)
    )
    target_values[[2, 9, 41, 68]] = np.nan
    factor_values[[5, 18, 47, 72]] = np.nan
    rows = [
        (time, asset)
        for asset in ("a", "b")
        for time in dates
    ]

    def build(values: np.ndarray, name: str) -> Panel:
        return _static_panel(
            [
                (time, asset, None if np.isnan(value) else float(value))
                for (time, asset), value in zip(rows, values, strict=True)
            ],
            name=name,
        )

    target = build(target_values, "target")
    factor = build(factor_values, "factor")
    expected = _rolling_ols_reference(
        _dense_values(target),
        _dense_values(factor)[:, None],
        group_size=group_size,
        window=12,
    )

    actual = rolling_ols(target, factor, window=12).compute()

    np.testing.assert_allclose(
        _dense_values(actual),
        expected,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "factor_values",
    [
        [2.0] * 8,
        [1e8 + offset * 1e-7 for offset in range(8)],
    ],
)
def test_rolling_ols_single_factor_falls_back_for_unstable_windows(
    factor_values: list[float],
) -> None:
    target_values = np.array(
        [4.0, 5.0, np.nan, 8.0, 9.0, 11.0, 13.0, 15.0]
    )
    target = _static_panel(
        [
            (f"2024-01-{index + 1:02d}", "a", value)
            for index, value in enumerate(target_values)
        ],
        name="target",
    )
    factor = _static_panel(
        [
            (f"2024-01-{index + 1:02d}", "a", value)
            for index, value in enumerate(factor_values)
        ],
        name="factor",
    )
    expected = _rolling_ols_reference(
        _dense_values(target),
        _dense_values(factor)[:, None],
        group_size=8,
        window=4,
    )

    actual = rolling_ols(target, factor, window=4).compute()

    np.testing.assert_allclose(
        _dense_values(actual),
        expected,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_rolling_ols_preserves_invalid_window_and_current_factor_nulls() -> None:
    target = _static_panel(
        [
            ("2024-01-01", "a", None),
            ("2024-01-02", "a", None),
            ("2024-01-03", "a", None),
            ("2024-01-04", "a", 7.0),
            ("2024-01-05", "a", 9.0),
            ("2024-01-06", "a", 11.0),
            ("2024-01-07", "a", 13.0),
        ],
        name="target",
    )
    factor = _static_panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", 3.0),
            ("2024-01-04", "a", 4.0),
            ("2024-01-05", "a", None),
            ("2024-01-06", "a", 6.0),
            ("2024-01-07", "a", 7.0),
        ],
        name="factor",
    )
    expected = _rolling_ols_reference(
        _dense_values(target),
        _dense_values(factor)[:, None],
        group_size=7,
        window=3,
    )

    actual = rolling_ols(target, factor, window=3).compute()

    assert np.isnan(_dense_values(actual)[3])
    assert np.isnan(_dense_values(actual)[4])
    np.testing.assert_allclose(
        _dense_values(actual),
        expected,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_fast_paths_preserve_dynamic_membership_and_traces() -> None:
    days = [date(2024, 1, day) for day in range(1, 6)]
    membership = pl.DataFrame(
        {
            "time": [days[0], days[1], days[3], days[4]],
            "asset_id": ["a"] * 4,
            "active": [True] * 4,
        }
    )
    domain = Domain(calendar=days, universe=membership)
    frame = pl.DataFrame(
        {
            "time": [days[0], days[1], days[3], days[4]],
            "asset_id": ["a"] * 4,
            "value": [1.0, 2.0, 2.0, 4.0],
            "observation_date": [days[0], days[1], days[3], days[4]],
            "base_available_date": [days[0], days[1], days[3], days[4]],
        }
    )
    source = Panel.from_domain(
        frame,
        domain,
        name="source",
        trace_columns=("observation_date", "base_available_date"),
    )

    ranked = rolling_rank(
        source,
        window=2,
        min_periods=1,
    ).compute()
    scored = rolling_zscore(
        source,
        window=2,
        min_periods=1,
        ddof=0,
    ).compute()
    regressed = rolling_ols(source, source, window=2).compute()
    regularized = rolling_lasso(
        source,
        source,
        window=2,
        max_iter=1,
        tolerance=0.0,
    ).compute()

    expected_keys = domain.grid_lazy().collect().sort(["time", "asset_id"])
    for output in (ranked, scored, regressed, regularized):
        collected = output.collect(include_traces=True)
        assert collected.select("time", "asset_id").equals(expected_keys)
        assert collected.get_column("base_available_date").to_list() == [
            days[0],
            days[1],
            days[3],
            days[4],
        ]
