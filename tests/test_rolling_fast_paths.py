from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import rolling_ols
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

    expected_keys = domain.grid_lazy().collect().sort(["time", "asset_id"])
    for output in (ranked, scored, regressed):
        collected = output.collect(include_traces=True)
        assert collected.select("time", "asset_id").equals(expected_keys)
        assert collected.get_column("base_available_date").to_list() == [
            days[0],
            days[1],
            days[3],
            days[4],
        ]
