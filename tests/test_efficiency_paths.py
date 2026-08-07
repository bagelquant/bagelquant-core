from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import bagelquant_core.transformer.rolling as rolling_module

from bagelquant_core import ExecutionRuntime, Graph
from bagelquant_core.composer import rolling_corr, rolling_cov
from bagelquant_core.transformer import (
    ewm_mean,
    orthogonalize,
    rolling_ols,
    rolling_percentile,
    rolling_rank,
)
from bagelquant_core.transformer.core import transformer

from helpers import panel, values


def test_panel_data_remains_defensive() -> None:
    source = panel([("2024-01-01", "a", 1.0)])

    changed = source.data.with_columns(pl.lit(99.0).alias("value"))

    assert changed["value"].to_list() == [99.0]
    assert source.data["value"].to_list() == [1.0]


def test_reusable_execution_runtime_hits_cache() -> None:
    calls = {"count": 0}

    @transformer
    def counted(frame: pl.DataFrame) -> pl.DataFrame:
        calls["count"] += 1
        return frame

    source = panel([("2024-01-01", "a", 1.0)])
    graph = counted(source, name="counted")
    runtime = ExecutionRuntime()

    graph.compute(runtime=runtime)
    graph.compute(runtime=runtime)

    assert calls["count"] == 1


def test_rolling_rank_and_percentile_fast_paths() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 2.0),
            ("2024-01-02", "a", 1.0),
            ("2024-01-03", "a", 3.0),
        ]
    )

    ranked = rolling_rank(source, window=2, min_periods=1)
    pct = rolling_percentile(source, window=2, min_periods=1)
    ranked.compute()
    pct.compute()

    assert values(ranked.output.data)[("2024-01-03", "a")] == 2.0
    assert values(pct.output.data)[("2024-01-02", "a")] == 0.5


def test_rank_and_percentile_siblings_share_comparison_kernel(
    monkeypatch,
) -> None:
    source = panel(
        [
            ("2024-01-01", "a", 2.0),
            ("2024-01-02", "a", 1.0),
            ("2024-01-03", "a", 3.0),
        ]
    )
    original = rolling_module._rolling_last_rank_pair
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        rolling_module,
        "_rolling_last_rank_pair",
        counted,
    )
    runtime = ExecutionRuntime()

    Graph(
        outputs=[
            rolling_rank(source, window=2, min_periods=1, name="rank"),
            rolling_percentile(
                source,
                window=2,
                min_periods=1,
                name="percentile",
            ),
        ]
    ).compute(runtime=runtime)

    assert calls["count"] == 1
    assert runtime.eager_barriers == 2
    assert runtime.materializations == 2


def test_ewm_mean_honors_adjusted_and_recursive_weighting() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", 3.0),
        ]
    )

    adjusted = ewm_mean(source, alpha=0.5, min_periods=0)
    recursive = ewm_mean(source, alpha=0.5, min_periods=0, adjust=False)
    adjusted.compute()
    recursive.compute()

    adjusted_values = values(adjusted.output.data)
    recursive_values = values(recursive.output.data)
    assert adjusted_values[("2024-01-01", "a")] == 1.0
    assert adjusted_values[("2024-01-02", "a")] == pytest.approx(5 / 3)
    assert adjusted_values[("2024-01-03", "a")] == pytest.approx(17 / 7)
    assert recursive_values[("2024-01-02", "a")] == 1.5
    assert recursive_values[("2024-01-03", "a")] == 2.25


def test_rolling_pair_composers_are_grouped_by_asset() -> None:
    left = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-01", "b", 10.0),
            ("2024-01-02", "b", 20.0),
        ],
        name="left",
    )
    right = panel(
        [
            ("2024-01-01", "a", 2.0),
            ("2024-01-02", "a", 4.0),
            ("2024-01-01", "b", 5.0),
            ("2024-01-02", "b", 10.0),
        ],
        name="right",
    )

    corr = rolling_corr(left, right, window=2)
    cov = rolling_cov(left, right, window=2)
    corr.compute()
    cov.compute()

    assert math.isclose(values(corr.output.data)[("2024-01-02", "a")], 1.0)
    assert math.isclose(values(cov.output.data)[("2024-01-02", "a")], 1.0)


def test_rolling_ols_predicts_current_value_from_prior_window() -> None:
    target = panel(
        [
            ("2024-01-01", "a", 3.0),
            ("2024-01-02", "a", 5.0),
            ("2024-01-03", "a", 7.0),
            ("2024-01-04", "a", 9.0),
        ],
        name="target",
    )
    factor = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", 3.0),
            ("2024-01-04", "a", 4.0),
        ],
        name="factor",
    )

    graph = rolling_ols(target, factors=(factor,), window=3)
    graph.compute()

    result = values(graph.output.data)
    assert result[("2024-01-03", "a")] is None
    assert math.isclose(result[("2024-01-04", "a")], 9.0)


def test_one_factor_orthogonalize_uses_closed_form_residuals() -> None:
    target = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 3.0),
            ("2024-01-01", "c", 5.0),
        ],
        name="target",
    )
    factor = panel(
        [
            ("2024-01-01", "a", 0.0),
            ("2024-01-01", "b", 1.0),
            ("2024-01-01", "c", 2.0),
        ],
        name="factor",
    )

    graph = orthogonalize(
        target, factors=(factor,), fit_intercept=True
    )
    graph.compute()

    assert all(math.isclose(value, 0.0, abs_tol=1e-12) for value in values(graph.output.data).values())


def test_orthogonalize_defaults_to_no_intercept() -> None:
    target = panel(
        [
            ("2024-01-01", "a", 3.0),
            ("2024-01-01", "b", 5.0),
            ("2024-01-01", "c", 7.0),
        ],
        name="target",
    )
    factor = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 2.0),
            ("2024-01-01", "c", 3.0),
        ],
        name="factor",
    )

    without_intercept = orthogonalize(target, factors=(factor,)).compute()
    with_intercept = orthogonalize(
        target, factors=(factor,), fit_intercept=True
    ).compute()

    assert any(
        not math.isclose(value, 0.0, abs_tol=1e-12)
        for value in values(without_intercept.data).values()
    )
    assert all(
        math.isclose(value, 0.0, abs_tol=1e-12)
        for value in values(with_intercept.data).values()
    )


def test_multi_factor_orthogonalize_matches_lstsq_reference() -> None:
    rng = np.random.default_rng(11)
    times = ["2024-01-01"] * 3 + ["2024-01-02"] * 8
    assets = [f"a{index}" for index in range(3)] + [
        f"b{index}" for index in range(8)
    ]
    first = rng.normal(size=len(times))
    second = rng.normal(size=len(times))
    target = 1.0 + 0.7 * first - 1.2 * second
    target += rng.normal(scale=0.02, size=len(times))
    target[5] = np.nan
    first[8] = np.nan

    def frame(values_: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": times[::-1],
                "asset_id": assets[::-1],
                "value": values_[::-1],
            }
        )

    actual = orthogonalize.operation(
        frame(target),
        factors=(frame(first), frame(second)),
        fit_intercept=True,
    )
    expected = np.full(len(times), np.nan)
    for time in sorted(set(times)):
        indices = np.flatnonzero(np.array(times) == time)
        group_y = target[indices]
        group_x = np.column_stack([first[indices], second[indices]])
        valid = np.isfinite(group_y) & np.isfinite(group_x).all(axis=1)
        if valid.sum() <= 2:
            continue
        design = np.column_stack([np.ones(valid.sum()), group_x[valid]])
        coefficients = np.linalg.lstsq(
            design,
            group_y[valid],
            rcond=None,
        )[0]
        expected[indices[valid]] = group_y[valid] - design @ coefficients

    actual_values = (
        actual.sort(["time", "asset_id"])
        .get_column("value")
        .to_numpy()
        .astype(float, copy=False)
    )
    expected_order = np.lexsort((np.array(assets), np.array(times)))
    np.testing.assert_allclose(
        actual_values,
        expected[expected_order],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
