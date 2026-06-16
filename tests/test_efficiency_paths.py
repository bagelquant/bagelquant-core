from __future__ import annotations

import math

import polars as pl

from bagelquant_core import ExecutionRuntime
from bagelquant_core.composer import (
    orthogonalize,
    rolling_corr,
    rolling_cov,
    rolling_ols,
)
from bagelquant_core.transformer import ewm_mean, rolling_percentile, rolling_rank
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


def test_ewm_mean_uses_recursive_weighting() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", 3.0),
        ]
    )

    graph = ewm_mean(source, alpha=0.5, min_periods=0)
    graph.compute()

    result = values(graph.output.data)
    assert result[("2024-01-01", "a")] == 1.0
    assert result[("2024-01-02", "a")] == 1.5
    assert result[("2024-01-03", "a")] == 2.25


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


def test_rolling_ols_uses_closed_form_slope() -> None:
    target = panel(
        [
            ("2024-01-01", "a", 3.0),
            ("2024-01-02", "a", 5.0),
            ("2024-01-03", "a", 7.0),
        ],
        name="target",
    )
    factor = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", 3.0),
        ],
        name="factor",
    )

    graph = rolling_ols(target, factor, window=3)
    graph.compute()

    assert math.isclose(values(graph.output.data)[("2024-01-03", "a")], 2.0)


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

    graph = orthogonalize(target, factor)
    graph.compute()

    assert all(math.isclose(value, 0.0, abs_tol=1e-12) for value in values(graph.output.data).values())
