"""Cross-sectional normalization transforms."""

from __future__ import annotations

import polars as pl

from ..frame import TIME, VALUE, cross_section_rank, panel_like
from .core import _expression_plan, transformer


@transformer
def rank(frame: pl.DataFrame) -> pl.DataFrame:
    return cross_section_rank(frame, pct=True)


@transformer
def zscore(frame: pl.DataFrame) -> pl.DataFrame:
    std = pl.col(VALUE).std(ddof=1).over(TIME)
    return panel_like(frame, (pl.col(VALUE) - pl.col(VALUE).mean().over(TIME)) / std)


@transformer
def winsorize(
    frame: pl.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    _validate_quantiles(lower, upper)
    lo = pl.col(VALUE).quantile(lower).over(TIME)
    hi = pl.col(VALUE).quantile(upper).over(TIME)
    return panel_like(frame, pl.col(VALUE).clip(lo, hi))


@transformer
def min_max_scale(frame: pl.DataFrame) -> pl.DataFrame:
    lo = pl.col(VALUE).min().over(TIME)
    hi = pl.col(VALUE).max().over(TIME)
    return panel_like(frame, (pl.col(VALUE) - lo) / (hi - lo))


@transformer
def normalize(frame: pl.DataFrame) -> pl.DataFrame:
    scaled = min_max_scale.operation(frame)
    return panel_like(scaled, 2.0 * pl.col(VALUE) - 1.0)


@transformer
def net_scale(frame: pl.DataFrame) -> pl.DataFrame:
    value = pl.col(VALUE)
    positive_sum = pl.when(value > 0).then(value).otherwise(0.0).sum().over(TIME)
    negative_sum = (
        pl.when(value < 0).then(value.abs()).otherwise(0.0).sum().over(TIME)
    )
    scaled = (
        pl.when(value > 0)
        .then(value / positive_sum)
        .when(value < 0)
        .then(value / negative_sum)
        .when(value == 0)
        .then(0.0)
        .otherwise(None)
    )
    return panel_like(frame, scaled)


def _validate_quantiles(lower: float, upper: float) -> None:
    if not 0 <= lower <= upper <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower <= upper <= 1")


def _plan_normalization(
    frame: pl.LazyFrame,
    config: dict[str, object],
    operation: str,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    value = pl.col(VALUE)
    if operation == "rank":
        expression = value.rank("average").over(TIME) / value.count().over(TIME)
    elif operation == "zscore":
        expression = (
            value - value.mean().over(TIME)
        ) / value.std(ddof=1).over(TIME)
    elif operation == "winsorize":
        lower = config.get("lower", 0.01)
        upper = config.get("upper", 0.99)
        _validate_quantiles(lower, upper)
        expression = value.clip(
            value.quantile(lower).over(TIME),
            value.quantile(upper).over(TIME),
        )
    elif operation in {"min_max_scale", "normalize"}:
        scaled = (
            value - value.min().over(TIME)
        ) / (value.max().over(TIME) - value.min().over(TIME))
        expression = 2.0 * scaled - 1.0 if operation == "normalize" else scaled
    else:
        positive_sum = (
            pl.when(value > 0).then(value).otherwise(0.0).sum().over(TIME)
        )
        negative_sum = (
            pl.when(value < 0)
            .then(value.abs())
            .otherwise(0.0)
            .sum()
            .over(TIME)
        )
        expression = (
            pl.when(value > 0)
            .then(value / positive_sum)
            .when(value < 0)
            .then(value / negative_sum)
            .when(value == 0)
            .then(0.0)
            .otherwise(None)
        )
    return _expression_plan(
        frame,
        expression,
        order,
        asset_time_ordered,
    )


for _plan_name, _plan_transformer in {
    "rank": rank,
    "zscore": zscore,
    "winsorize": winsorize,
    "min_max_scale": min_max_scale,
    "normalize": normalize,
    "net_scale": net_scale,
}.items():
    _plan_transformer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frame, config, order, asset_time_ordered, name=_plan_name: (
            _plan_normalization(
                frame,
                dict(config),
                name,
                order,
                asset_time_ordered,
            )
        )
    )
