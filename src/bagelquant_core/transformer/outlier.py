"""Outlier transforms."""

from __future__ import annotations

from numbers import Real

import polars as pl

from ..frame import TIME, VALUE, panel_like
from .core import _expression_plan, transformer


@transformer
def truncate(frame: pl.DataFrame, *, lower: Real, upper: Real) -> pl.DataFrame:
    _validate_bounds(lower, upper)
    return panel_like(frame, pl.col(VALUE).clip(float(lower), float(upper)))


@transformer
def trim(frame: pl.DataFrame, *, lower: Real, upper: Real) -> pl.DataFrame:
    _validate_bounds(lower, upper)
    return panel_like(
        frame,
        pl.when(pl.col(VALUE).is_between(float(lower), float(upper)))
        .then(pl.col(VALUE))
        .otherwise(None),
    )


@transformer
def trim_quantile(
    frame: pl.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    if not 0 <= lower <= upper <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower <= upper <= 1")
    lo = pl.col(VALUE).quantile(lower).over(TIME)
    hi = pl.col(VALUE).quantile(upper).over(TIME)
    return panel_like(
        frame,
        pl.when(pl.col(VALUE).is_between(lo, hi)).then(pl.col(VALUE)).otherwise(None),
    )


def _validate_bounds(lower: Real, upper: Real) -> None:
    if not isinstance(lower, Real) or not isinstance(upper, Real):
        raise TypeError("bounds must be real")
    if lower > upper:
        raise ValueError("lower must not exceed upper")


def _plan_outlier(
    frame: pl.LazyFrame,
    config: dict[str, object],
    operation: str,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    lower = config.get("lower", 0.01)
    upper = config.get("upper", 0.99)
    value = pl.col(VALUE)
    if operation in {"truncate", "trim"}:
        _validate_bounds(lower, upper)
        expression = (
            value.clip(float(lower), float(upper))
            if operation == "truncate"
            else pl.when(value.is_between(float(lower), float(upper)))
            .then(value)
            .otherwise(None)
        )
    else:
        if not 0 <= lower <= upper <= 1:
            raise ValueError("quantiles must satisfy 0 <= lower <= upper <= 1")
        lo = value.quantile(lower).over(TIME)
        hi = value.quantile(upper).over(TIME)
        expression = (
            pl.when(value.is_between(lo, hi)).then(value).otherwise(None)
        )
    return _expression_plan(
        frame,
        expression,
        order,
        asset_time_ordered,
    )


for _plan_name, _plan_transformer in {
    "truncate": truncate,
    "trim": trim,
    "trim_quantile": trim_quantile,
}.items():
    _plan_transformer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frame, config, order, asset_time_ordered, name=_plan_name: (
            _plan_outlier(
                frame,
                dict(config),
                name,
                order,
                asset_time_ordered,
            )
        )
    )
