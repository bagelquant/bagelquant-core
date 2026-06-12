"""Outlier transforms."""

from __future__ import annotations

from numbers import Real

import polars as pl

from ..frame import TIME, VALUE, panel_like
from .core import transformer


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
