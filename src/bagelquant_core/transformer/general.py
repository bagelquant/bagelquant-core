"""General-purpose transformers."""

from __future__ import annotations

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, panel_like, unary
from .core import transformer


@transformer
def nonnans(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).is_not_nan().cast(pl.Float64))


@transformer
def notnan(frame: pl.DataFrame) -> pl.DataFrame:
    return nonnans.operation(frame)


@transformer
def denoise(frame: pl.DataFrame, *, threshold: float = 1e-12) -> pl.DataFrame:
    return unary(
        frame,
        pl.when(pl.col(VALUE).abs() < threshold).then(0.0).otherwise(pl.col(VALUE)),
    )


@transformer
def posonly(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.when(pl.col(VALUE) > 0).then(pl.col(VALUE)).otherwise(None))


@transformer
def negonly(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.when(pl.col(VALUE) < 0).then(pl.col(VALUE)).otherwise(None))


@transformer
def lag(frame: pl.DataFrame, *, periods: int = 1) -> pl.DataFrame:
    _validate_periods(periods, operation="lag")
    return panel_like(
        frame.sort([ASSET_ID, TIME]), pl.col(VALUE).shift(periods).over(ASSET_ID)
    )


@transformer
def delta(frame: pl.DataFrame, *, interval: int = 1) -> pl.DataFrame:
    _validate_periods(interval, operation="delta")
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.col(VALUE) - pl.col(VALUE).shift(interval).over(ASSET_ID),
    )


@transformer
def rate_of_change(frame: pl.DataFrame, *, interval: int = 1) -> pl.DataFrame:
    _validate_periods(interval, operation="rate_of_change")
    previous = pl.col(VALUE).shift(interval).over(ASSET_ID)
    return panel_like(
        frame.sort([ASSET_ID, TIME]), (pl.col(VALUE) - previous) / previous
    )


@transformer
def remove_repeated(frame: pl.DataFrame) -> pl.DataFrame:
    previous = pl.col(VALUE).shift(1).over(ASSET_ID)
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.when(pl.col(VALUE) == previous).then(None).otherwise(pl.col(VALUE)),
    )


@transformer
def date_age_constraint(frame: pl.DataFrame, *, max_age: int) -> pl.DataFrame:
    if max_age < 0:
        raise ValueError("max_age must be non-negative")
    valid_age = pl.col(VALUE).is_not_null().cum_sum().over(ASSET_ID)
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.when(valid_age <= max_age).then(pl.col(VALUE)).otherwise(None),
    )


@transformer
def constant(frame: pl.DataFrame, *, value: float = 1) -> pl.DataFrame:
    return unary(frame, pl.lit(float(value)))


@transformer
def replace_inf(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(
        frame, pl.when(pl.col(VALUE).is_infinite()).then(None).otherwise(pl.col(VALUE))
    )


def _validate_periods(periods: int, *, operation: str) -> None:
    if not isinstance(periods, int) or isinstance(periods, bool):
        raise TypeError(f"{operation} periods must be an integer")
    if periods <= 0:
        raise ValueError(f"{operation} periods must be positive")
