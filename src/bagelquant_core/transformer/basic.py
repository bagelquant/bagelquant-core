"""Basic element-wise and time-series transformers."""

from __future__ import annotations

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, panel_like, unary
from .core import transformer


@transformer
def identity(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.clone()


@transformer
def abs_value(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).abs())


@transformer
def negate(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, -pl.col(VALUE))


@transformer
def diff(frame: pl.DataFrame, *, periods: int = 1) -> pl.DataFrame:
    _validate_periods(periods, "diff")
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        (pl.col(VALUE) - pl.col(VALUE).shift(periods).over(ASSET_ID)),
    )


@transformer
def pct_change(frame: pl.DataFrame, *, periods: int = 1) -> pl.DataFrame:
    _validate_periods(periods, "pct_change")
    previous = pl.col(VALUE).shift(periods).over(ASSET_ID)
    return panel_like(frame.sort([ASSET_ID, TIME]), pl.col(VALUE) / previous - 1.0)


def _validate_periods(periods: int, operation: str) -> None:
    if not isinstance(periods, int) or isinstance(periods, bool):
        raise TypeError(f"{operation} periods must be an integer")
    if periods == 0:
        raise ValueError(f"{operation} periods must not be zero")
