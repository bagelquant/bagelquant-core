"""Cross-sectional normalization transforms."""

from __future__ import annotations

import polars as pl

from ..frame import TIME, VALUE, cross_section_rank, panel_like
from .core import transformer


@transformer
def rank(frame: pl.DataFrame) -> pl.DataFrame:
    return cross_section_rank(frame)


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
    total = pl.col(VALUE).abs().sum().over(TIME)
    return panel_like(frame, pl.col(VALUE) / total)


@transformer
def net_scale(frame: pl.DataFrame) -> pl.DataFrame:
    gross = pl.col(VALUE).abs().sum().over(TIME)
    net = pl.col(VALUE).sum().over(TIME)
    return panel_like(frame, pl.col(VALUE) / gross - net / gross / pl.len().over(TIME))


def _validate_quantiles(lower: float, upper: float) -> None:
    if not 0 <= lower <= upper <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower <= upper <= 1")
