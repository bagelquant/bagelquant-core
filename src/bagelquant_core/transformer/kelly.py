"""Kelly-style helper transforms."""

from __future__ import annotations

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, panel_like
from .core import transformer
from .normalization import rank, zscore
from .rolling import _validate_window


@transformer
def kelly(frame: pl.DataFrame, *, window: int) -> pl.DataFrame:
    minp = _validate_window(window, window)
    data = frame.sort([ASSET_ID, TIME])
    mean = pl.col(VALUE).rolling_mean(window, min_samples=minp).over(ASSET_ID)
    var = pl.col(VALUE).rolling_var(window, min_samples=minp, ddof=1).over(ASSET_ID)
    return panel_like(data, mean / var)


@transformer
def kelly_nonan_standardize(frame: pl.DataFrame, *, window: int) -> pl.DataFrame:
    filled = panel_like(frame, pl.col(VALUE).fill_nan(None).fill_null(0.0))
    standardized = zscore.operation(filled)
    return kelly.operation(standardized, window=window)


@transformer
def kelly_rank_boxcox(
    frame: pl.DataFrame,
    *,
    window: int,
    lambda_: float = 0,
) -> pl.DataFrame:
    from .boxcox import boxcox

    transformed = boxcox.operation(rank.operation(frame), lambda_=lambda_)
    return kelly.operation(transformed, window=window)


@transformer
def kelly_rescaling_weight(frame: pl.DataFrame, *, window: int) -> pl.DataFrame:
    scored = kelly.operation(frame, window=window)
    return panel_like(scored, pl.col(VALUE).clip(0.0, 1.0))
