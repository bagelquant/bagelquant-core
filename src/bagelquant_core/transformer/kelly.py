"""Kelly-style helper transforms."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, panel_like
from .core import transformer
from .normalization import rank, zscore
from .rolling import rolling_mean, rolling_var


@transformer
def kelly(frame: pl.DataFrame, *, window: int) -> pl.DataFrame:
    mean = rolling_mean.operation(frame, window=window, min_periods=window).rename(
        {VALUE: "mean"}
    )
    var = rolling_var.operation(frame, window=window, min_periods=window).rename(
        {VALUE: "var"}
    )
    data = mean.join(var, on=["time", "asset_id"], how="inner")
    return panel_like(data, pl.col("mean") / pl.col("var"))


@transformer
def kelly_nonan_standardize(frame: pl.DataFrame, *, window: int) -> pl.DataFrame:
    return zscore.operation(kelly.operation(frame, window=window))


@transformer
def kelly_rank_boxcox(
    frame: pl.DataFrame,
    *,
    window: int,
    lambda_: float = 0,
) -> pl.DataFrame:
    from .boxcox import boxcox

    return boxcox.operation(
        rank.operation(kelly.operation(frame, window=window)),
        lambda_=lambda_,
    )


@transformer
def kelly_rescaling_weight(frame: pl.DataFrame, *, window: int) -> pl.DataFrame:
    scored = kelly.operation(frame, window=window)
    gross = pl.col(VALUE).abs().sum().over("time")
    return panel_like(scored, pl.col(VALUE) / gross)
