"""Kelly-style helper transforms."""

from __future__ import annotations

from numbers import Real

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


def _plan_kelly(
    frame: pl.LazyFrame,
    config: dict[str, object],
    operation: str,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    window = config.get("window")
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be positive")
    minp = _validate_window(window, window)
    source = frame if asset_time_ordered else frame.sort([ASSET_ID, TIME])
    output_order = order if asset_time_ordered else "asset_time"
    value = pl.col(VALUE)
    if operation == "kelly_nonan_standardize":
        filled = value.fill_nan(None).fill_null(0.0)
        source = source.with_columns(filled.alias("__kelly_input"))
        prepared = pl.col("__kelly_input")
        source = source.with_columns(
            (
                (prepared - prepared.mean().over(TIME))
                / prepared.std(ddof=1).over(TIME)
            ).alias("__kelly_value")
        )
        value = pl.col("__kelly_value")
    elif operation == "kelly_rank_boxcox":
        lambda_ = config.get("lambda_", 0)
        if not isinstance(lambda_, Real) or isinstance(lambda_, bool):
            raise TypeError("boxcox lambda_ must be a real number")
        source = source.with_columns(
            (
                value.rank("average").over(TIME)
                / value.count().over(TIME)
            ).alias("__kelly_rank")
        )
        ranked = pl.col("__kelly_rank")
        transformed = (
            ranked.log()
            if lambda_ == 0
            else (ranked.pow(float(lambda_)) - 1.0) / float(lambda_)
        )
        source = source.with_columns(
            pl.when(ranked > 0)
            .then(transformed)
            .otherwise(None)
            .alias("__kelly_value")
        )
        value = pl.col("__kelly_value")
    mean = value.rolling_mean(window, min_samples=minp).over(ASSET_ID)
    variance = value.rolling_var(
        window,
        min_samples=minp,
        ddof=1,
    ).over(ASSET_ID)
    expression = mean / variance
    if operation == "kelly_rescaling_weight":
        expression = expression.clip(0.0, 1.0)
    return (
        source.with_columns(expression.alias(VALUE)).select(
            TIME,
            ASSET_ID,
            VALUE,
        ),
        output_order,
        True,
    )


for _plan_name, _plan_transformer in {
    "kelly": kelly,
    "kelly_nonan_standardize": kelly_nonan_standardize,
    "kelly_rank_boxcox": kelly_rank_boxcox,
    "kelly_rescaling_weight": kelly_rescaling_weight,
}.items():
    _plan_transformer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frame, config, order, asset_time_ordered, name=_plan_name: (
            _plan_kelly(
                frame,
                dict(config),
                name,
                order,
                asset_time_ordered,
            )
        )
    )
