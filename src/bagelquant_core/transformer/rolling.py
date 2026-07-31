"""Rolling time-series transformers."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from ..frame import ASSET_ID, TIME, VALUE, panel_like
from .core import transformer

_MAX_WORKING_BYTES = 64 * 1024 * 1024
_RANK_BYTES_PER_WINDOW_VALUE = 8


def _validate_window(window: int, min_periods: int | None) -> int:
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("rolling window must be a positive integer")
    if min_periods is None:
        return window
    if (
        not isinstance(min_periods, int)
        or isinstance(min_periods, bool)
        or min_periods < 0
    ):
        raise ValueError("rolling min_periods must be a non-negative integer")
    if min_periods > window:
        raise ValueError("rolling min_periods must not exceed window")
    return min_periods


def _rolling_expr(frame: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    return panel_like(frame.sort([ASSET_ID, TIME]), expr.over(ASSET_ID))


@transformer
def rolling_mean(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_mean(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_std(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None, ddof: int = 1
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_std(
            window, min_samples=_validate_window(window, min_periods), ddof=ddof
        ),
    )


@transformer
def rolling_min(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_min(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_max(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_max(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_sum(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_sum(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_var(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None, ddof: int = 1
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_var(
            window, min_samples=_validate_window(window, min_periods), ddof=ddof
        ),
    )


@transformer
def rolling_median(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_median(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_skew(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    return _rolling_expr(
        frame,
        pl.col(VALUE)
        .fill_nan(None)
        .rolling_skew(window, min_samples=minp, bias=False),
    )


@transformer
def rolling_kurt(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    return _rolling_expr(
        frame,
        pl.col(VALUE)
        .fill_nan(None)
        .rolling_kurtosis(window, min_samples=minp, fisher=True, bias=False),
    )


@transformer
def rolling_percentile(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    return _rolling_last_rank_numpy(frame, window=window, min_periods=minp, pct=True)


@transformer
def rolling_rank(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    return _rolling_last_rank_numpy(frame, window=window, min_periods=minp, pct=False)


@transformer
def rolling_zscore(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None, ddof: int = 1
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    values = pl.col(VALUE).fill_nan(None)
    valid_count = (
        values.is_not_null()
        .cast(pl.UInt32)
        .rolling_sum(window, min_samples=0)
    )
    mean = values.rolling_mean(window, min_samples=minp)
    std = values.rolling_std(window, min_samples=minp, ddof=ddof)
    last_valid = values.forward_fill()
    return _rolling_expr(
        frame,
        pl.when((valid_count >= minp) & (valid_count > ddof) & (std != 0))
        .then((last_valid - mean) / std)
        .otherwise(None),
    )


def _alpha(
    *,
    com: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float:
    values = [value is not None for value in (com, span, halflife, alpha)]
    if sum(values) != 1:
        raise ValueError("ewm requires exactly one of com, span, halflife, or alpha")
    if alpha is not None:
        return float(alpha)
    if com is not None:
        return 1.0 / (1.0 + float(com))
    if span is not None:
        return 2.0 / (float(span) + 1.0)
    return 1.0 - float(np.exp(np.log(0.5) / float(halflife)))


@transformer
def ewm_mean(
    frame: pl.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
) -> pl.DataFrame:
    _alpha(com=com, span=span, halflife=halflife, alpha=alpha)
    return _rolling_expr(
        frame,
        pl.col(VALUE)
        .fill_nan(None)
        .ewm_mean(
            com=com,
            span=span,
            half_life=halflife,
            alpha=alpha,
            adjust=adjust,
            min_samples=min_periods,
            ignore_nulls=ignore_na,
        ),
    )


@transformer
def ewm_var(
    frame: pl.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    bias: bool = False,
) -> pl.DataFrame:
    _alpha(com=com, span=span, halflife=halflife, alpha=alpha)
    return _rolling_expr(
        frame,
        pl.col(VALUE)
        .fill_nan(None)
        .ewm_var(
            com=com,
            span=span,
            half_life=halflife,
            alpha=alpha,
            adjust=adjust,
            min_samples=min_periods,
            ignore_nulls=ignore_na,
            bias=bias,
        ),
    )


@transformer
def ewm_std(
    frame: pl.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    bias: bool = False,
) -> pl.DataFrame:
    _alpha(com=com, span=span, halflife=halflife, alpha=alpha)
    return _rolling_expr(
        frame,
        pl.col(VALUE)
        .fill_nan(None)
        .ewm_std(
            com=com,
            span=span,
            half_life=halflife,
            alpha=alpha,
            adjust=adjust,
            min_samples=min_periods,
            ignore_nulls=ignore_na,
            bias=bias,
        ),
    )


rolling_ewm = ewm_mean
rolling_ew_std = ewm_std


@transformer
def rolling_ewm_fw(
    frame: pl.DataFrame, *, halflife: float, min_periods: int = 0
) -> pl.DataFrame:
    if halflife <= 0:
        raise ValueError("rolling_ewm_fw halflife must be positive")
    return ewm_mean.operation(frame, halflife=halflife, min_periods=min_periods)


def _rolling_last_rank_numpy(
    frame: pl.DataFrame,
    *,
    window: int,
    min_periods: int,
    pct: bool,
) -> pl.DataFrame:
    """Rank the last non-null window value in bounded vectorized batches."""

    ordered = frame.sort([ASSET_ID, TIME])
    all_values = ordered.get_column(VALUE).to_numpy().astype(float, copy=False)
    result = np.full(len(all_values), np.nan, dtype=float)
    lengths = (
        ordered.group_by(ASSET_ID, maintain_order=True)
        .len()
        .get_column("len")
        .to_numpy()
    )
    offset = 0
    for length in lengths:
        end_offset = offset + int(length)
        values = all_values[offset:end_offset]
        finite = ~np.isnan(values)
        padded = np.pad(
            values,
            (window - 1, 0),
            mode="constant",
            constant_values=np.nan,
        )
        windows = sliding_window_view(padded, window)
        prefix = np.empty(len(values) + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(finite, out=prefix[1:])
        positions = np.arange(len(values))
        starts = np.maximum(positions + 1 - window, 0)
        counts = prefix[positions + 1] - prefix[starts]
        last_indices = np.where(finite, positions, -1)
        np.maximum.accumulate(last_indices, out=last_indices)
        last_values = np.full(len(values), np.nan, dtype=float)
        has_value = last_indices >= 0
        last_values[has_value] = values[last_indices[has_value]]
        rows_per_batch = max(
            1,
            _MAX_WORKING_BYTES
            // max(window * _RANK_BYTES_PER_WINDOW_VALUE, 1),
        )
        for start in range(0, len(values), rows_per_batch):
            end = min(start + rows_per_batch, len(values))
            batch = windows[start:end]
            batch_last = last_values[start:end]
            batch_counts = counts[start:end]
            less = np.count_nonzero(batch < batch_last[:, None], axis=1)
            equal = np.count_nonzero(batch == batch_last[:, None], axis=1)
            ranks = less + (equal + 1.0) / 2.0
            eligible = (batch_counts >= min_periods) & (batch_counts > 0)
            if pct:
                ranks = np.divide(
                    ranks,
                    batch_counts,
                    out=np.full_like(ranks, np.nan, dtype=float),
                    where=eligible,
                )
            result[offset + start : offset + end] = np.where(
                eligible,
                ranks,
                np.nan,
            )
        offset = end_offset
    if ordered.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64}
        )
    return (
        ordered.select(TIME, ASSET_ID)
        .with_columns(pl.Series(VALUE, result, nan_to_null=True))
        .sort([TIME, ASSET_ID])
    )
