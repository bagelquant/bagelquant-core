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


def _plan_rolling_operation(
    name: str,
    frame: pl.LazyFrame,
    config: dict[str, object],
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    """Build an order-aware lazy rolling plan for the internal executor."""

    source = frame if asset_time_ordered else frame.sort([ASSET_ID, TIME])
    output_order = order if asset_time_ordered else "asset_time"
    window = int(config.get("window", 0))
    min_periods = config.get("min_periods")

    if name.startswith("rolling_") and name != "rolling_ewm_fw":
        minp = _validate_window(
            window,
            None if min_periods is None else int(min_periods),
        )
        values = pl.col(VALUE)
        if name == "rolling_mean":
            expression = values.rolling_mean(window, min_samples=minp)
        elif name == "rolling_std":
            expression = values.rolling_std(
                window,
                min_samples=minp,
                ddof=int(config.get("ddof", 1)),
            )
        elif name == "rolling_min":
            expression = values.rolling_min(window, min_samples=minp)
        elif name == "rolling_max":
            expression = values.rolling_max(window, min_samples=minp)
        elif name == "rolling_sum":
            expression = values.rolling_sum(window, min_samples=minp)
        elif name == "rolling_var":
            expression = values.rolling_var(
                window,
                min_samples=minp,
                ddof=int(config.get("ddof", 1)),
            )
        elif name == "rolling_median":
            expression = values.rolling_median(window, min_samples=minp)
        elif name == "rolling_skew":
            expression = (
                values.fill_nan(None)
                .rolling_skew(window, min_samples=minp, bias=False)
            )
        elif name == "rolling_kurt":
            expression = (
                values.fill_nan(None)
                .rolling_kurtosis(
                    window,
                    min_samples=minp,
                    fisher=True,
                    bias=False,
                )
            )
        elif name == "rolling_zscore":
            clean = values.fill_nan(None)
            valid_count = (
                clean.is_not_null()
                .cast(pl.UInt32)
                .rolling_sum(window, min_samples=0)
            )
            mean = clean.rolling_mean(window, min_samples=minp)
            std = clean.rolling_std(
                window,
                min_samples=minp,
                ddof=int(config.get("ddof", 1)),
            )
            last_valid = clean.forward_fill()
            ddof = int(config.get("ddof", 1))
            expression = (
                pl.when(
                    (valid_count >= minp)
                    & (valid_count > ddof)
                    & (std != 0)
                )
                .then((last_valid - mean) / std)
                .otherwise(None)
            )
        else:
            raise ValueError(f"unsupported rolling plan operation: {name}")
    else:
        if name == "rolling_ewm_fw":
            halflife = float(config["halflife"])
            if halflife <= 0:
                raise ValueError("rolling_ewm_fw halflife must be positive")
            parameters: dict[str, object] = {
                "com": None,
                "span": None,
                "halflife": halflife,
                "alpha": None,
                "min_periods": int(config.get("min_periods", 0)),
                "adjust": True,
                "ignore_na": False,
            }
            name = "ewm_mean"
        else:
            parameters = config
        com = parameters.get("com")
        span = parameters.get("span")
        halflife = parameters.get("halflife")
        alpha = parameters.get("alpha")
        _alpha(
            com=None if com is None else float(com),
            span=None if span is None else float(span),
            halflife=None if halflife is None else float(halflife),
            alpha=None if alpha is None else float(alpha),
        )
        common = {
            "com": com,
            "span": span,
            "half_life": halflife,
            "alpha": alpha,
            "adjust": bool(parameters.get("adjust", True)),
            "min_samples": int(parameters.get("min_periods", 0)),
            "ignore_nulls": bool(parameters.get("ignore_na", False)),
        }
        clean = pl.col(VALUE).fill_nan(None)
        if name == "ewm_mean":
            expression = clean.ewm_mean(**common)
        elif name == "ewm_var":
            expression = clean.ewm_var(
                **common,
                bias=bool(parameters.get("bias", False)),
            )
        elif name == "ewm_std":
            expression = clean.ewm_std(
                **common,
                bias=bool(parameters.get("bias", False)),
            )
        else:
            raise ValueError(f"unsupported EWM plan operation: {name}")

    return (
        source.with_columns(
            expression.over(ASSET_ID).alias(VALUE)
        ).select(TIME, ASSET_ID, VALUE),
        output_order,
        True,
    )


def _register_plan_operation(name: str, operation: object) -> None:
    operation._set_plan_operation(  # type: ignore[attr-defined]
        lambda frame, config, order, asset_time_ordered: _plan_rolling_operation(
            name,
            frame,
            dict(config),
            order,
            asset_time_ordered,
        )
    )


for _plan_name, _plan_transformer in {
    "rolling_mean": rolling_mean,
    "rolling_std": rolling_std,
    "rolling_min": rolling_min,
    "rolling_max": rolling_max,
    "rolling_sum": rolling_sum,
    "rolling_var": rolling_var,
    "rolling_median": rolling_median,
    "rolling_skew": rolling_skew,
    "rolling_kurt": rolling_kurt,
    "rolling_zscore": rolling_zscore,
    "ewm_mean": ewm_mean,
    "ewm_var": ewm_var,
    "ewm_std": ewm_std,
    "rolling_ewm_fw": rolling_ewm_fw,
}.items():
    _register_plan_operation(_plan_name, _plan_transformer)


def _rolling_last_rank_numpy(
    frame: pl.DataFrame,
    *,
    window: int,
    min_periods: int,
    pct: bool,
) -> pl.DataFrame:
    """Rank the last non-null window value in bounded vectorized batches."""

    rank_frame, percentile_frame = _rolling_last_rank_pair(
        frame,
        window=window,
        min_periods=min_periods,
    )
    return percentile_frame if pct else rank_frame


def _rolling_last_rank_pair(
    frame: pl.DataFrame,
    *,
    window: int,
    min_periods: int,
    static_shape: tuple[int, int] | None = None,
    asset_permutation: np.ndarray | None = None,
    group_offsets: np.ndarray | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return exact rank and percentile while sharing window comparisons."""

    if frame.is_empty():
        empty = pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64}
        )
        return empty, empty.clone()
    if static_shape is not None:
        time_count, asset_count = static_shape
        if len(frame) != time_count * asset_count:
            raise ValueError("static eager layout does not match domain size")
        ordered = frame
        values_matrix = (
            frame.get_column(VALUE)
            .to_numpy()
            .astype(float, copy=False)
            .reshape(time_count, asset_count)
        )
        ranks_matrix = np.full(values_matrix.shape, np.nan, dtype=float)
        percentiles_matrix = np.full(
            values_matrix.shape,
            np.nan,
            dtype=float,
        )
        for asset_index in range(asset_count):
            ranks, percentiles = _rolling_rank_series(
                values_matrix[:, asset_index],
                window=window,
                min_periods=min_periods,
            )
            ranks_matrix[:, asset_index] = ranks
            percentiles_matrix[:, asset_index] = percentiles
        rank_values = ranks_matrix.reshape(-1)
        percentile_values = percentiles_matrix.reshape(-1)
    else:
        ordered = (
            frame[asset_permutation]
            if asset_permutation is not None
            else frame.sort([ASSET_ID, TIME])
        )
        all_values = (
            ordered.get_column(VALUE)
            .to_numpy()
            .astype(float, copy=False)
        )
        rank_values = np.full(len(all_values), np.nan, dtype=float)
        percentile_values = np.full(len(all_values), np.nan, dtype=float)
        offsets = group_offsets
        if offsets is None:
            lengths = (
                ordered.group_by(ASSET_ID, maintain_order=True)
                .len()
                .get_column("len")
                .to_numpy()
            )
            offsets = np.empty(len(lengths) + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
        for offset, end_offset in zip(
            offsets[:-1],
            offsets[1:],
            strict=True,
        ):
            ranks, percentiles = _rolling_rank_series(
                all_values[offset:end_offset],
                window=window,
                min_periods=min_periods,
            )
            rank_values[offset:end_offset] = ranks
            percentile_values[offset:end_offset] = percentiles

    keys = ordered.select(TIME, ASSET_ID)
    rank_frame = keys.with_columns(
        pl.Series(VALUE, rank_values, nan_to_null=True)
    )
    percentile_frame = keys.with_columns(
        pl.Series(VALUE, percentile_values, nan_to_null=True)
    )
    if static_shape is None:
        rank_frame = rank_frame.sort([TIME, ASSET_ID])
        percentile_frame = percentile_frame.sort([TIME, ASSET_ID])
    return rank_frame, percentile_frame


def _rolling_rank_series(
    values: np.ndarray,
    *,
    window: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    ranks_output = np.full(len(values), np.nan, dtype=float)
    percentile_output = np.full(len(values), np.nan, dtype=float)
    if not len(values):
        return ranks_output, percentile_output

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
        ranks_output[start:end] = np.where(eligible, ranks, np.nan)
        percentile_output[start:end] = np.divide(
            ranks,
            batch_counts,
            out=np.full_like(ranks, np.nan, dtype=float),
            where=eligible,
        )
    return ranks_output, percentile_output
