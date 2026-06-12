"""Polars panel frame helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import polars as pl

TIME = "time"
ASSET_ID = "asset_id"
VALUE = "value"
PANEL_KEYS = (TIME, ASSET_ID)


def normalize_time_series(values: Sequence[Any] | pl.Series) -> pl.Series:
    series = values if isinstance(values, pl.Series) else pl.Series(TIME, values)
    normalized = series.cast(pl.Date, strict=False)
    if normalized.is_empty():
        raise ValueError("calendar must contain at least one time")
    if normalized.null_count() > 0:
        raise ValueError("calendar times must be valid")
    if normalized.n_unique() != len(normalized):
        raise ValueError("calendar times must be unique")
    if normalized.sort().to_list() != normalized.to_list():
        raise ValueError("calendar times must be sorted ascending")
    return normalized


def normalize_asset_ids(values: Sequence[Any] | pl.Series) -> pl.Series:
    if isinstance(values, (str, bytes)):
        raise TypeError("universe must be a sequence of asset_id values")
    series = values if isinstance(values, pl.Series) else pl.Series(ASSET_ID, values)
    normalized = series.cast(pl.String)
    if normalized.is_empty():
        raise ValueError("universe must contain at least one asset_id")
    if normalized.null_count() > 0:
        raise ValueError("asset_id values must be valid")
    if normalized.n_unique() != len(normalized):
        raise ValueError("asset_id values must be unique")
    return normalized


def normalize_panel_frame(
    frame: pl.DataFrame,
    *,
    value_column: str = VALUE,
    numeric: bool = True,
) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("panel data must be a polars DataFrame")
    missing = [
        column for column in (*PANEL_KEYS, value_column) if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"panel data is missing required columns: {missing}")
    normalized = frame.select(*PANEL_KEYS, value_column).rename({value_column: VALUE})
    normalized = normalized.with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
    )
    if normalized.select(
        pl.any_horizontal(pl.col(TIME).is_null(), pl.col(ASSET_ID).is_null()).any()
    ).item():
        raise ValueError("panel keys must be valid")
    if normalized.select(pl.struct(PANEL_KEYS).is_duplicated().any()).item():
        raise ValueError("panel data must be unique by (time, asset_id)")
    if numeric and not normalized.schema[VALUE].is_numeric():
        raise TypeError("panel value column must be numeric")
    return normalized.sort(list(PANEL_KEYS))


def sort_panel(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(list(PANEL_KEYS))


def panel_like(frame: pl.DataFrame, value: pl.Expr) -> pl.DataFrame:
    return (
        frame.with_columns(value.alias(VALUE))
        .select(*PANEL_KEYS, VALUE)
        .sort(list(PANEL_KEYS))
    )


def unary(frame: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    return panel_like(frame, expr)


def align_frames(
    *frames: pl.DataFrame, join: str = "inner"
) -> tuple[pl.DataFrame, ...]:
    if join not in {"inner", "outer"}:
        raise ValueError("join must be either 'inner' or 'outer'")
    if len(frames) <= 1:
        return tuple(frames)
    keys = frames[0].select(*PANEL_KEYS)
    for frame in frames[1:]:
        keys = keys.join(frame.select(*PANEL_KEYS), on=list(PANEL_KEYS), how=join)
    keys = keys.unique().sort(list(PANEL_KEYS))
    return tuple(keys.join(frame, on=list(PANEL_KEYS), how="left") for frame in frames)


def binary(
    lhs: pl.DataFrame,
    rhs: pl.DataFrame,
    expr: Callable[[pl.Expr, pl.Expr], pl.Expr],
) -> pl.DataFrame:
    joined = lhs.rename({VALUE: "lhs"}).join(
        rhs.rename({VALUE: "rhs"}),
        on=list(PANEL_KEYS),
        how="inner",
    )
    return panel_like(joined, expr(pl.col("lhs"), pl.col("rhs")))


def nary(
    frames: Iterable[pl.DataFrame], reducer: Callable[[list[pl.Expr]], pl.Expr]
) -> pl.DataFrame:
    items = list(frames)
    if not items:
        raise ValueError("at least one frame is required")
    joined = items[0].rename({VALUE: "v0"})
    columns = [pl.col("v0")]
    for index, frame in enumerate(items[1:], start=1):
        name = f"v{index}"
        joined = joined.join(
            frame.rename({VALUE: name}), on=list(PANEL_KEYS), how="inner"
        )
        columns.append(pl.col(name))
    return panel_like(joined, reducer(columns))


def cross_section_rank(frame: pl.DataFrame, *, pct: bool = False) -> pl.DataFrame:
    rank = pl.col(VALUE).rank("average").over(TIME)
    if pct:
        rank = rank / pl.col(VALUE).count().over(TIME)
    return panel_like(frame, rank)


def cross_section_mean(frame: pl.DataFrame) -> pl.Expr:
    return pl.col(VALUE).mean().over(TIME)


def cross_section_std(frame: pl.DataFrame) -> pl.Expr:
    return pl.col(VALUE).std(ddof=1).over(TIME)


def rolling_expr(
    expr: pl.Expr, *, window: int, min_periods: int | None = None
) -> pl.Expr:
    resolved_min = window if min_periods is None else min_periods
    return expr.over(ASSET_ID).rolling_mean(
        window_size=window, min_samples=resolved_min
    )


def map_groups_numpy(
    frame: pl.DataFrame,
    func: Callable[[np.ndarray], np.ndarray],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for group in frame.sort([ASSET_ID, TIME]).partition_by(ASSET_ID):
        values = np.array(group[VALUE], dtype=float)
        output = func(values)
        for row, value in zip(group.iter_rows(named=True), output, strict=True):
            rows.append({TIME: row[TIME], ASSET_ID: row[ASSET_ID], VALUE: value})
    return pl.DataFrame(
        rows, schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64}
    ).sort(list(PANEL_KEYS))
