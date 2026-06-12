"""Missing-value transformers."""

from __future__ import annotations

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, panel_like, unary
from .core import transformer


@transformer
def fillna(frame: pl.DataFrame, *, value: float = 0) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).fill_null(value).fill_nan(value))


@transformer
def fillna_zero(frame: pl.DataFrame) -> pl.DataFrame:
    return fillna.operation(frame, value=0)


@transformer
def ffill(frame: pl.DataFrame, *, limit: int | None = None) -> pl.DataFrame:
    _validate_limit(limit)
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.col(VALUE).fill_null(strategy="forward", limit=limit).over(ASSET_ID),
    )


@transformer
def bfill(frame: pl.DataFrame, *, limit: int | None = None) -> pl.DataFrame:
    _validate_limit(limit)
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.col(VALUE).fill_null(strategy="backward", limit=limit).over(ASSET_ID),
    )


def _validate_limit(limit: int | None) -> None:
    if limit is None:
        return
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("fill limit must be a non-negative integer")
