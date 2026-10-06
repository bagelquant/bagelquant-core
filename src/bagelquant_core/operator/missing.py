"""Missing-value operators."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import ASSET_ID, TIME, VALUE, panel_like, unary
from bagelquant_core.operator._plans import _ordered_expression_plan
from bagelquant_core.operator._definition import operator


@operator
def fillna(frame: pl.DataFrame, *, value: float = 0) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).fill_null(value).fill_nan(value))


@operator
def fillna_zero(frame: pl.DataFrame) -> pl.DataFrame:
    return fillna.operation(frame, value=0)


@operator
def ffill(frame: pl.DataFrame, *, limit: int | None = None) -> pl.DataFrame:
    _validate_limit(limit)
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.col(VALUE)
        .fill_nan(None)
        .fill_null(strategy="forward", limit=limit)
        .over(ASSET_ID),
    )


@operator
def bfill(frame: pl.DataFrame, *, limit: int | None = None) -> pl.DataFrame:
    _validate_limit(limit)
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.col(VALUE)
        .fill_nan(None)
        .fill_null(strategy="backward", limit=limit)
        .over(ASSET_ID),
    )


def _validate_limit(limit: int | None) -> None:
    if limit is None:
        return
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("fill limit must be a non-negative integer")


def _plan_fill(
    frame: pl.LazyFrame,
    config: dict[str, object],
    order: str | None,
    asset_time_ordered: bool,
    *,
    strategy: str,
) -> tuple[pl.LazyFrame, str | None, bool]:
    raw_limit = config.get("limit")
    _validate_limit(raw_limit)
    limit = raw_limit
    return _ordered_expression_plan(
        frame,
        pl.col(VALUE)
        .fill_nan(None)
        .fill_null(strategy=strategy, limit=limit)
        .over(ASSET_ID),
        order,
        asset_time_ordered,
    )


ffill._set_plan_operation(  # type: ignore[attr-defined]
    lambda frame, config, order, asset_time_ordered: _plan_fill(
        frame,
        dict(config),
        order,
        asset_time_ordered,
        strategy="forward",
    )
)
bfill._set_plan_operation(  # type: ignore[attr-defined]
    lambda frame, config, order, asset_time_ordered: _plan_fill(
        frame,
        dict(config),
        order,
        asset_time_ordered,
        strategy="backward",
    )
)
