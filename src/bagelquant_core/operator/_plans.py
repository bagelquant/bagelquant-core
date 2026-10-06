"""Polars execution plan primitives."""
from __future__ import annotations
import polars as pl

def _ordered_expression_plan(
    frame: pl.LazyFrame,
    expression: pl.Expr,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    """Apply an asset-time expression without redundant physical sorting."""

    source = frame if asset_time_ordered else frame.sort(["asset_id", "time"])
    output_order = order if asset_time_ordered else "asset_time"
    return (
        source.with_columns(expression.alias("value")).select(
            "time",
            "asset_id",
            "value",
        ),
        output_order,
        True,
    )


def _expression_plan(
    frame: pl.LazyFrame,
    expression: pl.Expr,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    """Apply an order-independent expression without sorting keys."""

    return (
        frame.with_columns(expression.alias("value")).select(
            "time",
            "asset_id",
            "value",
        ),
        order,
        asset_time_ordered,
    )
