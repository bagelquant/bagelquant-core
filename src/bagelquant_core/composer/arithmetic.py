"""Arithmetic composers."""

from __future__ import annotations

import polars as pl

from ..frame import binary
from .core import _horizontal_value_plan, composer


def _safe_divide(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    """Return null where division is undefined because the divisor is zero."""

    return pl.when(right == 0).then(None).otherwise(left / right)


@composer
def add(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: left + right)


@composer
def sub(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: left - right)


@composer
def mul(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: left * right)


@composer
def div(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, _safe_divide)


def _plan_arithmetic(
    frames: tuple[pl.LazyFrame, ...],
    operation: str,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    combined, values = _horizontal_value_plan(frames)
    expressions = {
        "add": values[0] + values[1],
        "sub": values[0] - values[1],
        "mul": values[0] * values[1],
        "div": _safe_divide(values[0], values[1]),
    }
    return (
        combined.select(
            "time",
            "asset_id",
            expressions[operation].alias("value"),
        ),
        order,
        asset_time_ordered,
    )


for _plan_name, _plan_composer in {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
}.items():
    _plan_composer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frames, config, order, asset_time_ordered, name=_plan_name: (
            _plan_arithmetic(
                frames,
                name,
                order,
                asset_time_ordered,
            )
        )
    )
