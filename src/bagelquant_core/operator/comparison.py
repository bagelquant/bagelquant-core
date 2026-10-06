"""Logical and comparison operators."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import VALUE, binary, unary
from bagelquant_core.operator._definition import operator


@operator
def power_df(frame: pl.DataFrame, power: pl.DataFrame) -> pl.DataFrame:
    return binary(frame, power, lambda left, right: left.pow(right))


@operator
def and_(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(
        lhs,
        rhs,
        lambda left, right: (left.cast(pl.Boolean) & right.cast(pl.Boolean)).cast(
            pl.Float64
        ),
    )


@operator
def or_(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(
        lhs,
        rhs,
        lambda left, right: (left.cast(pl.Boolean) | right.cast(pl.Boolean)).cast(
            pl.Float64
        ),
    )


@operator
def not_(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, (~pl.col(VALUE).cast(pl.Boolean)).cast(pl.Float64))


@operator
def xand(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(
        lhs,
        rhs,
        lambda left, right: (
            left.cast(pl.Boolean) == right.cast(pl.Boolean)
        ).cast(pl.Float64),
    )


@operator
def xor(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(
        lhs,
        rhs,
        lambda left, right: (left.cast(pl.Boolean) ^ right.cast(pl.Boolean)).cast(
            pl.Float64
        ),
    )


@operator
def greater(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left > right).cast(pl.Float64))


@operator
def greater_equal(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left >= right).cast(pl.Float64))


@operator
def less(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left < right).cast(pl.Float64))


@operator
def less_equal(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left <= right).cast(pl.Float64))


@operator
def equal(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left == right).cast(pl.Float64))


power = power_df
