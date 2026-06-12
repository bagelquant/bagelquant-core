"""Logical and comparison composers."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, binary, unary
from .core import composer


@composer
def power_df(frame: pl.DataFrame, power: pl.DataFrame) -> pl.DataFrame:
    return binary(frame, power, lambda left, right: left.pow(right))


@composer
def and_(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(
        lhs,
        rhs,
        lambda left, right: (left.cast(pl.Boolean) & right.cast(pl.Boolean)).cast(
            pl.Float64
        ),
    )


@composer
def or_(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(
        lhs,
        rhs,
        lambda left, right: (left.cast(pl.Boolean) | right.cast(pl.Boolean)).cast(
            pl.Float64
        ),
    )


@composer
def not_(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, (~pl.col(VALUE).cast(pl.Boolean)).cast(pl.Float64))


@composer
def xand(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(
        lhs,
        rhs,
        lambda left, right: (left.cast(pl.Boolean) & right.cast(pl.Boolean)).cast(
            pl.Float64
        ),
    )


@composer
def xor(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(
        lhs,
        rhs,
        lambda left, right: (left.cast(pl.Boolean) ^ right.cast(pl.Boolean)).cast(
            pl.Float64
        ),
    )


@composer
def greater(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left > right).cast(pl.Float64))


@composer
def greater_equal(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left >= right).cast(pl.Float64))


@composer
def less(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left < right).cast(pl.Float64))


@composer
def less_equal(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left <= right).cast(pl.Float64))


@composer
def equal(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return binary(lhs, rhs, lambda left, right: (left == right).cast(pl.Float64))


power = power_df
