"""Arithmetic composers."""

from __future__ import annotations

import polars as pl

from ..frame import binary
from .core import composer


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
    return binary(lhs, rhs, lambda left, right: left / right)


subtract = sub
multiply = mul
divide = div
