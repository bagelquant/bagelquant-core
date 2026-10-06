"""Replacement operators."""

from __future__ import annotations

from numbers import Real

import polars as pl

from bagelquant_core.frame import VALUE, unary
from bagelquant_core.operator._definition import operator


@operator
def replace_non_nan(frame: pl.DataFrame, *, value: Real) -> pl.DataFrame:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("value must be real")
    return unary(
        frame,
        pl.when(pl.col(VALUE).is_not_null() & ~pl.col(VALUE).is_nan())
        .then(float(value))
        .otherwise(pl.col(VALUE)),
    )


@operator
def non_nan_to_one(frame: pl.DataFrame) -> pl.DataFrame:
    return replace_non_nan.operation(frame, value=1)


@operator
def non_nan_to_zero(frame: pl.DataFrame) -> pl.DataFrame:
    return replace_non_nan.operation(frame, value=0)
