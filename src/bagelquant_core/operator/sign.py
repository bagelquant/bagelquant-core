"""Sign operators."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import VALUE, unary
from bagelquant_core.operator._definition import operator


@operator
def sign(frame: pl.DataFrame) -> pl.DataFrame:
    value = pl.col(VALUE)
    valid = value.is_not_null() & ~value.is_nan()
    return unary(
        frame,
        pl.when(valid & (value > 0))
        .then(1.0)
        .when(valid & (value < 0))
        .then(-1.0)
        .when(valid & (value == 0))
        .then(0.0)
        .otherwise(None),
    )


@operator
def abs(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).abs())


@operator
def ceil(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).ceil())
