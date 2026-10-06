"""Trigonometric transforms."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import VALUE, unary
from bagelquant_core.operator._definition import operator


@operator
def sin(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).sin())


@operator
def cos(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).cos())


@operator
def arcsin(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).arcsin())


@operator
def arccos(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).arccos())


@operator
def arctanh(frame: pl.DataFrame) -> pl.DataFrame:
    value = pl.col(VALUE)
    transformed = 0.5 * ((1.0 + value) / (1.0 - value)).log()
    return unary(frame, pl.when(value.abs() < 1).then(transformed).otherwise(None))


@operator
def arctan(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).arctan())


@operator
def trig(frame: pl.DataFrame) -> pl.DataFrame:
    value = pl.col(VALUE)
    transformed = value.arccos() * value.arcsin()
    return unary(frame, pl.when(value.abs() <= 1).then(transformed).otherwise(None))
