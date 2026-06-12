"""Trigonometric transforms."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, unary
from .core import transformer


@transformer
def sin(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).sin())


@transformer
def cos(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).cos())


@transformer
def arcsin(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).arcsin())


@transformer
def arccos(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).arccos())


@transformer
def arctanh(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, 0.5 * ((1.0 + pl.col(VALUE)) / (1.0 - pl.col(VALUE))).log())


@transformer
def arctan(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).arctan())


@transformer
def trig(frame: pl.DataFrame) -> pl.DataFrame:
    return sin.operation(frame)


@transformer
def fisher(frame: pl.DataFrame) -> pl.DataFrame:
    return arctanh.operation(frame)
