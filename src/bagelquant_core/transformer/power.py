"""Power transforms."""

from __future__ import annotations

from numbers import Real

import polars as pl

from ..frame import VALUE, unary
from .core import transformer


@transformer
def power(frame: pl.DataFrame, *, exponent: Real) -> pl.DataFrame:
    _validate_exponent(exponent)
    return unary(frame, pl.col(VALUE).pow(float(exponent)))


@transformer
def signed_power(frame: pl.DataFrame, *, exponent: Real) -> pl.DataFrame:
    _validate_exponent(exponent)
    return unary(frame, pl.col(VALUE).sign() * pl.col(VALUE).abs().pow(float(exponent)))


@transformer
def sqrt(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).sqrt())


def _validate_exponent(exponent: Real) -> None:
    if not isinstance(exponent, Real) or isinstance(exponent, bool):
        raise TypeError("exponent must be real")
