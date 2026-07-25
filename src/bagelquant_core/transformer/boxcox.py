"""Box-Cox transform."""

from __future__ import annotations

from numbers import Real

import polars as pl

from ..frame import VALUE, unary
from .core import transformer


@transformer
def boxcox(frame: pl.DataFrame, *, lambda_: float = 0) -> pl.DataFrame:
    if not isinstance(lambda_, Real) or isinstance(lambda_, bool):
        raise TypeError("boxcox lambda_ must be a real number")
    value = pl.col(VALUE)
    transformed = (
        value.log()
        if lambda_ == 0
        else (value.pow(float(lambda_)) - 1.0) / float(lambda_)
    )
    expr = pl.when(value > 0).then(transformed).otherwise(None)
    return unary(frame, expr)
