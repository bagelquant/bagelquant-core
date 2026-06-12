"""Box-Cox transform."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, unary
from .core import transformer


@transformer
def boxcox(frame: pl.DataFrame, *, lambda_: float = 0) -> pl.DataFrame:
    value = pl.col(VALUE)
    expr = value.log() if lambda_ == 0 else (value.pow(lambda_) - 1.0) / lambda_
    return unary(frame, expr)
