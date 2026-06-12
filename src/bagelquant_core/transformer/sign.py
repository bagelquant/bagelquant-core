"""Sign transformers."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, unary
from .core import transformer


@transformer
def sign(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(
        frame,
        pl.when(pl.col(VALUE) > 0)
        .then(1.0)
        .when(pl.col(VALUE) < 0)
        .then(-1.0)
        .otherwise(0.0),
    )


@transformer
def abs(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).abs())
