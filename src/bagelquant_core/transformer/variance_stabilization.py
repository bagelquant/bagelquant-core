"""Variance-stabilizing transforms."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, unary
from .core import transformer


@transformer
def anscombe(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, 2.0 * (pl.col(VALUE) + 3.0 / 8.0).sqrt())


@transformer
def freeman(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).sqrt() + (pl.col(VALUE) + 1.0).sqrt())
