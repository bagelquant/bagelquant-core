"""Variance-stabilizing transforms."""

from __future__ import annotations

import polars as pl

from ..frame import TIME, VALUE, panel_like, unary
from .core import transformer


@transformer
def anscombe(frame: pl.DataFrame) -> pl.DataFrame:
    translated = pl.col(VALUE) - pl.col(VALUE).min().over(TIME)
    return panel_like(frame, 2.0 * (translated + 3.0 / 8.0).sqrt())


@transformer
def freeman(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).sqrt() + (pl.col(VALUE) + 1.0).sqrt())
