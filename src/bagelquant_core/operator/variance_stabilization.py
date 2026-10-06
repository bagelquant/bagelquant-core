"""Variance-stabilizing transforms."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import TIME, VALUE, panel_like, unary
from bagelquant_core.operator._definition import operator


@operator
def anscombe(frame: pl.DataFrame) -> pl.DataFrame:
    translated = pl.col(VALUE) - pl.col(VALUE).min().over(TIME)
    return panel_like(frame, 2.0 * (translated + 3.0 / 8.0).sqrt())


@operator
def freeman(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).sqrt() + (pl.col(VALUE) + 1.0).sqrt())
