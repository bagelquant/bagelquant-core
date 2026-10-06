"""Translation transforms."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import TIME, VALUE, panel_like
from bagelquant_core.operator._definition import operator


@operator
def demean(frame: pl.DataFrame) -> pl.DataFrame:
    return panel_like(frame, pl.col(VALUE) - pl.col(VALUE).mean().over(TIME))


@operator
def translate_to_pos(frame: pl.DataFrame) -> pl.DataFrame:
    return panel_like(frame, pl.col(VALUE) - pl.col(VALUE).min().over(TIME))
