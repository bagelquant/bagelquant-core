"""Translation transforms."""

from __future__ import annotations

import polars as pl

from ..frame import TIME, VALUE, panel_like
from .core import transformer


@transformer
def demean(frame: pl.DataFrame) -> pl.DataFrame:
    return panel_like(frame, pl.col(VALUE) - pl.col(VALUE).mean().over(TIME))


@transformer
def translate_to_pos(frame: pl.DataFrame) -> pl.DataFrame:
    return panel_like(frame, pl.col(VALUE) - pl.col(VALUE).min().over(TIME))
