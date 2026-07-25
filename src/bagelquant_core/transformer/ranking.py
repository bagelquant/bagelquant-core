"""Ranking transforms."""

from __future__ import annotations

import polars as pl

from ..frame import TIME, VALUE, cross_section_rank, panel_like, unary
from .core import transformer


@transformer
def rankpct(frame: pl.DataFrame) -> pl.DataFrame:
    dense = pl.col(VALUE).rank("dense").over(TIME)
    distinct = pl.col(VALUE).n_unique().over(TIME)
    return panel_like(frame, dense / distinct)


@transformer
def nrank(frame: pl.DataFrame) -> pl.DataFrame:
    pct = cross_section_rank(frame, pct=True)
    return unary(pct, 2.0 * pl.col(VALUE) - 1.0)


@transformer
def logrank(frame: pl.DataFrame) -> pl.DataFrame:
    ranked = cross_section_rank(frame, pct=True)
    return unary(ranked, pl.col(VALUE).log())
