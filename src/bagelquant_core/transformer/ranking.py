"""Ranking transforms."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, cross_section_rank, unary
from .core import transformer


@transformer
def rankpct(frame: pl.DataFrame) -> pl.DataFrame:
    return cross_section_rank(frame, pct=True)


@transformer
def nrank(frame: pl.DataFrame) -> pl.DataFrame:
    pct = rankpct.operation(frame)
    return unary(pct, 2.0 * pl.col(VALUE) - 1.0)


@transformer
def logrank(frame: pl.DataFrame) -> pl.DataFrame:
    ranked = cross_section_rank(frame)
    return unary(ranked, pl.col(VALUE).log())
