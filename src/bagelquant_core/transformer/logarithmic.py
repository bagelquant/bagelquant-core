"""Logarithmic transforms."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, cross_section_rank, unary
from .core import transformer


@transformer
def log(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).log())


@transformer
def log1p(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, (1.0 + pl.col(VALUE)).log())


@transformer
def signed_log1p(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).sign() * (1.0 + pl.col(VALUE).abs()).log())


@transformer
def log_rank(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(cross_section_rank(frame), pl.col(VALUE).log())


@transformer
def inv_log_sqrt_rank(frame: pl.DataFrame) -> pl.DataFrame:
    ranked = cross_section_rank(frame)
    return unary(ranked, 1.0 / (1.0 + pl.col(VALUE)).log().sqrt())
