"""Logarithmic transforms."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import VALUE, cross_section_rank, unary
from bagelquant_core.operator._definition import operator


@operator
def log(frame: pl.DataFrame) -> pl.DataFrame:
    value = pl.col(VALUE)
    return unary(frame, pl.when(value > 0).then(value.log()).otherwise(None))


@operator
def log1p(frame: pl.DataFrame) -> pl.DataFrame:
    value = pl.col(VALUE)
    return unary(
        frame, pl.when(value > -1).then((1.0 + value).log()).otherwise(None)
    )


@operator
def signed_log1p(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, pl.col(VALUE).sign() * (1.0 + pl.col(VALUE).abs()).log())


@operator
def log_rank(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(cross_section_rank(frame, pct=True), pl.col(VALUE).log())


@operator
def inv_log_sqrt_rank(frame: pl.DataFrame) -> pl.DataFrame:
    ranked = cross_section_rank(frame, pct=True)
    return unary(ranked, -pl.col(VALUE).log() / pl.col(VALUE).sqrt())
