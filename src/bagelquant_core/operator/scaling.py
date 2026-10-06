"""Scaling operators."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import binary
from bagelquant_core.operator._definition import operator


@operator
def vol_scale(frame: pl.DataFrame, *, volatility: pl.DataFrame) -> pl.DataFrame:
    return binary(frame, volatility, lambda value, vol: value / vol)
