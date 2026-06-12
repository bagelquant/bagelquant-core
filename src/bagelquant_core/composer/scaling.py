"""Scaling composers."""

from __future__ import annotations

import polars as pl

from ..frame import binary
from .core import composer


@composer
def vol_scale(frame: pl.DataFrame, volatility: pl.DataFrame) -> pl.DataFrame:
    return binary(frame, volatility, lambda value, vol: value / vol)
