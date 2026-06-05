"""Trigonometric transformers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .core import transformer


@transformer
def sin(frame: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, np.sin(frame))


@transformer
def cos(frame: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, np.cos(frame))


@transformer
def arcsin(frame: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, np.arcsin(frame.where(frame.abs() <= 1)))


@transformer
def arccos(frame: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, np.arccos(frame.where(frame.abs() <= 1)))


@transformer
def trig(frame: pd.DataFrame) -> pd.DataFrame:
    """Return arccos(value) * arcsin(value)."""

    valid = frame.where(frame.abs() <= 1)
    return cast(pd.DataFrame, np.arccos(valid) * np.arcsin(valid))


@transformer
def arctanh(frame: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, np.arctanh(frame.where(frame.abs() < 1)))


@transformer
def arctan(frame: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, np.arctan(frame))


@transformer
def fisher(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Fisher's z-transform to values strictly inside (-1, 1)."""

    valid = frame.where(frame.abs() < 1)
    return np.log((1 + valid) / (1 - valid)).div(2)
