"""Pairwise arithmetic composers."""

from __future__ import annotations

import pandas as pd

from .core import composer


@composer
def add(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    """Add two frames."""

    return lhs + rhs


@composer
def sub(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    """Subtract the second frame from the first."""

    return lhs - rhs


@composer
def mul(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    """Multiply two frames."""

    return lhs * rhs


@composer
def div(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    """Divide the first frame by the second."""

    return lhs / rhs


subtract = sub
multiply = mul
divide = div
