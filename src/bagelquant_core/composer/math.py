"""Element-wise mathematical and logical composers."""

from __future__ import annotations

import pandas as pd

from .core import composer


@composer
def power_df(frame: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    return frame.pow(power)


@composer
def and_(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return (lhs.astype(bool) & rhs.astype(bool)).astype(float)


@composer
def or_(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return (lhs.astype(bool) | rhs.astype(bool)).astype(float)


@composer
def not_(frame: pd.DataFrame) -> pd.DataFrame:
    return (~frame.astype(bool)).astype(float)


@composer
def xand(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    """Return logical equivalence."""

    return lhs.astype(bool).eq(rhs.astype(bool)).astype(float)


@composer
def xor(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return (lhs.astype(bool) ^ rhs.astype(bool)).astype(float)


@composer
def greater(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return lhs.gt(rhs).astype(float)


@composer
def greater_equal(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return lhs.ge(rhs).astype(float)


@composer
def less(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return lhs.lt(rhs).astype(float)


@composer
def less_equal(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return lhs.le(rhs).astype(float)


@composer
def equal(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return lhs.eq(rhs).astype(float)


power = power_df
