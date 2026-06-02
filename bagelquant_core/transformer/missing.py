"""Missing-value transformers."""

from __future__ import annotations

import pandas as pd

from .core import transformer


@transformer
def fillna(frame: pd.DataFrame, *, value: float = 0) -> pd.DataFrame:
    """Fill missing values with a numeric scalar."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("fillna value must be a real number")
    return frame.fillna(value)


@transformer
def fillna_zero(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with zero."""

    return frame.fillna(0)


@transformer
def ffill(frame: pd.DataFrame, *, limit: int | None = None) -> pd.DataFrame:
    """Forward-fill missing values over time."""

    return frame.ffill(limit=_validate_limit(limit))


@transformer
def bfill(frame: pd.DataFrame, *, limit: int | None = None) -> pd.DataFrame:
    """Backward-fill missing values over time."""

    return frame.bfill(limit=_validate_limit(limit))


def _validate_limit(limit: int | None) -> int | None:
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise ValueError("fill limit must be a positive integer")
    return limit
