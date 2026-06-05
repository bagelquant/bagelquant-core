"""Category-level cross-sectional operations."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from ..composer import composer


def _by_category(
    frame: pd.DataFrame,
    categories: pd.DataFrame,
    operation: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    output = frame.copy()
    for index in frame.index:
        values = frame.loc[index]
        labels = categories.loc[index]
        output.loc[index] = values.groupby(labels, dropna=True).transform(operation)
    return output


@composer
def category_demean(
    frame: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    """Subtract each asset's category mean within each row."""

    return _by_category(frame, categories, lambda values: values - values.mean())


@composer
def category_mean(
    frame: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    """Replace each asset value with its category mean within each row."""

    return _by_category(frame, categories, lambda values: values * 0 + values.mean())


@composer
def category_rank(
    frame: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    """Return percentile ranks within each category and row."""

    return _by_category(frame, categories, lambda values: values.rank(pct=True))


@composer
def category_zscore(
    frame: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    """Return z-scores within each category and row."""

    def zscore(values: pd.Series) -> pd.Series:
        std = values.std()
        if std == 0:
            return values * float("nan")
        return (values - values.mean()) / std

    return _by_category(frame, categories, zscore)
