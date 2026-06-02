"""Cross-sectional and group composers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .core import composer


def _by_group(
    frame: pd.DataFrame,
    group: pd.DataFrame,
    operation: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    output = frame.copy()
    for index in frame.index:
        output.loc[index] = frame.loc[index].groupby(group.loc[index], dropna=True).transform(operation)
    return output


@composer
def orthogonalize(frame: pd.DataFrame, *factors: pd.DataFrame) -> pd.DataFrame:
    """Return row-wise residuals after projecting values onto factors."""

    if not factors:
        raise ValueError("orthogonalize requires at least one factor")
    output = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for row in frame.index:
        target = frame.loc[row].to_numpy(dtype=float)
        features = np.column_stack([factor.loc[row].to_numpy(dtype=float) for factor in factors])
        valid = np.isfinite(target) & np.isfinite(features).all(axis=1)
        if valid.sum() <= features.shape[1]:
            continue
        design = np.column_stack([np.ones(valid.sum()), features[valid]])
        coefficients = np.linalg.lstsq(design, target[valid], rcond=None)[0]
        output.loc[row, valid] = target[valid] - design @ coefficients
    return output


@composer
def group_rank(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values.rank())


@composer
def group_mean(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values * 0 + values.mean())


@composer
def group_max(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values * 0 + values.max())


@composer
def group_min(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values * 0 + values.min())


@composer
def group_median(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values * 0 + values.median())


@composer
def group_std(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values * 0 + values.std())


@composer
def group_demean(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values - values.mean())


@composer
def group_zscore(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: (values - values.mean()) / values.std())


@composer
def group_rankpct(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values.rank(method="dense", pct=True))


@composer
def group_percentile(frame: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _by_group(frame, group, lambda values: values.rank(pct=True))
