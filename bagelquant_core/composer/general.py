"""General-purpose composers."""

from __future__ import annotations

import pandas as pd

from .core import composer


@composer
def project(frame: pd.DataFrame, binary: pd.DataFrame) -> pd.DataFrame:
    """Project values onto cells where a binary availability frame equals one."""

    return frame.where(binary.eq(1))


@composer
def mask(
    frame: pd.DataFrame,
    mask_frame: pd.DataFrame,
    *,
    replace_value: float = float("nan"),
) -> pd.DataFrame:
    """Replace values where a mask frame is false."""

    if not isinstance(replace_value, (int, float)) or isinstance(replace_value, bool):
        raise TypeError("mask replace_value must be a real number")
    return frame.where(mask_frame.fillna(0).astype(bool), replace_value)


@composer
def coalesce(*frames: pd.DataFrame) -> pd.DataFrame:
    """Return the first non-missing value from each cell."""

    if not frames:
        raise ValueError("coalesce requires at least one frame")
    output = frames[0].copy()
    for frame in frames[1:]:
        output = output.combine_first(frame)
    return output
