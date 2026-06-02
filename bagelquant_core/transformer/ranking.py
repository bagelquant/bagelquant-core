"""Cross-sectional rank transformers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .core import transformer


@transformer
def rankpct(frame: pd.DataFrame) -> pd.DataFrame:
    """Return dense percentile ranks across each row."""

    return frame.rank(axis=1, method="dense", pct=True)


@transformer
def nrank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return percentile ranks normalized to [-1, 1]."""

    return frame.rank(axis=1, pct=True).mul(2).sub(1)


@transformer
def logrank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return natural logarithms of percentile ranks."""

    return cast(pd.DataFrame, np.log(frame.rank(axis=1, pct=True)))
