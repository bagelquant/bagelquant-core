"""Sign transformers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .core import transformer


@transformer
def sign(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the element-wise sign."""

    return cast(pd.DataFrame, np.sign(frame))


@transformer
def abs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise absolute values."""

    return frame.abs()
