"""Box-Cox transformer."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .core import transformer


@transformer
def boxcox(frame: pd.DataFrame, *, lambda_: float = 0) -> pd.DataFrame:
    """Apply the Box-Cox transform to strictly positive values."""

    if not isinstance(lambda_, (int, float)) or isinstance(lambda_, bool):
        raise TypeError("boxcox lambda_ must be a real number")
    positive = frame.where(frame > 0)
    if lambda_ == 0:
        return cast(pd.DataFrame, np.log(positive))
    return positive.pow(lambda_).sub(1).div(lambda_)
