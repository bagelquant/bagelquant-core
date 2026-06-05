"""Multi-input scaling composers."""

from __future__ import annotations

import pandas as pd

from .core import composer


@composer
def vol_scale(frame: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
    """Scale values by volatility."""

    return frame.div(volatility)
