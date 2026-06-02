"""Variance-stabilizing square-root transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import transformer


def _translate_to_pos(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.min(axis=1), axis=0)


@transformer
def anscombe(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Anscombe's variance-stabilizing transform after translation."""

    return 2 * np.sqrt(_translate_to_pos(frame) + 3 / 8)


@transformer
def freeman(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the Freeman-Tukey variance-stabilizing transform."""

    positive = frame.where(frame >= 0)
    return np.sqrt(positive) + np.sqrt(positive + 1)
