"""Cross-sectional translation transformers."""

from __future__ import annotations

import pandas as pd

from .core import transformer


@transformer
def demean(frame: pd.DataFrame) -> pd.DataFrame:
    """Subtract each row's cross-sectional mean."""

    return frame.sub(frame.mean(axis=1), axis=0)


@transformer
def translate_to_pos(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate each row so its minimum value becomes zero."""

    return frame.sub(frame.min(axis=1), axis=0)
