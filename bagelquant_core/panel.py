"""
Panel Module

Key concept in bagelquant-core, see docs/01_concepts/panel.md for more details.
"""

import pandas as pd
from dataclasses import dataclass


@dataclass(slots=True)
class Panel:
    """
    Leaf node in DAG.
    """

    name: str
    data: pd.DataFrame

    @property
    def parents(self) -> tuple:
        return ()

    def compute(self) -> pd.DataFrame:
        return self.data

