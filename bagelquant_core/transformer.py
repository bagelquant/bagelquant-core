"""
Transformer Module

Key concept in bagelquant-core, see docs/01_concepts/transformer.md for more details.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Transformer(ABC):
    """
    Unary transformations of panels

    1 input -> 1 output
    """
    
    def __init__(self, parents) -> None:
        self._parents = parents

    @property
    def parents(self):
        return (self._parents,)

    @abstractmethod
    def compute(self, x: pd.DataFrame) -> pd.DataFrame:
        ...


class Rank(Transformer):


    def compute(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.rank(axis=1, pct=True)


class ZScore(Transformer):


    def compute(self, x: pd.DataFrame) -> pd.DataFrame:
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        return x.sub(mean, axis=0).div(std, axis=0)


class Winsorize(Transformer):

    def __init__(self, parent, lower=0.01, upper=0.99) -> None:
        super().__init__(parent)

        self.lower = lower
        self.upper = upper

    def compute(self, x: pd.DataFrame) -> pd.DataFrame:
        lower = x.quantile(self.lower, axis=1)
        upper = x.quantile(self.upper, axis=1)
        return x.clip(lower=lower, upper=upper, axis=0)

