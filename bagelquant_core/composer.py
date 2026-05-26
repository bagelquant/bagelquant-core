"""
Composer Module

Key concept in bagelquant-core, see docs/01_concepts/composer.md for more details.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class Composer(ABC):
    """
    Multi-input combinatin  

    N inputs -> 1 output (N >= 2)
    """
    arity: int | None = None

    def __init__(self, *parents) -> None:
        if self.arity is not None and len(parents) != self.arity:
            raise ValueError(f"{self.__class__.__name__} "
                             f"expects {self.arity} parents, "
                             f"got {len(parents)}")

        self._parents = tuple(parents)

    @property
    def parents(self):
        return self._parents

    @abstractmethod
    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        ...


class Add(Composer):

    arity = 2

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        lhs, rhs = inputs
        return lhs + rhs


class Sub(Composer):

    arity = 2

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        lhs, rhs = inputs
        return lhs - rhs


class Mul(Composer):

    arity = 2

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        lhs, rhs = inputs
        return lhs * rhs


class Div(Composer):

    arity = 2

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        lhs, rhs = inputs
        return lhs / rhs


class WeightedSum(Composer):


    def __init__(self, *parents, weights: Sequence[float]) -> None:

        self.weights = weights

        super().__init__(*parents)

        if len(weights) != len(self._parents):
            raise ValueError("weights mismatch")

    arity = None  # dynamic arity

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(0, index=inputs[0].index, columns=inputs[0].columns)
        for input, weight in zip(inputs, self.weights):
            output += input * weight
        return output

