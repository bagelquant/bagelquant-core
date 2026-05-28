"""
Composer Module

Key concept in bagelquant-core, see docs/01_concepts/composer.md for more details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import pandas as pd

from .node import Node
from .registry import Registry

COMPOSER_REGISTRY: Registry["Composer"] = Registry("composer")


class Composer(Node, ABC):
    """
    Multi-input combination.

    N inputs -> 1 output (N >= 2)
    """

    node_type = "composer"
    arity: int | None = None

    def __init__(
        self,
        *parents: Node,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if len(parents) < 1:
            raise ValueError("Composer requires at least one parent")
        if self.arity is not None and len(parents) != self.arity:
            raise ValueError(
                f"{self.__class__.__name__} expects {self.arity} parents, "
                f"got {len(parents)}"
            )

        super().__init__(name=name, metadata=metadata)
        self._parents = tuple(parents)

    @property
    def parents(self) -> tuple[Node, ...]:
        return self._parents

    @abstractmethod
    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        ...


@COMPOSER_REGISTRY.register("add")
class Add(Composer):
    arity = 2

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        lhs, rhs = inputs
        return lhs + rhs


@COMPOSER_REGISTRY.register("sub")
class Sub(Composer):
    arity = 2

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        lhs, rhs = inputs
        return lhs - rhs


@COMPOSER_REGISTRY.register("mul")
class Mul(Composer):
    arity = 2

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        lhs, rhs = inputs
        return lhs * rhs


@COMPOSER_REGISTRY.register("div")
class Div(Composer):
    arity = 2

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        lhs, rhs = inputs
        return lhs / rhs


@COMPOSER_REGISTRY.register("weighted_sum")
class WeightedSum(Composer):
    arity = None  # dynamic arity

    def __init__(
        self,
        *parents: Node,
        weights: Sequence[float],
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.weights = tuple(weights)
        super().__init__(*parents, name=name, metadata=metadata)

        if len(self.weights) != len(self._parents):
            raise ValueError("weights mismatch")

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(0, index=inputs[0].index, columns=inputs[0].columns)
        for input_frame, weight in zip(inputs, self.weights):
            output += input_frame * weight
        return output

    def config(self) -> Mapping[str, Any]:
        return {"weights": list(self.weights)}
