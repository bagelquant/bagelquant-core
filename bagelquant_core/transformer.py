"""
Transformer Module

Key concept in bagelquant-core, see docs/01_concepts/transformer.md for more details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd

from .node import Node
from .registry import Registry

TRANSFORMER_REGISTRY: Registry["Transformer"] = Registry("transformer")


class Transformer(Node, ABC):
    """
    Unary transformations of panels.

    1 input -> 1 output
    """

    node_type = "transformer"

    def __init__(
        self,
        parent: Node,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, metadata=metadata)
        self._parent = parent

    @property
    def parents(self) -> tuple[Node, ...]:
        return (self._parent,)

    @abstractmethod
    def compute(self, x: pd.DataFrame) -> pd.DataFrame:
        ...


@TRANSFORMER_REGISTRY.register("rank")
class Rank(Transformer):
    def compute(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.rank(axis=1, pct=True)


@TRANSFORMER_REGISTRY.register("zscore")
class ZScore(Transformer):
    def compute(self, x: pd.DataFrame) -> pd.DataFrame:
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        return x.sub(mean, axis=0).div(std, axis=0)


@TRANSFORMER_REGISTRY.register("winsorize")
class Winsorize(Transformer):
    def __init__(
        self,
        parent: Node,
        lower: float = 0.01,
        upper: float = 0.99,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, metadata=metadata)
        self.lower = lower
        self.upper = upper

    def compute(self, x: pd.DataFrame) -> pd.DataFrame:
        lower = x.quantile(self.lower, axis=1)
        upper = x.quantile(self.upper, axis=1)
        return x.clip(lower=lower, upper=upper, axis=0)

    def config(self) -> Mapping[str, Any]:
        return {"lower": self.lower, "upper": self.upper}
