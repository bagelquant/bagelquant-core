from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pandas as pd

from .hashing import hash_mapping

if TYPE_CHECKING:
    from .panel import Panel

_NAME_COUNTERS: dict[type, count] = {}


def _next_default_name(cls: type) -> str:
    counter = _NAME_COUNTERS.setdefault(cls, count(1))
    return f"{cls.__name__}_{next(counter)}"


@dataclass(frozen=True, slots=True)
class NodeSpec:
    name: str
    node_type: str
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]
    parents: tuple[str, ...]


class Node(ABC):
    node_type: ClassVar[str] = "node"

    def __init__(
        self,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name or _next_default_name(self.__class__)
        self.metadata = dict(metadata or {})
        self._output: Panel | None = None

    @property
    @abstractmethod
    def parents(self) -> tuple["Node", ...]:
        ...

    @abstractmethod
    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        ...

    def config(self) -> Mapping[str, Any]:
        return {}

    @property
    def output(self) -> "Panel":
        if self._output is None:
            raise RuntimeError(f"Node '{self.name}' has not been executed")
        return self._output

    def set_output(self, output: "Panel") -> None:
        self._output = output

    def signature(self) -> str:
        payload = {
            "node_type": self.node_type,
            "class": self.__class__.__name__,
            "name": self.name,
            "config": self.config(),
        }
        return hash_mapping(payload)

    def spec(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            node_type=self.node_type,
            config=self.config(),
            metadata=self.metadata,
            parents=tuple(parent.name for parent in self.parents),
        )
