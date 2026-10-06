"""Shared definition model for source and derived Nodes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import polars as pl

from .hashing import hash_mapping

if TYPE_CHECKING:
    from .node import Node

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
    inputs: tuple[str, ...]
    panel_parameters: Mapping[str, tuple[str, ...]]


class NodeDefinition:
    node_type: ClassVar[str] = "node"

    def __init__(
        self,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name or _next_default_name(self.__class__)
        self.metadata = dict(metadata or {})
        self._output: Node | None = None

    @property
    def parents(self) -> tuple["Node", ...]: ...

    def compute(self, *inputs: pl.DataFrame) -> pl.DataFrame: ...

    def config(self) -> Mapping[str, Any]:
        return {}

    @property
    def output(self) -> "Node":
        if self._output is None:
            raise RuntimeError(f"Node '{self.name}' has not been executed")
        return self._output

    def set_output(self, output: "Node") -> None:
        self._output = output

    def signature(self) -> str:
        payload = {
            "node_type": self.node_type,
            "class": self.__class__.__name__,
            "name": self.name,
            "config": self.config(),
        }
        if self.node_type == "operator":
            from .operator import OPERATOR_REGISTRY
            from .operator_state import operator_input_context
            payload["implementation"] = OPERATOR_REGISTRY.get(self.config()["operator"]).version
            payload["context"] = operator_input_context(self.logical_id)
        return hash_mapping(payload)

    @property
    def logical_id(self) -> str:
        """Content address of the calculation, excluding names and data state."""

        from .logical import LogicalNodeSpec, normalized_operator_config

        if hasattr(self, "_logical_id_override"):
            return self._logical_id_override

        if self.node_type == "input":
            return LogicalNodeSpec.create(
                node_type="input", input_key=getattr(self, "source_key", self.name),
                parameters={"value_type": self.config().get("value_type", "numeric")},
            ).node_id
        config = dict(self.config())
        operation = config.pop("operator")
        return LogicalNodeSpec.create(
            node_type="operator", operator=operation,
            parameters=normalized_operator_config(operation, config),
            inputs=tuple(parent.logical_id for parent in self.spec_inputs()),
            panel_parameters={role: tuple(parent.logical_id for parent in parents)
                              for role, parents in self.spec_panel_parameters().items()},
        ).node_id

    def spec_inputs(self) -> tuple["Node", ...]:
        """Return semantic operation inputs for graph serialization."""

        return self.parents

    def spec_panel_parameters(self) -> Mapping[str, tuple["Node", ...]]:
        """Return named auxiliary Node dependencies for serialization."""

        return {}

    def spec(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            node_type=self.node_type,
            config=self.config(),
            metadata=self.metadata,
            inputs=tuple(parent.name for parent in self.spec_inputs()),
            panel_parameters={
                name: tuple(parent.name for parent in parents)
                for name, parents in self.spec_panel_parameters().items()
            },
        )

    def dag(self) -> dict[str, Any]:
        """Export this node's full dependency closure before execution."""
        from .graph import Graph

        return Graph._from_nodes((self,)).spec().to_dict()

    def mermaid(self) -> str:
        """Render all primary and named auxiliary dependency edges."""
        from .graph import Graph

        return Graph._from_nodes((self,)).spec().mermaid()
