"""
Core graph-building machinery for unary transformations.

Use ``@transformer`` to turn a DataFrame function into a public function that
accepts a Panel or Graph and returns a lazy Graph.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from itertools import count
from typing import TYPE_CHECKING, Any, Mapping

import polars as pl

from .._operation import as_node, operation_name
from ..node import Node
from ..registry import Registry

if TYPE_CHECKING:
    from ..graph import Graph
    from ..panel import Panel

TRANSFORMER_REGISTRY: Registry["TransformerFunction"] = Registry("transformer")


class TransformerFunction:
    """Callable graph builder created by the ``@transformer`` decorator."""

    def __init__(
        self,
        operation: Callable[..., pl.DataFrame],
        *,
        registry_name: str | None = None,
    ) -> None:
        self.operation = operation
        self.registry_name = registry_name or operation_name(operation)
        self.display_name = operation.__name__
        self._counter = count(1)
        update_wrapper(self, operation)

    def __call__(
        self,
        source: "Panel | Graph[Panel]",
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **config: Any,
    ) -> "Graph[Panel]":
        from ..graph import Graph

        return Graph._from_nodes(
            (
                _TransformerNode(
                    parent=as_node(source, kind="Transformer"),
                    operation=self,
                    config=config,
                    name=name or f"{self.display_name}_{next(self._counter)}",
                    metadata=metadata,
                ),
            )
        )


class _TransformerNode(Node):
    node_type = "transformer"

    def __init__(
        self,
        parent: Node,
        operation: TransformerFunction,
        config: Mapping[str, Any],
        name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, metadata=metadata)
        self._parent = parent
        self._operation = operation
        self._config = dict(config)

    @property
    def parents(self) -> tuple[Node, ...]:
        return (self._parent,)

    def compute(self, *inputs: pl.DataFrame) -> pl.DataFrame:
        if len(inputs) != 1:
            raise ValueError("Transformer node requires exactly one input")
        frame = inputs[0]
        return self._operation.operation(frame, **self._config)

    def config(self) -> Mapping[str, Any]:
        return {
            "transformer": operation_name(self._operation.operation),
            **self._config,
        }


def transformer(operation: Callable[..., pl.DataFrame]) -> TransformerFunction:
    """Decorate a DataFrame function as a graph-building transformer."""

    wrapped = TransformerFunction(operation)
    TRANSFORMER_REGISTRY.add(wrapped.registry_name, wrapped)
    return wrapped
