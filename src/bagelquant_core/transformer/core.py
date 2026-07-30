"""
Core graph-building machinery for unary transformations.

Use ``@transformer`` to turn a DataFrame function into a public function that
accepts a Panel or Graph and returns a lazy Graph.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from itertools import count
from typing import TYPE_CHECKING, Any, Mapping, overload

import polars as pl

from .._documentation import ensure_operation_docstring
from .._operation import as_node, operation_name
from ..node import Node
from ..operation_contract import OperationContract, default_operation_contract
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
        contract: OperationContract | None = None,
    ) -> None:
        ensure_operation_docstring(operation)
        self.operation = operation
        self.registry_name = registry_name or operation_name(operation)
        self.display_name = operation.__name__
        self.contract = contract or default_operation_contract(
            operation, kind="transformer"
        )
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

    @property
    def operation(self) -> TransformerFunction:
        return self._operation

    @property
    def contract(self) -> OperationContract:
        return self._operation.contract

    def config(self) -> Mapping[str, Any]:
        return {
            "transformer": operation_name(self._operation.operation),
            **self._config,
        }


@overload
def transformer(operation: Callable[..., pl.DataFrame]) -> TransformerFunction: ...


@overload
def transformer(
    operation: None = None,
    *,
    contract: OperationContract,
) -> Callable[[Callable[..., pl.DataFrame]], TransformerFunction]: ...


def transformer(
    operation: Callable[..., pl.DataFrame] | None = None,
    *,
    contract: OperationContract | None = None,
) -> TransformerFunction | Callable[[Callable[..., pl.DataFrame]], TransformerFunction]:
    """Decorate a frame operation and attach its execution contract."""

    def decorate(func: Callable[..., pl.DataFrame]) -> TransformerFunction:
        wrapped = TransformerFunction(func, contract=contract)
        TRANSFORMER_REGISTRY.add(wrapped.registry_name, wrapped)
        return wrapped

    return decorate(operation) if operation is not None else decorate
