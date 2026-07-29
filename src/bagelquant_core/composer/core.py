"""
Core graph-building machinery for multi-input compositions.

Use ``@composer`` to turn a DataFrame function into a public function that
accepts Panel or Graph inputs and returns a lazy Graph.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from itertools import count
from typing import TYPE_CHECKING, Any, Mapping, overload

import polars as pl

from .._operation import as_node, operation_name
from ..node import Node
from ..operation_contract import OperationContract, default_operation_contract
from ..registry import Registry

if TYPE_CHECKING:
    from ..graph import Graph
    from ..panel import Panel

COMPOSER_REGISTRY: Registry["ComposerFunction"] = Registry("composer")


class ComposerFunction:
    """Callable graph builder created by the ``@composer`` decorator."""

    def __init__(
        self,
        operation: Callable[..., pl.DataFrame],
        *,
        registry_name: str | None = None,
        contract: OperationContract | None = None,
    ) -> None:
        self.operation = operation
        self.registry_name = registry_name or operation_name(operation)
        self.display_name = operation.__name__
        self.contract = contract or default_operation_contract(
            operation, kind="composer"
        )
        self._counter = count(1)
        update_wrapper(self, operation)

    def __call__(
        self,
        *sources: "Panel | Graph[Panel]",
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **config: Any,
    ) -> "Graph[Panel]":
        from ..graph import Graph

        if not sources:
            raise ValueError("Composer requires at least one Panel or Graph")
        return Graph._from_nodes(
            (
                _ComposerNode(
                    parents=tuple(
                        as_node(source, kind="Composer") for source in sources
                    ),
                    operation=self,
                    config=config,
                    name=name or f"{self.display_name}_{next(self._counter)}",
                    metadata=metadata,
                ),
            )
        )


class _ComposerNode(Node):
    node_type = "composer"

    def __init__(
        self,
        parents: tuple[Node, ...],
        operation: ComposerFunction,
        config: Mapping[str, Any],
        name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, metadata=metadata)
        self._parents = parents
        self._operation = operation
        self._config = dict(config)

    @property
    def parents(self) -> tuple[Node, ...]:
        return self._parents

    def compute(self, *frames: pl.DataFrame) -> pl.DataFrame:
        return self._operation.operation(*frames, **self._config)

    @property
    def operation(self) -> ComposerFunction:
        return self._operation

    @property
    def contract(self) -> OperationContract:
        return self._operation.contract

    def config(self) -> Mapping[str, Any]:
        return {
            "composer": operation_name(self._operation.operation),
            **self._config,
        }


@overload
def composer(operation: Callable[..., pl.DataFrame]) -> ComposerFunction: ...


@overload
def composer(
    operation: None = None,
    *,
    contract: OperationContract,
) -> Callable[[Callable[..., pl.DataFrame]], ComposerFunction]: ...


def composer(
    operation: Callable[..., pl.DataFrame] | None = None,
    *,
    contract: OperationContract | None = None,
) -> ComposerFunction | Callable[[Callable[..., pl.DataFrame]], ComposerFunction]:
    """Decorate a frame operation and attach its execution contract."""

    def decorate(func: Callable[..., pl.DataFrame]) -> ComposerFunction:
        wrapped = ComposerFunction(func, contract=contract)
        COMPOSER_REGISTRY.add(wrapped.registry_name, wrapped)
        return wrapped

    return decorate(operation) if operation is not None else decorate
