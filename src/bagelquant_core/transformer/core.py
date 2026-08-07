"""
Core graph-building machinery for unary transformations.

Use ``@transformer`` to turn a DataFrame function into a public function that
accepts a Panel or Graph and returns a lazy Graph.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from inspect import Parameter, signature
from itertools import count
from types import UnionType
from typing import TYPE_CHECKING, Any, Mapping, Union, get_args, get_origin, get_type_hints, overload

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

PlanOperation = Callable[
    [pl.LazyFrame, Mapping[str, Any], str | None, bool],
    tuple[pl.LazyFrame, str | None, bool],
]


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
        self.panel_parameter_kinds = _panel_parameter_kinds(operation)
        self._plan_operation: PlanOperation | None = None
        self._counter = count(1)
        update_wrapper(self, operation)

    def _set_plan_operation(self, operation: PlanOperation) -> None:
        self._plan_operation = operation

    def __call__(
        self,
        source: "Panel | Graph[Panel]",
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **config: Any,
    ) -> "Graph[Panel]":
        from ..graph import Graph

        scalar_config = dict(config)
        panel_parameters: dict[str, tuple[Node, ...]] = {}
        for parameter, multiple in self.panel_parameter_kinds.items():
            if parameter not in scalar_config:
                continue
            raw = scalar_config.pop(parameter)
            values = tuple(raw) if multiple else (raw,)
            if not values:
                raise ValueError(
                    f"Transformer Panel parameter {parameter!r} requires at least one Panel"
                )
            panel_parameters[parameter] = tuple(
                as_node(value, kind=f"Transformer Panel parameter {parameter!r}")
                for value in values
            )

        return Graph._from_nodes(
            (
                _TransformerNode(
                    parent=as_node(source, kind="Transformer"),
                    panel_parameters=panel_parameters,
                    operation=self,
                    config=scalar_config,
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
        panel_parameters: Mapping[str, tuple[Node, ...]],
        operation: TransformerFunction,
        config: Mapping[str, Any],
        name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, metadata=metadata)
        self._parent = parent
        self._panel_parameters = {
            key: tuple(value) for key, value in panel_parameters.items()
        }
        self._operation = operation
        self._config = dict(config)

    @property
    def parents(self) -> tuple[Node, ...]:
        return (
            self._parent,
            *(
                parent
                for values in self._panel_parameters.values()
                for parent in values
            ),
        )

    def spec_inputs(self) -> tuple[Node, ...]:
        return (self._parent,)

    def spec_panel_parameters(self) -> Mapping[str, tuple[Node, ...]]:
        return self._panel_parameters

    def compute(self, *inputs: pl.DataFrame) -> pl.DataFrame:
        expected = 1 + sum(len(value) for value in self._panel_parameters.values())
        if len(inputs) != expected:
            raise ValueError(
                f"Transformer node requires {expected} dependency frames"
            )
        offset = 1
        panel_config: dict[str, Any] = {}
        for parameter, nodes in self._panel_parameters.items():
            values = inputs[offset : offset + len(nodes)]
            offset += len(nodes)
            panel_config[parameter] = (
                tuple(values)
                if self._operation.panel_parameter_kinds[parameter]
                else values[0]
            )
        return self._operation.operation(
            inputs[0], **panel_config, **self._config
        )

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


def _panel_parameter_kinds(
    operation: Callable[..., pl.DataFrame],
) -> dict[str, bool]:
    """Return keyword-only Panel parameters and whether each is variadic."""

    hints = get_type_hints(operation)
    result: dict[str, bool] = {}
    for parameter in signature(operation).parameters.values():
        if parameter.kind != Parameter.KEYWORD_ONLY:
            continue
        annotation = hints.get(parameter.name, parameter.annotation)
        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if annotation in {pl.DataFrame, pl.LazyFrame}:
            result[parameter.name] = False
        elif origin in {tuple, list} and arguments and arguments[0] in {
            pl.DataFrame,
            pl.LazyFrame,
        }:
            result[parameter.name] = True
        elif origin in {UnionType, Union}:
            non_null = tuple(value for value in arguments if value is not type(None))
            if len(non_null) == 1 and non_null[0] in {pl.DataFrame, pl.LazyFrame}:
                result[parameter.name] = False
    return result


def _ordered_expression_plan(
    frame: pl.LazyFrame,
    expression: pl.Expr,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    """Apply an asset-time expression without redundant physical sorting."""

    source = frame if asset_time_ordered else frame.sort(["asset_id", "time"])
    output_order = order if asset_time_ordered else "asset_time"
    return (
        source.with_columns(expression.alias("value")).select(
            "time",
            "asset_id",
            "value",
        ),
        output_order,
        True,
    )


def _expression_plan(
    frame: pl.LazyFrame,
    expression: pl.Expr,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    """Apply an order-independent expression without sorting keys."""

    return (
        frame.with_columns(expression.alias("value")).select(
            "time",
            "asset_id",
            "value",
        ),
        order,
        asset_time_ordered,
    )
