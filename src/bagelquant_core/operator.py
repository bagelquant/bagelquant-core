"""The single allowlisted operation and operation-node contract."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import update_wrapper
from inspect import Parameter, signature
from itertools import count
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import polars as pl

from ._documentation import ensure_operation_docstring
from .node import Node
from .operation_contract import OperationContract, default_operation_contract
from .registry import Registry

OPERATOR_REGISTRY: Registry["Operator"] = Registry("operator")


class Operator:
    """A numerical operation with explicit inputs, parameters and execution rules.

    ``input_mode`` describes the frame calling convention, rather than a node
    class. All calls create the same ``OperationNode``. The runtime may lower a
    pure operation to an existing sparse Polars plan.
    """

    def __init__(self, operation: Callable[..., pl.DataFrame], *,
                 registry_name: str | None = None,
                 contract: OperationContract | None = None,
                 input_mode: str = "transformer", minimum_inputs: int = 1,
                 maximum_inputs: int | None = 1,
                 output_type: str | None = None, factory: Callable[..., Any] | None = None,
                 version: str = "1") -> None:
        ensure_operation_docstring(operation)
        if not isinstance(version, str) or not version.strip():
            raise ValueError("operator version must be a nonempty string")
        self.version = version
        self.operation = operation
        self.registry_name = registry_name or f"{operation.__module__}.{operation.__qualname__}"
        self.display_name = operation.__name__
        self.input_mode = input_mode
        self.minimum_inputs = minimum_inputs
        self.maximum_inputs = maximum_inputs
        self.output_type = output_type
        self.factory = factory
        self.contract = contract or default_operation_contract(operation, kind=input_mode)
        self.panel_parameter_kinds = panel_parameter_kinds(operation)
        self._plan_operation: Callable[..., Any] | None = None
        self._counter = count(1)
        update_wrapper(self, operation)

    def _set_plan_operation(self, operation: Callable[..., Any]) -> None:
        self._plan_operation = operation

    def __call__(self, *sources: Any, name: str | None = None,
                 metadata: Mapping[str, Any] | None = None, **config: Any):
        from ._operation import as_node
        from .graph import Graph

        self.validate_input_count(len(sources))
        scalar = dict(config)
        auxiliary: dict[str, tuple[Node, ...]] = {}
        for parameter, multiple in self.panel_parameter_kinds.items():
            if parameter in scalar:
                raw = scalar[parameter]
                if raw is None:
                    continue
                scalar.pop(parameter)
                values = tuple(raw) if multiple else (raw,)
                if not values:
                    raise ValueError(f"operator input {parameter!r} must not be empty")
                auxiliary[parameter] = tuple(as_node(value, kind="Operator") for value in values)
        node = OperationNode(
            inputs=tuple(as_node(value, kind="Operator") for value in sources),
            panel_parameters=auxiliary, operation=self, config=scalar,
            name=name or f"{self.display_name}_{next(self._counter)}", metadata=metadata,
        )
        return Graph._from_nodes((node,))

    def validate_input_count(self, count: int) -> None:
        if count < self.minimum_inputs or (self.maximum_inputs is not None and count > self.maximum_inputs):
            raise ValueError(f"{self.display_name} requires {self.minimum_inputs}..{self.maximum_inputs or 'N'} inputs, got {count}")


class OperationNode(Node):
    """One node implementation for unary, multi-input and model operations."""

    node_type = "operator"

    def __init__(self, *, inputs: tuple[Node, ...],
                 panel_parameters: Mapping[str, tuple[Node, ...]], operation: Any,
                 config: Mapping[str, Any], name: str,
                 metadata: Mapping[str, Any] | None = None,
                 input_mode: str | None = None) -> None:
        super().__init__(name=name, metadata=metadata)
        self._inputs = inputs
        self._panel_parameters = dict(panel_parameters)
        self._operation = operation
        self._config = dict(config)
        self.execution_kind = input_mode or operation.input_mode

    @property
    def parents(self) -> tuple[Node, ...]:
        return (*self._inputs, *(node for values in self._panel_parameters.values() for node in values))

    def spec_inputs(self) -> tuple[Node, ...]:
        return self._inputs

    def spec_panel_parameters(self) -> Mapping[str, tuple[Node, ...]]:
        return self._panel_parameters

    @property
    def operation(self):
        return self._operation

    @property
    def contract(self) -> OperationContract:
        return self._operation.contract

    def config(self) -> Mapping[str, Any]:
        if self.execution_kind == "prediction_composer":
            return {"operator": f"prediction:{self._operation.kind}", **self._config}
        return {"operator": self._operation.registry_name, **self._config}

    def compute(self, *frames: pl.DataFrame) -> pl.DataFrame:
        if len(frames) != len(self.parents):
            raise ValueError(f"{self.name} requires {len(self.parents)} dependency frames")
        if self.execution_kind == "prediction_composer":
            return self._operation._compute_frames(*frames, alpha_count=self._config["alpha_count"])
        auxiliary: dict[str, Any] = {}
        offset = len(self._inputs)
        for parameter, nodes in self._panel_parameters.items():
            values = frames[offset:offset + len(nodes)]
            offset += len(nodes)
            auxiliary[parameter] = tuple(values) if self._operation.panel_parameter_kinds[parameter] else values[0]
        from .operator_state import operator_checkpoint_node
        signature = {"operator":self._operation.registry_name,"version":self._operation.version,
            "parameters":{name:value for name,value in self._config.items() if name != "anchor_offset"}}
        with operator_checkpoint_node(self.name, signature):
            return self._operation.operation(*frames[:len(self._inputs)], **auxiliary, **self._config)


class OperationCatalog:
    """A category view of the single registry, without duplicate registration."""

    def __init__(self, input_mode: str):
        self.input_mode = input_mode

    def add(self, name: str, item: Operator) -> None:
        OPERATOR_REGISTRY.add(name, item)

    def get(self, name: str) -> Operator:
        item = OPERATOR_REGISTRY.get(name)
        if item.input_mode != self.input_mode:
            raise KeyError(f"{name} is not in the {self.input_mode} catalog")
        return item

    def names(self) -> tuple[str, ...]:
        return tuple(name for name in OPERATOR_REGISTRY.names() if OPERATOR_REGISTRY.get(name).input_mode == self.input_mode)


def panel_parameter_kinds(operation: Callable[..., Any]) -> dict[str, bool]:
    """Infer explicitly annotated keyword-only panel dependencies."""
    hints = get_type_hints(operation)
    result: dict[str, bool] = {}
    for parameter in signature(operation).parameters.values():
        if parameter.kind != Parameter.KEYWORD_ONLY:
            continue
        annotation = hints.get(parameter.name, parameter.annotation)
        origin, arguments = get_origin(annotation), get_args(annotation)
        if annotation in {pl.DataFrame, pl.LazyFrame}:
            result[parameter.name] = False
        elif origin in {tuple, list} and arguments and arguments[0] in {pl.DataFrame, pl.LazyFrame}:
            result[parameter.name] = True
        elif origin in {UnionType, Union}:
            non_null = tuple(value for value in arguments if value is not type(None))
            if len(non_null) == 1 and non_null[0] in {pl.DataFrame, pl.LazyFrame}:
                result[parameter.name] = False
    return result
