"""The single allowlisted operation and operation-node contract."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import update_wrapper
from inspect import Parameter, signature
from itertools import count
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import polars as pl

from bagelquant_core._documentation import ensure_operation_docstring
from bagelquant_core.node import Node
from bagelquant_core.operation_contract import OperationContract, default_operation_contract
from bagelquant_core.registry import Registry

OPERATOR_REGISTRY: Registry["Operator"] = Registry("operator")


class Operator:
    """A numerical operation with explicit inputs, parameters and execution rules.

    Calls construct typed Nodes. The runtime lowers pure operations to sparse
    Polars plans under the same declared contract.
    """

    def __init__(self, operation: Callable[..., pl.DataFrame], *,
                 registry_name: str | None = None,
                 contract: OperationContract | None = None,
                 minimum_inputs: int = 1,
                 maximum_inputs: int | None = 1,
                 output_type: str | None = None, factory: Callable[..., Any] | None = None,
                 version: str = "1", input_types=None, auxiliary_types=None) -> None:
        ensure_operation_docstring(operation)
        if not isinstance(version, str) or not version.strip():
            raise ValueError("operator version must be a nonempty string")
        self.version = version
        self.operation = operation
        self.registry_name = registry_name or f"{operation.__module__}.{operation.__qualname__}"
        self.display_name = operation.__name__
        self.minimum_inputs = minimum_inputs
        self.maximum_inputs = maximum_inputs
        categorical = operation.__name__ in {"identity", "project_domain", "lag", "ffill", "bfill", "mask"} and operation.__module__.startswith("bagelquant_core.operator.")
        self.input_types = tuple(input_types or (("numeric", "prediction", "weights", "category") if categorical else ("numeric", "prediction", "weights")))
        self.auxiliary_types = dict(auxiliary_types or {})
        self.output_type = output_type
        self.factory = factory
        self.contract = contract or default_operation_contract(operation)
        self.panel_parameter_kinds = ({"targets": False, "availability": False} if factory is not None and factory.supervised else {}) if factory is not None else panel_parameter_kinds(operation)
        self._plan_operation: Callable[..., Any] | None = None
        self._counter = count(1)
        update_wrapper(self, operation)

    def _set_plan_operation(self, operation: Callable[..., Any]) -> None:
        self._plan_operation = operation

    def __call__(self, *sources: Any, name: str | None = None,
                 metadata: Mapping[str, Any] | None = None, **config: Any):
        from bagelquant_core._operation import as_node

        self.validate_input_count(len(sources))
        scalar = dict(config)
        auxiliary: dict[str, tuple[Node, ...]] = {}
        for parameter, multiple in self.panel_parameter_kinds.items():
            if parameter in scalar:
                raw = scalar[parameter]
                if raw is None:
                    scalar.pop(parameter)
                    continue
                scalar.pop(parameter)
                values = tuple(raw) if multiple else (raw,)
                if not values:
                    raise ValueError(f"operator input {parameter!r} must not be empty")
                auxiliary[parameter] = tuple(as_node(value, kind="Operator") for value in values)
        if self.factory is not None:
            model = self.factory(**scalar)
            model._validate_alpha_count(len(sources))
            scalar["alpha_count"] = len(sources)
        else:
            model = self
        node = Node.from_operation(
            inputs=tuple(as_node(value, kind="Operator") for value in sources),
            panel_parameters=auxiliary, operation=model, config=scalar,
            name=name or f"{self.display_name}_{next(self._counter)}", metadata=metadata,
        )
        return node

    def resolve_type(self, inputs, auxiliary):
        return self.infer_type(tuple(node.value_type for node in inputs),
            {role: tuple(node.value_type for node in nodes) for role, nodes in auxiliary.items()})

    def infer_type(self, primary, auxiliary):
        """The single type-flow rule for Python, DSL, saved graphs and caches."""
        from bagelquant_core.node import ValueType
        primary = tuple(ValueType(value) for value in primary)
        self.validate_input_count(len(primary))
        if any(value not in self.input_types for value in primary):
            raise TypeError("operator peer input type is not supported")
        if ValueType.PREDICTION in primary and any(value != ValueType.PREDICTION for value in primary):
            raise TypeError("prediction and numeric peer inputs cannot be mixed")
        for role, values in auxiliary.items():
            if role not in self.panel_parameter_kinds:
                raise TypeError(f"unknown auxiliary input role: {role}")
            expected = ValueType(self.auxiliary_types.get(role, "category" if role == "group" else "weights" if role == "reference" else "numeric"))
            if any(ValueType(value) != expected for value in values):
                raise TypeError(f"auxiliary {role!r} requires {expected.value} Nodes")
        if self.factory is not None:
            if any(value != ValueType.NUMERIC for value in primary):
                raise TypeError("prediction model inputs must be numeric")
            if set(auxiliary) != set(self.panel_parameter_kinds):
                raise TypeError("model auxiliary inputs do not match its declared roles")
            return ValueType.PREDICTION
        if self.output_type is not None:
            return ValueType(self.output_type)
        if len(primary) == 1:
            return primary[0]
        if ValueType.CATEGORY in primary:
            raise TypeError("categorical peer inputs require a categorical operator")
        return primary[0]

    def history(self, parameters):
        """Return the declared preceding-coordinate requirement, or full history."""
        declared = self.contract.history
        if declared is None and self.operation.__module__.startswith("bagelquant_core.operator."):
            from bagelquant_core.operation_contract import _finite_history
            declared = _finite_history(self.operation, parameters)
        value = declared(parameters) if callable(declared) else declared
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError("operator history must be a nonnegative observation count or None")
        return value

    def validate_input_count(self, count: int) -> None:
        if count < self.minimum_inputs or (self.maximum_inputs is not None and count > self.maximum_inputs):
            raise ValueError(f"{self.display_name} requires {self.minimum_inputs}..{self.maximum_inputs or 'N'} inputs, got {count}")


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


def operator(operation=None, *, contract=None, minimum_inputs=None, maximum_inputs=None, output_type=None, version="1", input_types=None, auxiliary_types=None):
    """Register one numerical callable; peer arity follows its signature."""
    def decorate(func):
        parameters = tuple(signature(func).parameters.values())
        peers = [p for p in parameters if p.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}]
        variadic = any(p.kind == Parameter.VAR_POSITIONAL for p in parameters)
        lower = minimum_inputs if minimum_inputs is not None else max((1 if output_type == "prediction" else 2) if variadic else 0, len([p for p in peers if p.default is Parameter.empty]))
        upper = maximum_inputs if maximum_inputs is not None else (None if variadic else len(peers))
        wrapped = Operator(func, contract=contract,
                           minimum_inputs=lower, maximum_inputs=upper, output_type=output_type, version=version, input_types=input_types, auxiliary_types=auxiliary_types)
        OPERATOR_REGISTRY.add(wrapped.registry_name, wrapped)
        return wrapped
    return decorate(operation) if operation is not None else decorate
