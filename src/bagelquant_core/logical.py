"""Immutable content-addressed DAG definitions, independent of data state."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from bagelquant_core.hashing import hash_mapping

if TYPE_CHECKING:
    from bagelquant_core.graph import GraphSpec

LOGICAL_GRAPH_SCHEMA = "logical_dag.v1"


def canonical_parameters(value: Any) -> Any:
    """Freeze portable JSON values; never hash process-dependent repr strings."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("logical parameters must contain finite numbers")
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("logical parameter keys must be strings")
        return MappingProxyType({key: canonical_parameters(value[key]) for key in sorted(value)})
    if isinstance(value, (tuple, list)):
        return tuple(canonical_parameters(item) for item in value)
    raise TypeError(f"unsupported logical parameter type: {type(value).__name__}")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def normalized_operator_config(operator_name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Make omitted and explicitly supplied scalar defaults the same definition."""

    from bagelquant_core.operator import OPERATOR_REGISTRY

    operator = OPERATOR_REGISTRY.get(operator_name)
    parameters = dict(config)
    execution_anchor = operator_name.endswith((".rolling_elastic_net_prediction", ".rolling_lightgbm_prediction"))
    if execution_anchor and not isinstance(parameters.get("anchor_offset"), str):
        parameters.pop("anchor_offset", None)
    for name in operator.panel_parameter_kinds:
        if parameters.get(name) is None:
            parameters.pop(name, None)
    callable_ = operator.factory if operator.factory is not None else operator.operation
    for parameter in signature(callable_).parameters.values():
        if execution_anchor and parameter.name == "anchor_offset":
            continue
        if parameter.name in operator.panel_parameter_kinds:
            continue
        if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
            continue
        if operator.factory is None and parameter.kind != Parameter.KEYWORD_ONLY:
            continue
        if parameter.default is not Parameter.empty:
            parameters.setdefault(parameter.name, parameter.default)
    return _json_value(canonical_parameters(parameters))


@dataclass(frozen=True, slots=True)
class LogicalNodeSpec:
    node_id: str
    node_type: str
    input_key: str | None
    operator: str | None
    parameters: Mapping[str, Any]
    inputs: tuple[str, ...]
    panel_parameters: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", canonical_parameters(self.parameters))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "panel_parameters", MappingProxyType(
            {role: tuple(parents) for role, parents in sorted(self.panel_parameters.items())}))

    @classmethod
    def create(cls, *, node_type: str, input_key: str | None = None,
               operator: str | None = None, parameters: Mapping[str, Any] | None = None,
               inputs: Sequence[str] = (), panel_parameters: Mapping[str, Sequence[str]] | None = None) -> LogicalNodeSpec:
        if node_type not in {"input", "operator"}:
            raise ValueError("logical node type must be input or operator")
        if node_type == "input":
            if not isinstance(input_key, str) or not input_key or operator or inputs or panel_parameters:
                raise ValueError("logical input must have a source key and no dependencies")
        elif not isinstance(operator, str) or not operator or input_key is not None:
            raise ValueError("logical operation must have an operator and no source key")
        if any(not isinstance(parent, str) or not parent for parent in inputs):
            raise ValueError("logical inputs must contain nonempty node IDs")
        if any(not isinstance(role, str) or not role or not values or
               any(not isinstance(parent, str) or not parent for parent in values)
               for role, values in (panel_parameters or {}).items()):
            raise ValueError("logical auxiliary inputs require named nonempty node ID lists")
        declared = dict(parameters or {})
        if node_type == "input":
            declared.setdefault("value_type", "numeric")
            if declared["value_type"] not in {"numeric", "category", "prediction", "weights"}:
                raise ValueError("logical input value_type must be numeric, category, prediction or weights")
        if operator in {"bagelquant_core.operator.arithmetic.add", "bagelquant_core.operator.arithmetic.mul"} and len(inputs) == 2:
            inputs = tuple(sorted(inputs))
        frozen_parameters = canonical_parameters(declared)
        auxiliary = MappingProxyType({key: tuple(values) for key, values in sorted((panel_parameters or {}).items())})
        body = {"node_type": node_type, "input_key": input_key, "operator": operator,
                "parameters": _json_value(frozen_parameters), "inputs": list(inputs),
                "panel_parameters": {key: list(values) for key, values in auxiliary.items()}}
        return cls(hash_mapping(body), node_type, input_key, operator,
                   frozen_parameters, tuple(inputs), auxiliary)

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "node_type": self.node_type,
                "input_key": self.input_key, "operator": self.operator,
                "parameters": _json_value(self.parameters), "inputs": list(self.inputs),
                "panel_parameters": {key: list(values) for key, values in self.panel_parameters.items()}}


@dataclass(frozen=True, slots=True)
class LogicalGraphSpec:
    """One reachable logical DAG; union interns equal nodes across branches."""

    outputs: Mapping[str, str]
    nodes: tuple[LogicalNodeSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        seen: set[str] = set()
        for node in self.nodes:
            if node.node_id in seen:
                raise ValueError("logical graph has duplicate node IDs")
            if node.node_type not in {"input", "operator"}:
                raise ValueError("logical node type must be input or operator")
            if node.node_type == "input":
                if not node.input_key or node.operator or node.inputs or node.panel_parameters:
                    raise ValueError("logical input must have a source key and no dependencies")
            elif not node.operator or node.input_key:
                raise ValueError("logical operation must have an operator and no source key")
            dependencies = (*node.inputs, *(parent for values in node.panel_parameters.values() for parent in values))
            if any(parent not in seen for parent in dependencies):
                raise ValueError("logical dependencies must precede their node")
            reproduced = LogicalNodeSpec.create(node_type=node.node_type, input_key=node.input_key,
                operator=node.operator, parameters=node.parameters, inputs=node.inputs,
                panel_parameters=node.panel_parameters)
            if node.node_id != reproduced.node_id:
                raise ValueError("logical node content does not match its ID")
            seen.add(node.node_id)
        if (self.nodes and not self.outputs) or any(not isinstance(alias, str) or not alias or node_id not in seen
                                   for alias, node_id in self.outputs.items()):
            raise ValueError("logical outputs must reference known nodes with nonempty aliases")

    @property
    def identity(self) -> str:
        return hash_mapping({"schema": LOGICAL_GRAPH_SCHEMA, "roots": sorted(set(self.outputs.values()))})

    def to_dict(self) -> dict[str, Any]:
        return {"schema": LOGICAL_GRAPH_SCHEMA, "outputs": dict(self.outputs),
                "nodes": [node.to_dict() for node in self.nodes]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LogicalGraphSpec:
        if value.get("schema") != LOGICAL_GRAPH_SCHEMA:
            raise ValueError(f"logical graph schema must be {LOGICAL_GRAPH_SCHEMA}")
        nodes = []
        for raw in value["nodes"]:
            node = LogicalNodeSpec.create(node_type=raw["node_type"], input_key=raw.get("input_key"),
                operator=raw.get("operator"), parameters=raw.get("parameters", {}),
                inputs=raw.get("inputs", ()), panel_parameters=raw.get("panel_parameters", {}))
            if node.node_id != raw["node_id"]:
                raise ValueError("logical node content does not match its ID")
            nodes.append(node)
        return cls(value["outputs"], tuple(nodes))

    def normalized(self):
        """Apply only proven identity and binary exchange rules to definitions."""
        mapping, nodes = {}, {}
        for node in self.nodes:
            if node.operator == "bagelquant_core.operator.basic.identity" and len(node.inputs) == 1 and not node.parameters and not node.panel_parameters:
                mapping[node.node_id] = mapping[node.inputs[0]]
                continue
            canonical = LogicalNodeSpec.create(node_type=node.node_type, input_key=node.input_key,
                operator=node.operator, parameters=node.parameters,
                inputs=tuple(mapping[parent] for parent in node.inputs),
                panel_parameters={role: tuple(mapping[parent] for parent in values) for role, values in node.panel_parameters.items()})
            mapping[node.node_id] = canonical.node_id
            nodes.setdefault(canonical.node_id, canonical)
        return LogicalGraphSpec({alias: mapping[node_id] for alias, node_id in self.outputs.items()}, tuple(nodes.values())), mapping

    def union(self, *graphs: LogicalGraphSpec) -> LogicalGraphSpec:
        nodes: dict[str, LogicalNodeSpec] = {}
        outputs: dict[str, str] = {}
        for graph in (self, *graphs):
            for node in graph.nodes:
                existing = nodes.setdefault(node.node_id, node)
                if existing.to_dict() != node.to_dict():
                    raise ValueError("logical node hash collision")
            for alias, node_id in graph.outputs.items():
                if alias in outputs and outputs[alias] != node_id:
                    raise ValueError(f"duplicate logical output alias: {alias}")
                outputs[alias] = node_id
        return LogicalGraphSpec(outputs, tuple(nodes.values()))

    def mermaid(self) -> str:
        lines = ["flowchart TD"]
        for node in self.nodes:
            label = node.input_key or node.operator or node.node_id
            lines.append(f"  n{node.node_id}[{json.dumps(label)}]")
            for parent in node.inputs:
                lines.append(f"  n{parent} --> n{node.node_id}")
            for role, parents in node.panel_parameters.items():
                for parent in parents:
                    lines.append(f"  n{parent} -->|{json.dumps(role)}| n{node.node_id}")
        return "\n".join(lines)


def canonicalize_graph(specification: GraphSpec | Mapping[str, Any], *, input_keys: Mapping[str, str] | None = None,
                       output_aliases: Mapping[str, str] | None = None, validate_types: bool = True) -> LogicalGraphSpec:
    """Compile a local authoring template into the sole persistent DAG format.

``input_keys`` maps template input names to permanent semantic source keys.
``output_aliases`` maps requested aliases to template output names. Neither
aliases nor node display metadata participate in node identity.
"""

    from bagelquant_core.graph import Graph

    spec = Graph.validate_spec(specification, validate_types=validate_types)
    by_name: dict[str, LogicalNodeSpec] = {}
    unique: dict[str, LogicalNodeSpec] = {}
    for node in spec.nodes:
        if node.node_type == "input":
            value_type = node.config.get("value_type", node.metadata.get("value_type", "numeric"))
            if value_type not in {"numeric", "category", "prediction", "weights"}:
                raise TypeError(f"unsupported Node value type: {value_type}")
            logical = LogicalNodeSpec.create(node_type="input", input_key=(input_keys or {}).get(node.name, node.name),
                                            parameters={"value_type": value_type})
        else:
            parameters = dict(node.config)
            operator = parameters.pop("operator")
            logical = LogicalNodeSpec.create(node_type="operator", operator=operator,
                parameters=normalized_operator_config(operator, parameters),
                inputs=tuple(by_name[parent].node_id for parent in node.inputs),
                panel_parameters={role: tuple(by_name[parent].node_id for parent in parents)
                                  for role, parents in node.panel_parameters.items()})
        by_name[node.name] = logical
        unique.setdefault(logical.node_id, logical)
    requested = output_aliases or {name: name for name in spec.outputs}
    if any(name not in spec.outputs for name in requested.values()):
        raise ValueError("logical aliases must name declared template outputs")
    outputs = {alias: by_name[name].node_id for alias, name in requested.items()}
    reachable: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in reachable:
            return
        reachable.add(node_id)
        node = unique[node_id]
        for parent in (*node.inputs, *(parent for values in node.panel_parameters.values() for parent in values)):
            visit(parent)

    for node_id in outputs.values():
        visit(node_id)
    return LogicalGraphSpec(outputs, tuple(node for node in unique.values() if node.node_id in reachable)).normalized()[0]


__all__ = ["LogicalNodeSpec", "LogicalGraphSpec", "canonicalize_graph"]
