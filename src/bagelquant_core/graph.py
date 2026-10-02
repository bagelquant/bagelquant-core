"""Lazy graph expression objects for BagelQuant Core.

Graphs collect panel inputs, transformer nodes, and composer nodes into a
validated DAG. They can be inspected through ``spec()`` or evaluated by an
``ExecutionRuntime``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Any, Generic, Iterable, Sequence, TypeVar, cast

from .node import Node, NodeSpec

if TYPE_CHECKING:
    from .execution import ExecutionRuntime
    from .panel import Panel


class GraphValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GraphSpec:
    outputs: tuple[str, ...]
    nodes: tuple[NodeSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible graph specification."""

        return {
            "outputs": list(self.outputs),
            "nodes": [
                {
                    "name": node.name,
                    "node_type": node.node_type,
                    "config": dict(node.config),
                    "metadata": dict(node.metadata),
                    "inputs": list(node.inputs),
                    "panel_parameters": {
                        name: list(values)
                        for name, values in node.panel_parameters.items()
                    },
                }
                for node in self.nodes
            ],
        }

    def mermaid(self) -> str:
        """Render the frozen DAG with labelled auxiliary edges."""
        import html
        identifiers = {node.name: f"n{index}" for index, node in enumerate(self.nodes)}
        lines = ["flowchart TD"]
        for node in self.nodes:
            label = html.escape(node.name, quote=True).replace("\n", " ")
            lines.append(f'  {identifiers[node.name]}["{label}"]')
            for parent in node.inputs:
                lines.append(f"  {identifiers[parent]} --> {identifiers[node.name]}")
            for parameter, parents in node.panel_parameters.items():
                for parent in parents:
                    label = html.escape(parameter, quote=True)
                    lines.append(f'  {identifiers[parent]} -->|"{label}"| {identifiers[node.name]}')
            for role, parents in node.metadata.get("context_dependencies", {}).items():
                for parent in parents:
                    label = html.escape(role, quote=True)
                    lines.append(f'  {identifiers[parent]} -.->|"{label}"| {identifiers[node.name]}')
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphSpec":
        """Validate and construct a graph specification from JSON data."""

        outputs = value.get("outputs")
        nodes = value.get("nodes")
        if (
            not isinstance(outputs, list)
            or not outputs
            or not all(isinstance(item, str) and item for item in outputs)
        ):
            raise GraphValidationError(
                "graph outputs must be a non-empty string list"
            )
        if not isinstance(nodes, list) or not nodes:
            raise GraphValidationError("graph nodes must be a non-empty list")
        parsed: list[NodeSpec] = []
        for raw in nodes:
            if not isinstance(raw, Mapping):
                raise GraphValidationError("each graph node must be an object")
            try:
                name = raw["name"]
                node_type = raw["node_type"]
            except KeyError as error:
                raise GraphValidationError(
                    f"graph node is missing {error.args[0]}"
                ) from error
            config = raw.get("config", {})
            metadata = raw.get("metadata", {})
            inputs = raw.get("inputs", [])
            panel_parameters = raw.get("panel_parameters", {})
            context = metadata.get("context_dependencies", {}) if isinstance(metadata, Mapping) else {}
            if not isinstance(context, Mapping) or any(not isinstance(role, str) or not isinstance(parents, list)
                    or not parents or any(not isinstance(parent, str) or not parent for parent in parents)
                    for role, parents in context.items()):
                raise GraphValidationError("context_dependencies must map roles to non-empty node name lists")
            if not isinstance(name, str) or not name:
                raise GraphValidationError("graph node name must be a non-empty string")
            if node_type not in {"panel", "operator"}:
                raise GraphValidationError(
                    f"unsupported graph node type: {node_type!r}"
                )
            if not isinstance(config, Mapping) or not isinstance(metadata, Mapping):
                raise GraphValidationError(
                    f"graph node {name!r} config and metadata must be objects"
                )
            if not isinstance(inputs, list) or not all(
                isinstance(parent, str) and parent for parent in inputs
            ):
                raise GraphValidationError(
                    f"graph node {name!r} inputs must be a string list"
                )
            if not isinstance(panel_parameters, Mapping) or any(
                not isinstance(parameter, str)
                or not parameter
                or not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value for value in values)
                for parameter, values in panel_parameters.items()
            ):
                raise GraphValidationError(
                    f"graph node {name!r} panel_parameters must map names "
                    "to non-empty string lists"
                )
            parsed.append(
                NodeSpec(
                    name=name,
                    node_type=node_type,
                    config=dict(config),
                    metadata=dict(metadata),
                    inputs=tuple(inputs),
                    panel_parameters={
                        str(parameter): tuple(values)
                        for parameter, values in panel_parameters.items()
                    },
                )
            )
        return cls(outputs=tuple(outputs), nodes=tuple(parsed))


OutputT = TypeVar("OutputT", covariant=True)


class Graph(Generic[OutputT]):
    """
    Public graph expression API.

    Graphs represent lazy logic chains. Panels are explicit data inputs and
    execution outputs.
    """

    def __init__(
        self,
        *,
        outputs: Sequence["Graph[Panel]"] | None = None,
        _nodes: Sequence[Node] | None = None,
    ) -> None:
        sources = sum(value is not None for value in (outputs, _nodes))
        if sources != 1:
            raise ValueError("Graph requires exactly one of outputs or _nodes")

        if outputs is not None:
            if not outputs:
                raise ValueError("Graph requires at least one output")
            self._outputs = tuple(graph._single_output() for graph in outputs)
        else:
            assert _nodes is not None
            if not _nodes:
                raise ValueError("Graph requires at least one output")
            self._outputs = tuple(_nodes)

        self._nodes = self._collect_nodes(self._outputs)
        self.validate()

    @classmethod
    def _from_nodes(cls, nodes: Sequence[Node]) -> "Graph[Panel]":
        return Graph(_nodes=nodes)

    @classmethod
    def compile(
        cls, specification: GraphSpec | Mapping[str, Any]
    ) -> "CompiledGraph":
        """Validate and resolve a reusable declarative graph template."""

        return CompiledGraph(cls.validate_spec(specification))

    @classmethod
    def from_spec(
        cls,
        specification: GraphSpec | Mapping[str, Any],
        *,
        inputs: Mapping[str, "Panel"],
    ) -> "Graph[Panel]":
        """Compile a declarative graph using registered safe operations.

        Panel nodes are symbolic references resolved from ``inputs``.
        Transformer and composer names resolve through BagelQuant's registries;
        arbitrary Python callables are never deserialized.
        """

        spec = cls.validate_spec(specification)
        return cls._from_validated_spec(spec, inputs=inputs)

    @classmethod
    def _from_validated_spec(
        cls,
        spec: GraphSpec,
        *,
        inputs: Mapping[str, "Panel"],
    ) -> "Graph[Panel]":
        """Bind inputs without repeating topology and signature validation."""

        from .operator import OPERATOR_REGISTRY, OperationNode

        by_name: dict[str, Node] = {}
        for node in spec.nodes:
            if node.node_type == "panel":
                if node.name not in inputs:
                    raise GraphValidationError(f"missing symbolic panel input: {node.name}")
                by_name[node.name] = inputs[node.name]
                continue
            config = dict(node.config)
            registered = OPERATOR_REGISTRY.get(config.pop("operator"))
            operation = registered
            if registered.factory is not None:
                parameters = {key: value for key, value in config.items() if key != "alpha_count"}
                operation = registered.factory(**parameters)
            by_name[node.name] = OperationNode(
                inputs=tuple(by_name[parent] for parent in node.inputs),
                panel_parameters={key: tuple(by_name[parent] for parent in values)
                                  for key, values in node.panel_parameters.items()},
                operation=operation, config=config, input_mode=registered.input_mode,
                name=node.name, metadata=node.metadata,
            )
        return Graph(_nodes=tuple(by_name[name] for name in spec.outputs))

    @classmethod
    def validate_spec(
        cls, specification: GraphSpec | Mapping[str, Any]
    ) -> GraphSpec:
        """Validate topology, registered operators, and operator parameters."""

        from .operator import OPERATOR_REGISTRY

        spec = specification if isinstance(specification, GraphSpec) else GraphSpec.from_dict(specification)
        declared_names = [node.name for node in spec.nodes]
        if len(declared_names) != len(set(declared_names)):
            raise GraphValidationError("graph specification has duplicate node names")
        seen: set[str] = set()
        for node in spec.nodes:
            if node.node_type not in {"panel", "operator"}:
                raise GraphValidationError(f"unsupported graph node type: {node.node_type!r}")
            dependencies = (*node.inputs, *(value for values in node.panel_parameters.values() for value in values))
            context = tuple(parent for parents in node.metadata.get("context_dependencies", {}).values() for parent in parents)
            missing = [parent for parent in (*dependencies, *context) if parent not in seen]
            if missing:
                raise GraphValidationError(f"graph node {node.name!r} has unresolved or forward dependencies: {missing}")
            if node.node_type == "panel":
                if dependencies:
                    raise GraphValidationError(f"panel node {node.name!r} cannot have dependencies")
                seen.add(node.name)
                continue
            config = dict(node.config)
            operation_name = config.pop("operator", None)
            if not isinstance(operation_name, str) or not operation_name:
                raise GraphValidationError(f"graph node {node.name!r} is missing 'operator'")
            try:
                operation = OPERATOR_REGISTRY.get(operation_name)
                operation.validate_input_count(len(node.inputs))
                if operation.factory is not None:
                    if node.panel_parameters:
                        raise ValueError("model label inputs must be positional dependencies")
                    alpha_count = config.pop("alpha_count", None)
                    if not isinstance(alpha_count, int) or isinstance(alpha_count, bool) or alpha_count <= 0:
                        raise ValueError("alpha_count must be a positive integer")
                    model = operation.factory(**config)
                    model._validate_alpha_count(alpha_count)
                    expected = alpha_count + (2 if model.supervised else 0)
                    if len(node.inputs) != expected:
                        raise ValueError(f"expected {expected} inputs, got {len(node.inputs)}")
                else:
                    unknown = set(node.panel_parameters) - set(operation.panel_parameter_kinds)
                    if unknown:
                        raise ValueError(f"unknown auxiliary inputs: {sorted(unknown)}")
                    for key, values in node.panel_parameters.items():
                        if not operation.panel_parameter_kinds[key] and len(values) != 1:
                            raise ValueError(f"auxiliary input {key!r} requires one panel")
                    panel_arguments = {key: tuple(object() for _ in values)
                        if operation.panel_parameter_kinds[key] else object()
                        for key, values in node.panel_parameters.items()}
                    signature(operation.operation).bind(*(object() for _ in node.inputs), **panel_arguments, **config)
            except (KeyError, TypeError, ValueError) as error:
                raise GraphValidationError(f"invalid operator {node.name!r}: {error}") from error
            seen.add(node.name)
        missing_outputs = [name for name in spec.outputs if name not in seen]
        if missing_outputs:
            raise GraphValidationError(f"graph outputs reference unknown nodes: {missing_outputs}")
        return spec

    @property
    def nodes(self) -> tuple[Node, ...]:
        return self._nodes

    @property
    def name(self) -> str:
        return self._single_output().name

    @property
    def output(self) -> OutputT:
        if len(self._outputs) == 1:
            return cast(OutputT, self._outputs[0].output)
        return cast(OutputT, {node.name: node.output for node in self._outputs})

    def compute(
        self,
        runtime: "ExecutionRuntime | None" = None,
        *,
        dense_output: bool = True,
    ) -> OutputT:
        from .execution import ExecutionRuntime

        executor = runtime or ExecutionRuntime()
        return cast(OutputT, executor.run(self, dense_output=dense_output))

    def _single_output(self) -> Node:
        if len(self._outputs) != 1:
            raise ValueError("Operation requires a Graph with exactly one output")
        return self._outputs[0]

    def _collect_nodes(self, outputs: Iterable[Node]) -> tuple[Node, ...]:
        seen: set[int] = set()
        ordered: list[Node] = []

        def visit(node: Node) -> None:
            node_id = id(node)
            if node_id in seen:
                return
            seen.add(node_id)
            for parent in node.parents:
                if not isinstance(parent, Node):
                    raise GraphValidationError(
                        f"Invalid parent type on {node.name}: {type(parent)}"
                    )
                visit(parent)
            ordered.append(node)

        for output in outputs:
            visit(output)
        return tuple(ordered)

    def validate(self) -> None:
        self._validate_unique_names()
        self._validate_cycles()
        self._validate_parents()

    def _validate_unique_names(self) -> None:
        seen: dict[str, Node] = {}
        for node in self._nodes:
            if node.name in seen and seen[node.name] is not node:
                raise GraphValidationError(
                    f"Duplicate node name: {node.name}. "
                    "Provide unique names for graph nodes."
                )
            seen[node.name] = node

    def _validate_cycles(self) -> None:
        visiting: set[int] = set()
        visited: set[int] = set()

        def dfs(node: Node) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            if node_id in visiting:
                raise GraphValidationError("Cycle detected in graph")
            visiting.add(node_id)
            for parent in node.parents:
                dfs(parent)
            visiting.remove(node_id)
            visited.add(node_id)

        for node in self._outputs:
            dfs(node)

    def _validate_parents(self) -> None:
        for node in self._nodes:
            for parent in node.parents:
                if not isinstance(parent, Node):
                    raise GraphValidationError(f"invalid parent on {node.name}: {type(parent)}")
            if node.node_type == "operator":
                if node.execution_kind == "prediction_composer":
                    alpha_count = node.config()["alpha_count"]
                    node.operation._validate_alpha_count(alpha_count)
                    expected = alpha_count + (2 if node.operation.supervised else 0)
                    if len(node.spec_inputs()) != expected:
                        raise GraphValidationError(f"{node.name} requires {expected} inputs")
                else:
                    node.operation.validate_input_count(len(node.spec_inputs()))

    def topological_sort(self) -> tuple[Node, ...]:
        return self._nodes

    def spec(self) -> GraphSpec:
        return GraphSpec(
            outputs=tuple(node.name for node in self._outputs),
            nodes=tuple(node.spec() for node in self._nodes),
        )


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    """Validated graph topology reusable with different panel inputs."""

    specification: GraphSpec

    def bind(self, inputs: Mapping[str, "Panel"]) -> Graph["Panel"]:
        return Graph._from_validated_spec(self.specification, inputs=inputs)

    def compute(
        self,
        inputs: Mapping[str, "Panel"],
        *,
        runtime: "ExecutionRuntime | None" = None,
        dense_output: bool = True,
    ) -> "Panel | Mapping[str, Panel]":
        from .execution import ExecutionRuntime

        executor = runtime or ExecutionRuntime()
        return executor.run(
            self.bind(inputs),
            dense_output=dense_output,
        )
