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
            if not isinstance(name, str) or not name:
                raise GraphValidationError("graph node name must be a non-empty string")
            if node_type not in {
                "panel",
                "transformer",
                "composer",
                "signal_composer",
            }:
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

        from .composer import COMPOSER_REGISTRY
        from .signal import SIGNAL_COMPOSER_REGISTRY, SignalTrainingContext
        from .transformer import TRANSFORMER_REGISTRY

        by_name: dict[str, Node] = {}

        for node in spec.nodes:
            if node.node_type == "panel":
                try:
                    panel = inputs[node.name]
                except KeyError as error:
                    raise GraphValidationError(
                        f"missing symbolic panel input: {node.name}"
                    ) from error
                by_name[node.name] = panel
                continue

            config = dict(node.config)
            operation_key = node.node_type
            operation_name = cast(str, config.pop(operation_key))
            input_nodes = tuple(by_name[parent] for parent in node.inputs)
            try:
                if node.node_type == "transformer":
                    transformer = TRANSFORMER_REGISTRY.get(operation_name)
                    panel_parameters = {
                        parameter: (
                            tuple(
                                Graph._from_nodes((by_name[value],))
                                for value in values
                            )
                            if transformer.panel_parameter_kinds.get(parameter)
                            else Graph._from_nodes((by_name[values[0]],))
                        )
                        for parameter, values in node.panel_parameters.items()
                    }
                    built = transformer(
                        Graph._from_nodes((input_nodes[0],)),
                        name=node.name,
                        metadata=node.metadata,
                        **panel_parameters,
                        **config,
                    )
                elif node.node_type == "composer":
                    built = COMPOSER_REGISTRY.get(operation_name)(
                        *(Graph._from_nodes((parent,)) for parent in input_nodes),
                        name=node.name,
                        metadata=node.metadata,
                        **config,
                    )
                else:
                    alpha_count = int(config.pop("alpha_count"))
                    standardization = str(config.pop("standardization"))
                    composer_type = SIGNAL_COMPOSER_REGISTRY.get(operation_name)
                    composer = (
                        composer_type(int(config.pop("window")))
                        if "window" in config
                        else composer_type()
                    )
                    if config:
                        raise GraphValidationError(
                            f"unknown signal composer parameters: {sorted(config)}"
                        )
                    training = (
                        SignalTrainingContext(
                            Graph._from_nodes((input_nodes[-2],)),
                            Graph._from_nodes((input_nodes[-1],)),
                        )
                        if composer.supervised
                        else None
                    )
                    built = composer.compose(
                        *(
                            Graph._from_nodes((parent,))
                            for parent in input_nodes[:alpha_count]
                        ),
                        standardization=standardization,
                        training=training,
                        name=node.name,
                        metadata=node.metadata,
                    )
            except KeyError as error:
                raise GraphValidationError(str(error)) from error
            by_name[node.name] = built._single_output()

        return Graph(_nodes=tuple(by_name[name] for name in spec.outputs))

    @classmethod
    def validate_spec(
        cls, specification: GraphSpec | Mapping[str, Any]
    ) -> GraphSpec:
        """Validate topology, registered operators, and operator parameters."""

        from .composer import COMPOSER_REGISTRY
        from .signal import SIGNAL_COMPOSER_REGISTRY
        from .transformer import TRANSFORMER_REGISTRY

        spec = (
            specification
            if isinstance(specification, GraphSpec)
            else GraphSpec.from_dict(specification)
        )
        declared_names = [node.name for node in spec.nodes]
        if len(declared_names) != len(set(declared_names)):
            raise GraphValidationError("graph specification has duplicate node names")
        seen: set[str] = set()
        for node in spec.nodes:
            dependencies = (
                *node.inputs,
                *(
                    value
                    for values in node.panel_parameters.values()
                    for value in values
                ),
            )
            missing = [parent for parent in dependencies if parent not in seen]
            if missing:
                raise GraphValidationError(
                    f"graph node {node.name!r} has unresolved or forward dependencies: "
                    f"{missing}"
                )
            if node.node_type == "panel":
                if node.inputs or node.panel_parameters:
                    raise GraphValidationError(
                        f"panel node {node.name!r} cannot have dependencies"
                    )
                seen.add(node.name)
                continue
            config = dict(node.config)
            operation_name = config.pop(node.node_type, None)
            if not isinstance(operation_name, str) or not operation_name:
                raise GraphValidationError(
                    f"graph node {node.name!r} is missing {node.node_type!r}"
                )
            if node.node_type == "transformer" and len(node.inputs) != 1:
                raise GraphValidationError(
                    f"transformer {node.name!r} must have one input"
                )
            if node.node_type == "composer" and len(node.inputs) < 2:
                raise GraphValidationError(
                    f"composer {node.name!r} requires at least two inputs"
                )
            if node.node_type == "signal_composer" and not node.inputs:
                raise GraphValidationError(
                    f"signal_composer {node.name!r} requires at least one input"
                )
            if node.node_type != "transformer" and node.panel_parameters:
                raise GraphValidationError(
                    f"{node.node_type} {node.name!r} cannot declare panel_parameters"
                )
            try:
                if node.node_type == "transformer":
                    operation = TRANSFORMER_REGISTRY.get(operation_name)
                elif node.node_type == "composer":
                    operation = COMPOSER_REGISTRY.get(operation_name)
                else:
                    composer_type = SIGNAL_COMPOSER_REGISTRY.get(operation_name)
                    window = config.get("window")
                    operation = composer_type(int(window)) if window is not None else composer_type()
            except KeyError as error:
                raise GraphValidationError(str(error)) from error
            try:
                if node.node_type != "signal_composer":
                    if node.node_type == "transformer":
                        panel_arguments = {
                            parameter: (
                                tuple(object() for _ in values)
                                if operation.panel_parameter_kinds.get(parameter)
                                else object()
                            )
                            for parameter, values in node.panel_parameters.items()
                        }
                        signature(operation.operation).bind(
                            object(), **panel_arguments, **config
                        )
                    else:
                        signature(operation.operation).bind(
                            *(object() for _ in node.inputs), **config
                        )
                else:
                    alpha_count = config.get("alpha_count")
                    standardization = config.get("standardization")
                    if (
                        not isinstance(alpha_count, int)
                        or isinstance(alpha_count, bool)
                        or alpha_count <= 0
                    ):
                        raise TypeError("alpha_count must be a positive integer")
                    operation._validate_alpha_count(alpha_count)
                    from .signal import SignalStandardization

                    SignalStandardization(standardization)
                    expected = alpha_count + (2 if operation.supervised else 0)
                    if len(node.inputs) != expected:
                        raise TypeError(
                            f"expected {expected} inputs, got {len(node.inputs)}"
                        )
            except TypeError as error:
                raise GraphValidationError(
                    f"invalid parameters for graph node {node.name!r}: {error}"
                ) from error
            seen.add(node.name)
        missing_outputs = [name for name in spec.outputs if name not in seen]
        if missing_outputs:
            raise GraphValidationError(
                f"graph outputs reference unknown nodes: {missing_outputs}"
            )
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
                    raise GraphValidationError(
                        f"Invalid parent type on {node.name}: {type(parent)}"
                    )

            if node.node_type == "transformer" and len(node.spec_inputs()) != 1:
                raise GraphValidationError(
                    f"Transformer '{node.name}' must have exactly one input"
                )

            if node.node_type == "composer" and len(node.spec_inputs()) < 2:
                raise GraphValidationError(
                    f"Composer '{node.name}' must have at least two inputs"
                )

            if node.node_type == "signal_composer" and len(node.parents) < 1:
                raise GraphValidationError(
                    f"SignalComposer '{node.name}' must have at least one parent"
                )

            if node.node_type != "signal_composer" and any(
                parent.node_type == "signal_composer"
                or parent.__class__.__name__ == "SignalPanel"
                for parent in node.parents
            ):
                raise GraphValidationError(
                    "SignalComposer outputs are terminal and cannot feed other nodes"
                )

        output_ids = {id(node) for node in self._outputs}
        for node in self._nodes:
            if node.node_type == "signal_composer" and id(node) not in output_ids:
                raise GraphValidationError("SignalComposer nodes must be graph outputs")

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
