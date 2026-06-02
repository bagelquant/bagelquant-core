from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Iterable, Mapping, Sequence, TypeVar, cast

from .node import Node, NodeSpec

if TYPE_CHECKING:
    from .panel import Panel


class GraphValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GraphSpec:
    outputs: tuple[str, ...]
    nodes: tuple[NodeSpec, ...]


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

    def compute(self) -> OutputT:
        from .execution import _ExecutionRuntime

        return cast(OutputT, _ExecutionRuntime().run(self))

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

            if node.node_type == "transformer" and len(node.parents) != 1:
                raise GraphValidationError(
                    f"Transformer '{node.name}' must have exactly one parent"
                )

            if node.node_type == "composer" and len(node.parents) < 1:
                raise GraphValidationError(
                    f"Composer '{node.name}' must have at least one parent"
                )

    def topological_sort(self) -> tuple[Node, ...]:
        return self._nodes

    def spec(self) -> GraphSpec:
        return GraphSpec(
            outputs=tuple(node.name for node in self._outputs),
            nodes=tuple(node.spec() for node in self._nodes),
        )
