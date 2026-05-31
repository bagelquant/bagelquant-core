"""
Execution engine for the computational graph.
"""

from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd

from .graph import Graph
from .hashing import hash_dataframe
from .node import Node
from .panel import Panel

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Deterministic executor with memoization cache.
    """

    def __init__(self, alignment: str = "inner") -> None:
        self.cache: dict[tuple[str, tuple[str, ...]], Panel] = {}
        self._alignment = alignment

    def run(self, graph: Graph) -> Panel | Mapping[str, Panel]:
        if not isinstance(graph, Graph):
            raise TypeError("ExecutionEngine.run expects a Graph")
        if len(graph.outputs) == 1:
            return self._run_node(graph.outputs[0])
        return self._run_graph(graph)

    def _run_graph(self, graph: Graph) -> Mapping[str, Panel]:
        results: dict[str, Panel] = {}
        for node in graph.outputs:
            results[node.name] = self._run_node(node)
        return results

    def _run_node(self, node: Node) -> Panel:
        if isinstance(node, Panel):
            return node

        inputs = [self._run_node(parent) for parent in node.parents]
        if len(inputs) > 1:
            frames = Panel.align_frames(
                *(panel.data for panel in inputs),
                join=self._alignment,
            )
        else:
            frames = tuple(panel.data for panel in inputs)

        input_hashes = tuple(hash_dataframe(frame) for frame in frames)
        cache_key = (node.signature(), input_hashes)

        if cache_key in self.cache:
            logger.debug("Cache hit: %s", node.name)
            output = self.cache[cache_key]
            node.set_output(output)
            return output

        result = node.compute(*frames)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Node '{node.name}' returned {type(result)}; expected DataFrame"
            )

        output = Panel(result, name=node.name, metadata=node.metadata)
        self.cache[cache_key] = output
        node.set_output(output)
        return output

    def plan(self, graph: Graph) -> tuple[Node, ...]:
        graph.validate()
        return graph.topological_sort()

    def clear_cache(self) -> None:
        self.cache.clear()
