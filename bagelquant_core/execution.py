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
        self.cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}
        self._alignment = alignment

    def run(self, target: Node | Graph) -> pd.DataFrame | Mapping[str, pd.DataFrame]:
        if isinstance(target, Graph):
            return self._run_graph(target)
        return self._run_node(target)

    def _run_graph(self, graph: Graph) -> Mapping[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for node in graph.outputs:
            results[node.name] = self._run_node(node)
        return results

    def _run_node(self, node: Node) -> pd.DataFrame:
        inputs = [self._run_node(parent) for parent in node.parents]
        if len(inputs) > 1:
            inputs = list(Panel.align_frames(*inputs, join=self._alignment))

        input_hashes = tuple(hash_dataframe(frame) for frame in inputs)
        cache_key = (node.signature(), input_hashes)

        if cache_key in self.cache:
            logger.debug("Cache hit: %s", node.name)
            return self.cache[cache_key]

        result = node.compute(*inputs)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Node '{node.name}' returned {type(result)}; expected DataFrame"
            )

        self.cache[cache_key] = result
        return result

    def plan(self, graph: Graph) -> tuple[Node, ...]:
        graph.validate()
        return graph.topological_sort()

    def clear_cache(self) -> None:
        self.cache.clear()
