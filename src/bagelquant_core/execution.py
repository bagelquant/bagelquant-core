"""
Internal execution runtime for computational graphs.
"""

from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd

from .graph import Graph
from .hashing import hash_dataframe
from .node import Node
from .panel import Domain, Panel

logger = logging.getLogger(__name__)


class _ExecutionRuntime:
    """
    Deterministic executor with memoization cache.
    """

    def __init__(self, alignment: str = "inner") -> None:
        self.cache: dict[tuple[str, tuple[str, ...], str], Panel] = {}
        self._alignment = alignment

    def run(self, graph: Graph) -> Panel | Mapping[str, Panel]:
        if not isinstance(graph, Graph):
            raise TypeError("_ExecutionRuntime.run expects a Graph")
        evaluated: dict[int, Panel] = {}
        if len(graph._outputs) == 1:
            return self._run_node(graph._outputs[0], evaluated)
        return self._run_graph(graph, evaluated)

    def _run_graph(
        self,
        graph: Graph,
        evaluated: dict[int, Panel],
    ) -> Mapping[str, Panel]:
        results: dict[str, Panel] = {}
        for node in graph._outputs:
            results[node.name] = self._run_node(node, evaluated)
        return results

    def _run_node(self, node: Node, evaluated: dict[int, Panel]) -> Panel:
        node_id = id(node)
        if node_id in evaluated:
            return evaluated[node_id]
        if isinstance(node, Panel):
            evaluated[node_id] = node
            return node

        inputs = [self._run_node(parent, evaluated) for parent in node.parents]
        domain = self._resolve_domain(inputs)
        if len(inputs) > 1:
            frames = Panel.align_frames(
                *(panel._data for panel in inputs),
                join=self._alignment,
            )
            input_hashes = tuple(
                panel._data_hash
                if frame is panel._data
                else hash_dataframe(frame)
                for panel, frame in zip(inputs, frames)
            )
        else:
            frames = tuple(panel._data for panel in inputs)
            input_hashes = tuple(panel._data_hash for panel in inputs)
        cache_key = (node.signature(), input_hashes, domain.signature)

        if cache_key in self.cache:
            logger.debug("Cache hit: %s", node.name)
            output = self.cache[cache_key]
            node.set_output(output)
            evaluated[node_id] = output
            return output

        result = node.compute(*frames)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Node '{node.name}' returned {type(result)}; expected DataFrame"
            )

        output = Panel._materialize(
            result,
            domain=domain,
            name=node.name,
            metadata=node.metadata,
        )
        self.cache[cache_key] = output
        node.set_output(output)
        evaluated[node_id] = output
        return output

    def plan(self, graph: Graph) -> tuple[Node, ...]:
        graph.validate()
        return graph.topological_sort()

    def clear_cache(self) -> None:
        self.cache.clear()

    @staticmethod
    def _resolve_domain(inputs: list[Panel]) -> Domain:
        if not inputs:
            raise ValueError("Derived nodes require at least one panel input")
        domain = inputs[0].domain
        if any(not domain.equivalent_to(panel.domain) for panel in inputs[1:]):
            raise ValueError("Composer inputs must use equivalent Domains")
        return domain
