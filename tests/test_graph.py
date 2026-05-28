import pandas as pd
import pytest

from bagelquant_core.graph import Graph, GraphValidationError
from bagelquant_core.node import Node
from bagelquant_core.panel import Panel
from bagelquant_core.transformer import Rank


class DummyNode(Node):
    node_type = "transformer"

    def __init__(self, parent: Node | None = None, name: str | None = None) -> None:
        super().__init__(name=name)
        self._parent = parent

    @property
    def parents(self) -> tuple[Node, ...]:
        return (self._parent,) if self._parent else ()

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()


def test_graph_topological_sort() -> None:
    data = pd.DataFrame({"a": [1, 2]})
    panel = Panel("price", data)
    node = Rank(panel, name="ranked")
    graph = Graph(outputs=[node])

    ordered = graph.topological_sort()
    assert ordered[0] is panel
    assert ordered[-1] is node


def test_graph_detects_cycle() -> None:
    node_a = DummyNode(name="a")
    node_b = DummyNode(parent=node_a, name="b")
    node_a._parent = node_b

    with pytest.raises(GraphValidationError):
        Graph(outputs=[node_a])
