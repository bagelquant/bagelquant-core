import pandas as pd
import pytest

from bagelquant_core import Graph, Panel
from bagelquant_core.composer import composer
from bagelquant_core.graph import GraphValidationError
from bagelquant_core.node import Node
from bagelquant_core.transformer import rank, transformer, zscore
from tests.helpers import make_panel


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


@transformer
def double(frame: pd.DataFrame) -> pd.DataFrame:
    return frame * 2


@transformer
def scale(frame: pd.DataFrame, *, factor: int) -> pd.DataFrame:
    return frame * factor


@composer
def sum_frames(lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame:
    return lhs + rhs


def test_graph_topological_sort() -> None:
    price = make_panel(pd.DataFrame({"a": [1, 2]}), name="price")
    ranked = rank(price, name="ranked")

    ordered = ranked.topological_sort()

    assert ordered[0] is price
    assert ordered[-1].name == "ranked"


def test_graph_detects_cycle() -> None:
    node_a = DummyNode(name="a")
    node_b = DummyNode(parent=node_a, name="b")
    node_a._parent = node_b

    with pytest.raises(GraphValidationError):
        Graph._from_nodes((node_a,))


def test_graph_rejects_invalid_parent_type() -> None:
    node = DummyNode(name="invalid_parent")
    node._parent = "not a node"  # type: ignore[assignment]

    with pytest.raises(GraphValidationError, match="Invalid parent type"):
        Graph._from_nodes((node,))


def test_graph_supports_user_defined_function_operations() -> None:
    left = make_panel(pd.DataFrame({"a": [1, 2]}), name="left")
    right = make_panel(pd.DataFrame({"a": [10, 20]}), name="right")

    doubled = double(left, name="doubled")
    combined = sum_frames(doubled, right, name="combined")

    assert combined.compute().data["a"].tolist() == [12, 24]


def test_user_defined_transformer_config_is_in_graph_spec() -> None:
    price = make_panel(pd.DataFrame({"a": [1, 2]}), name="price")

    scaled = scale(price, factor=3, name="scaled")

    assert scaled.spec().nodes[-1].config["factor"] == 3


def test_zscore_constant_cross_section_does_not_return_infinity() -> None:
    constant = make_panel(pd.DataFrame({"a": [1.0], "b": [1.0]}), name="constant")

    result = zscore(constant).compute()

    assert result.data.isna().all(axis=None)


def test_execution_populates_intermediate_graph_outputs() -> None:
    price = make_panel(pd.DataFrame({"a": [1, 2]}), name="price")
    doubled = double(price, name="doubled")
    final = scale(doubled, factor=3, name="final")

    with pytest.raises(RuntimeError):
        _ = doubled.output

    final.compute()

    assert doubled.output.data["a"].tolist() == [2, 4]
    assert final.output.data["a"].tolist() == [6, 12]
