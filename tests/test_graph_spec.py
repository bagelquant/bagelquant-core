from __future__ import annotations

import polars as pl
import pytest

from bagelquant_core import CategoryPanel, Domain, Graph, GraphSpec, GraphValidationError, Panel
from bagelquant_core.composer import div
from bagelquant_core.transformer import group_demean, rank


def _inputs() -> dict[str, Panel]:
    domain = Domain(calendar=["2024-01-02"], universe=["A", "B"])
    return {
        "book": Panel.from_domain(
            pl.DataFrame(
                {
                    "time": ["2024-01-02", "2024-01-02"],
                    "asset_id": ["A", "B"],
                    "value": [10.0, 6.0],
                }
            ),
            domain,
            name="book",
        ),
        "price": Panel.from_domain(
            pl.DataFrame(
                {
                    "time": ["2024-01-02", "2024-01-02"],
                    "asset_id": ["A", "B"],
                    "value": [5.0, 6.0],
                }
            ),
            domain,
            name="price",
        ),
    }


def test_graph_spec_round_trip_and_compilation() -> None:
    inputs = _inputs()
    original = rank(div(inputs["book"], inputs["price"], name="ratio"), name="ranked")
    specification = original.spec()

    restored = Graph.from_spec(
        GraphSpec.from_dict(specification.to_dict()),
        inputs=inputs,
    )

    assert restored.spec().to_dict() == specification.to_dict()
    assert restored.compute().collect(dense=True).equals(original.compute().collect(dense=True))


def test_graph_spec_round_trips_named_panel_parameters() -> None:
    inputs = _inputs()
    groups = CategoryPanel.from_domain(
        pl.DataFrame(
            {
                "time": ["2024-01-02", "2024-01-02"],
                "asset_id": ["A", "B"],
                "value": ["one", "one"],
            }
        ),
        inputs["book"].domain,
        name="industry",
    )
    all_inputs = {**inputs, "industry": groups}
    original = group_demean(
        inputs["book"], group=groups, name="industry_neutral"
    )
    specification = original.spec()
    node = specification.to_dict()["nodes"][-1]

    assert node["inputs"] == ["book"]
    assert node["panel_parameters"] == {"group": ["industry"]}
    restored = Graph.from_spec(specification, inputs=all_inputs)
    assert restored.spec().to_dict() == specification.to_dict()
    assert restored.compute().collect(dense=True).equals(original.compute().collect(dense=True))


def test_graph_spec_rejects_unknown_operations() -> None:
    specification = {
        "outputs": ["bad"],
        "nodes": [
            {"name": "book", "node_type": "panel", "inputs": []},
            {
                "name": "bad",
                "node_type": "transformer",
                "inputs": ["book"],
                "config": {"transformer": "unknown.operator"},
            },
        ],
    }

    with pytest.raises(GraphValidationError, match="Unknown transformer"):
        Graph.from_spec(specification, inputs=_inputs())


def test_graph_spec_rejects_forward_references() -> None:
    specification = {
        "outputs": ["ranked"],
        "nodes": [
            {
                "name": "ranked",
                "node_type": "transformer",
                "inputs": ["book"],
                "config": {
                    "transformer": "bagelquant_core.transformer.ranking.rank"
                },
            },
            {"name": "book", "node_type": "panel", "inputs": []},
        ],
    }

    with pytest.raises(GraphValidationError, match="forward dependencies"):
        Graph.from_spec(specification, inputs=_inputs())


def test_graph_spec_rejects_invalid_operator_parameters() -> None:
    specification = rank(
        _inputs()["book"], name="ranked"
    ).spec().to_dict()
    specification["nodes"][-1]["config"]["unknown_parameter"] = True

    with pytest.raises(GraphValidationError, match="invalid parameters"):
        Graph.validate_spec(specification)
