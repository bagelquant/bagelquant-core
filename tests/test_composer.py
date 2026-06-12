from __future__ import annotations

from bagelquant_core.composer import add, group_mean

from helpers import panel, values


def test_add_joins_on_time_asset_id() -> None:
    left = panel([("2024-01-01", "a", 1.0), ("2024-01-01", "b", 2.0)], name="left")
    right = panel([("2024-01-01", "a", 10.0), ("2024-01-01", "b", 20.0)], name="right")

    graph = add(left, right)
    graph.compute()

    assert values(graph.output.data) == {
        ("2024-01-01", "a"): 11.0,
        ("2024-01-01", "b"): 22.0,
    }


def test_group_mean_uses_group_panel() -> None:
    frame = panel([("2024-01-01", "a", 1.0), ("2024-01-01", "b", 3.0)], name="x")
    group = panel([("2024-01-01", "a", 1.0), ("2024-01-01", "b", 1.0)], name="g")

    graph = group_mean(frame, group)
    graph.compute()

    assert values(graph.output.data) == {
        ("2024-01-01", "a"): 2.0,
        ("2024-01-01", "b"): 2.0,
    }
