from __future__ import annotations

from bagelquant_core.transformer import rank, rolling_mean, zscore

from helpers import panel, values


def test_cross_sectional_rank_uses_time_groups() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 2.0),
            ("2024-01-01", "b", 1.0),
        ]
    )

    graph = rank(source)
    graph.compute()

    assert values(graph.output.data) == {
        ("2024-01-01", "a"): 2.0,
        ("2024-01-01", "b"): 1.0,
    }


def test_rolling_mean_uses_asset_id_groups() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 3.0),
            ("2024-01-01", "b", 10.0),
            ("2024-01-02", "b", 20.0),
        ]
    )

    graph = rolling_mean(source, window=2, min_periods=1)
    graph.compute()

    assert values(graph.output.data)[("2024-01-02", "a")] == 2.0
    assert values(graph.output.data)[("2024-01-02", "b")] == 15.0


def test_zscore_constant_cross_section_returns_null_or_nan() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-01", "b", 1.0),
        ]
    )

    graph = zscore(source)
    graph.compute()

    assert (
        graph.output.data["value"].null_count()
        + graph.output.data["value"].is_nan().sum()
        == 2
    )
