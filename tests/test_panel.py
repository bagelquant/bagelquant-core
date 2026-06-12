from __future__ import annotations

import polars as pl

from bagelquant_core import CategoryPanel, Domain, Panel


def test_panel_normalizes_to_time_asset_id_grid() -> None:
    domain = Domain(calendar=["2024-01-01", "2024-01-02"], universe=["a", "b"])
    panel = Panel.from_domain(
        pl.DataFrame({"time": ["2024-01-01"], "asset_id": ["a"], "value": [1.0]}),
        domain,
    )

    assert panel.data.columns == ["time", "asset_id", "value"]
    assert panel.data.height == 4
    assert panel.data.filter(pl.col("asset_id") == "a")["value"].to_list()[0] == 1.0


def test_dynamic_membership_masks_inactive_rows() -> None:
    domain = Domain(
        calendar=["2024-01-01"],
        universe=pl.DataFrame(
            {
                "time": ["2024-01-01", "2024-01-01"],
                "asset_id": ["a", "b"],
                "active": [True, False],
            }
        ),
    )
    panel = Panel.from_domain(
        pl.DataFrame(
            {
                "time": ["2024-01-01", "2024-01-01"],
                "asset_id": ["a", "b"],
                "value": [1.0, 2.0],
            }
        ),
        domain,
    )

    assert panel.data.to_dicts() == [
        {"time": panel.data["time"][0], "asset_id": "a", "value": 1.0}
    ]


def test_category_panel_accepts_non_numeric_values() -> None:
    domain = Domain(calendar=["2024-01-01"], universe=["a"])
    panel = CategoryPanel.from_domain(
        pl.DataFrame({"time": ["2024-01-01"], "asset_id": ["a"], "value": ["tech"]}),
        domain,
    )

    assert panel.data["value"].to_list() == ["tech"]
