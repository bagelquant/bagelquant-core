from __future__ import annotations

from datetime import date

import pytest
import polars as pl

from bagelquant_core import CategoryPanel, Domain, Panel
from bagelquant_core.composer import add
from bagelquant_core.transformer import rolling_mean


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


def test_dynamic_membership_treats_missing_rows_as_inactive() -> None:
    domain = Domain(
        calendar=["2024-01-01"],
        universe=pl.DataFrame(
            {
                "time": ["2024-01-01"],
                "asset_id": ["a"],
                "active": [True],
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

    assert domain.asset_ids.to_list() == ["a"]
    assert panel.data.to_dicts() == [
        {"time": panel.data["time"][0], "asset_id": "a", "value": 1.0}
    ]


def test_dynamic_membership_is_not_forward_filled() -> None:
    domain = Domain(
        calendar=["2024-01-01", "2024-01-02"],
        universe=pl.DataFrame(
            {
                "time": ["2024-01-01"],
                "asset_id": ["a"],
                "active": [True],
            }
        ),
    )

    assert domain.membership.sort(["time", "asset_id"]).to_dicts() == [
        {"time": domain.times[0], "asset_id": "a", "active": True},
        {"time": domain.times[1], "asset_id": "a", "active": False},
    ]


def test_dynamic_universe_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        Domain(
            calendar=["2024-01-01"],
            universe=pl.DataFrame({"time": ["2024-01-01"], "asset_id": ["a"]}),
        )


def test_dynamic_universe_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match=r"unique by \(time, asset_id\)"):
        Domain(
            calendar=["2024-01-01"],
            universe=pl.DataFrame(
                {
                    "time": ["2024-01-01", "2024-01-01"],
                    "asset_id": ["a", "a"],
                    "active": [True, False],
                }
            ),
        )


def test_dynamic_universe_rejects_invalid_keys() -> None:
    with pytest.raises(ValueError, match="keys must be valid"):
        Domain(
            calendar=["2024-01-01"],
            universe=pl.DataFrame(
                {
                    "time": ["not-a-date"],
                    "asset_id": ["a"],
                    "active": [True],
                }
            ),
        )

    with pytest.raises(ValueError, match="keys must be valid"):
        Domain(
            calendar=["2024-01-01"],
            universe=pl.DataFrame(
                {
                    "time": ["2024-01-01"],
                    "asset_id": [None],
                    "active": [True],
                }
            ),
        )


def test_dynamic_membership_is_reapplied_after_transformers() -> None:
    domain = Domain(
        calendar=["2024-01-01", "2024-01-02"],
        universe=pl.DataFrame(
            {
                "time": ["2024-01-01"],
                "asset_id": ["a"],
                "active": [True],
            }
        ),
    )
    source = Panel.from_domain(
        pl.DataFrame(
            {
                "time": ["2024-01-01"],
                "asset_id": ["a"],
                "value": [1.0],
            }
        ),
        domain,
    )

    graph = rolling_mean(source, window=2, min_periods=1)
    graph.compute()

    assert graph.output.data.to_dicts() == [
        {"time": domain.times[0], "asset_id": "a", "value": 1.0}
    ]


def test_dynamic_membership_is_reapplied_after_composers() -> None:
    domain = Domain(
        calendar=["2024-01-01", "2024-01-02"],
        universe=pl.DataFrame(
            {
                "time": ["2024-01-01"],
                "asset_id": ["a"],
                "active": [True],
            }
        ),
    )
    left = Panel.from_domain(
        pl.DataFrame({"time": ["2024-01-01"], "asset_id": ["a"], "value": [1.0]}),
        domain,
    )
    right = Panel.from_domain(
        pl.DataFrame({"time": ["2024-01-01"], "asset_id": ["a"], "value": [2.0]}),
        domain,
    )

    graph = add(left, right)
    graph.compute()

    assert graph.output.data.to_dicts() == [
        {"time": domain.times[0], "asset_id": "a", "value": 3.0}
    ]


def test_dynamic_domain_signatures_include_membership() -> None:
    left = Domain(
        calendar=["2024-01-01"],
        universe=pl.DataFrame(
            {"time": ["2024-01-01"], "asset_id": ["a"], "active": [True]}
        ),
    )
    right = Domain(
        calendar=["2024-01-01"],
        universe=pl.DataFrame(
            {"time": ["2024-01-01"], "asset_id": ["a"], "active": [False]}
        ),
    )

    assert left.signature != right.signature


def test_domain_accepts_series_calendar_and_static_universe() -> None:
    domain = Domain(
        calendar=pl.Series("sessions", ["2024-01-01", "2024-01-02"]),
        universe=pl.Series("symbols", ["b", "a"]),
    )

    assert domain.times.to_list() == [
        date(2024, 1, 1),
        date(2024, 1, 2),
    ]
    assert domain.asset_ids.to_list() == ["a", "b"]
    assert domain.membership.sort(["time", "asset_id"]).to_dicts() == [
        {"time": domain.times[0], "asset_id": "a", "active": True},
        {"time": domain.times[0], "asset_id": "b", "active": True},
        {"time": domain.times[1], "asset_id": "a", "active": True},
        {"time": domain.times[1], "asset_id": "b", "active": True},
    ]


def test_category_panel_accepts_non_numeric_values() -> None:
    domain = Domain(calendar=["2024-01-01"], universe=["a"])
    panel = CategoryPanel.from_domain(
        pl.DataFrame({"time": ["2024-01-01"], "asset_id": ["a"], "value": ["tech"]}),
        domain,
    )

    assert panel.data["value"].to_list() == ["tech"]
