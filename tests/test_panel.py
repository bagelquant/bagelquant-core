import numpy as np
import pandas as pd
import pytest

from bagelquant_core import CategoryPanel, Domain, Panel
from bagelquant_core.composer import add, weighted_sum
from bagelquant_core.panel import domain as domain_module
from bagelquant_core.transformer import fillna_zero


def domain(
    *,
    universe: list[str] | pd.DataFrame | None = None,
    region: str = "US",
    start_date: str = "2024-01-02",
    end_date: str = "2024-01-05",
) -> Domain:
    return Domain(
        region=region,
        universe=universe if universe is not None else ["a"],
        start_date=start_date,
        end_date=end_date,
    )


@pytest.mark.parametrize(
    ("region", "holiday"),
    [
        ("US", "2024-07-04"),
        ("CN", "2024-10-01"),
        ("HK", "2024-07-01"),
    ],
)
def test_domain_excludes_regional_exchange_holidays(region: str, holiday: str) -> None:
    scope = domain(
        region=region,
        start_date=holiday,
        end_date=holiday,
    )

    assert pd.Timestamp(holiday) not in scope.sessions


def test_domain_empty_frame_uses_trading_sessions_and_assets() -> None:
    scope = domain(universe=["a", "b"])

    frame = scope.empty_frame()

    assert frame.index.equals(scope.sessions)
    assert frame.columns.tolist() == ["a", "b"]
    assert frame.isna().all(axis=None)


def test_domain_reuses_cached_calendar_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(domain_module.CALENDAR_CACHE_ENV, str(tmp_path))
    scope = domain()

    def fail_on_retrieval(*args, **kwargs):
        raise AssertionError("cached sessions should avoid calendar retrieval")

    monkeypatch.setattr(domain_module.xcals, "get_calendar", fail_on_retrieval)

    cached_scope = domain()

    assert cached_scope.sessions.equals(scope.sessions)
    assert cached_scope.equivalent_to(scope)


def test_domain_refreshes_cache_when_range_expands(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(domain_module.CALENDAR_CACHE_ENV, str(tmp_path))
    domain_module._write_calendar_cache(
        "XNYS",
        pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    calls = []

    class Calendar:
        sessions = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        )

    def retrieve(*args, **kwargs):
        calls.append((args, kwargs))
        return Calendar()

    monkeypatch.setattr(domain_module.xcals, "get_calendar", retrieve)

    scope = domain(end_date="2024-01-05")

    assert scope.sessions.equals(Calendar.sessions)
    assert calls


def test_domain_recovers_from_malformed_calendar_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(domain_module.CALENDAR_CACHE_ENV, str(tmp_path))
    cache_path = domain_module._calendar_cache_path("XNYS")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not json")
    calls = []

    class Calendar:
        sessions = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        )

    def retrieve(*args, **kwargs):
        calls.append((args, kwargs))
        return Calendar()

    monkeypatch.setattr(domain_module.xcals, "get_calendar", retrieve)

    scope = domain()

    assert scope.sessions.equals(Calendar.sessions)
    assert calls


def test_dynamic_domain_treats_missing_membership_rows_as_inactive() -> None:
    membership = pd.DataFrame(
        {"a": [True], "b": [False]},
        index=pd.to_datetime(["2024-01-03"]),
    )
    scope = domain(universe=membership)

    assert scope.membership.loc[pd.Timestamp("2024-01-03")].tolist() == [True, False]
    assert not scope.membership.loc[pd.Timestamp("2024-01-04")].any()


def test_panel_requires_from_domain() -> None:
    with pytest.raises(TypeError, match="Panel.from_domain"):
        Panel(pd.DataFrame({"a": [1]}))


def test_panel_rejects_non_numeric_data() -> None:
    scope = domain(universe=["a", "b"])
    data = pd.DataFrame(
        {"a": [1, 2], "b": ["x", "y"]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    with pytest.raises(TypeError):
        Panel.from_domain(data, scope, name="bad")


def test_panel_from_domain_reindexes_and_masks_dynamic_input() -> None:
    membership = pd.DataFrame(
        {"a": [True], "b": [False]},
        index=pd.to_datetime(["2024-01-03"]),
    )
    scope = domain(universe=membership)
    data = pd.DataFrame(
        {"a": [1.0], "b": [2.0], "extra": [3.0]},
        index=pd.to_datetime(["2024-01-03"]),
    )

    panel = Panel.from_domain(data, scope)

    assert panel.data.index.equals(scope.sessions)
    assert panel.data.columns.tolist() == ["a", "b"]
    assert panel.data.loc[pd.Timestamp("2024-01-03"), "a"] == 1.0
    assert np.isnan(panel.data.loc[pd.Timestamp("2024-01-03"), "b"])


def test_category_panel_from_domain_masks_inactive_labels() -> None:
    membership = pd.DataFrame(
        {"a": [True], "b": [False]},
        index=pd.to_datetime(["2024-01-03"]),
    )
    scope = domain(universe=membership)
    data = pd.DataFrame(
        {"a": ["tech"], "b": ["finance"]},
        index=pd.to_datetime(["2024-01-03"]),
    )

    panel = CategoryPanel.from_domain(data, scope)

    assert panel.data.loc[pd.Timestamp("2024-01-03"), "a"] == "tech"
    assert pd.isna(panel.data.loc[pd.Timestamp("2024-01-03"), "b"])


def test_panel_data_is_a_defensive_copy() -> None:
    scope = domain()
    panel = Panel.from_domain(
        pd.DataFrame({"a": [1]}, index=pd.to_datetime(["2024-01-02"])),
        scope,
    )

    data = panel.data
    data.loc[pd.Timestamp("2024-01-02"), "a"] = 99

    assert panel.data.loc[pd.Timestamp("2024-01-02"), "a"] == 1


def test_derived_outputs_reapply_dynamic_membership_mask() -> None:
    membership = pd.DataFrame(
        {"a": [True, False]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    scope = domain(universe=membership, end_date="2024-01-03")
    panel = Panel.from_domain(
        pd.DataFrame(
            {"a": [1.0, 2.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
        scope,
    )

    result = fillna_zero(panel).compute()

    assert result.domain.equivalent_to(scope)
    assert np.isnan(result.data.loc[pd.Timestamp("2024-01-03"), "a"])


def test_static_domain_outputs_allow_fill_operations() -> None:
    scope = domain()
    panel = Panel.from_domain(scope.empty_frame(), scope)

    result = fillna_zero(panel).compute()

    assert result.data["a"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_composer_rejects_incompatible_domains() -> None:
    left_scope = domain(universe=["a"])
    right_scope = domain(universe=["b"])
    left = Panel.from_domain(left_scope.empty_frame().fillna(1), left_scope)
    right = Panel.from_domain(right_scope.empty_frame().fillna(2), right_scope)

    with pytest.raises(ValueError, match="equivalent Domains"):
        add(left, right).compute()


def test_weighted_sum_combines_compatible_domain_inputs() -> None:
    scope = domain()
    left = Panel.from_domain(scope.empty_frame().fillna(1.0), scope)
    right = Panel.from_domain(scope.empty_frame().fillna(3.0), scope)

    result = weighted_sum(left, right, weights=[0.25, 0.75]).compute()

    assert result.data["a"].tolist() == [2.5, 2.5, 2.5, 2.5]
    assert result.domain.equivalent_to(scope)
