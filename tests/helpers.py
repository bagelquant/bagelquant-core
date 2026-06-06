from __future__ import annotations

from typing import Any

import pandas as pd

from bagelquant_core import CategoryPanel, Domain, Panel


def domain_for(columns: pd.Index, rows: int) -> Domain:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    return Domain(
        calendar=dates,
        universe=list(columns),
    )


def make_panel(
    data: pd.DataFrame,
    *,
    name: str | None = None,
) -> Panel:
    domain = domain_for(data.columns, len(data))
    normalized = data.copy()
    normalized.index = domain.sessions
    return Panel.from_domain(normalized, domain, name=name)


def make_category_panel(
    data: pd.DataFrame,
    *,
    name: str | None = None,
) -> CategoryPanel:
    domain = domain_for(data.columns, len(data))
    normalized = data.copy()
    normalized.index = domain.sessions
    return CategoryPanel.from_domain(normalized, domain, name=name)
