from __future__ import annotations

import polars as pl

from bagelquant_core import Domain, Node


def panel(values: list[tuple[str, str, float]], *, name: str = "panel") -> Node:
    domain = Domain(
        calendar=sorted({time for time, _, _ in values}),
        universe=sorted({asset for _, asset, _ in values}),
    )
    return Node.from_domain(
        pl.DataFrame(values, schema=["time", "asset_id", "value"], orient="row"),
        domain,
        name=name,
    )


def values(frame: pl.DataFrame) -> dict[tuple[str, str], float | None]:
    return {
        (str(row["time"]), row["asset_id"]): row["value"]
        for row in frame.sort(["time", "asset_id"]).to_dicts()
    }
