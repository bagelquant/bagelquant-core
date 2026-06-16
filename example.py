from __future__ import annotations

import math
from datetime import date

import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import div, weighted_mean
from bagelquant_core.transformer import rank, rolling_mean, winsorize, zscore


def _build_dynamic_membership(
    calendar: pl.Series,
    assets: list[str],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for asset_index, asset_id in enumerate(assets):
        start_offset = asset_index % 45
        end_offset = len(calendar) - 1 - (asset_index % 30)
        suspension_stride = 17 + (asset_index % 7)

        for day_index, session in enumerate(calendar):
            within_life = start_offset <= day_index <= end_offset
            not_suspended = (day_index + asset_index) % suspension_stride != 0
            if within_life and not_suspended:
                rows.append(
                    {
                        "time": session,
                        "asset_id": asset_id,
                        "active": True,
                    }
                )

    return pl.DataFrame(rows)


def _build_factor_inputs(
    calendar: pl.Series,
    assets: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    price_rows: list[dict[str, object]] = []
    book_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    for asset_index, asset_id in enumerate(assets):
        base_price = 25.0 + asset_index * 1.75
        base_book = 8.0 + asset_index * 0.35
        for day_index, session in enumerate(calendar):
            seasonal = math.sin((day_index + asset_index) / 19.0)
            trend = day_index * (0.015 + asset_index * 0.0002)
            price = base_price + trend + seasonal
            book = base_book + day_index * 0.006 + math.cos(day_index / 23.0) * 0.2
            quality = 0.45 + (asset_index % 11) * 0.035 + seasonal * 0.04

            price_rows.append(
                {"time": session, "asset_id": asset_id, "value": round(price, 4)}
            )
            book_rows.append(
                {"time": session, "asset_id": asset_id, "value": round(book, 4)}
            )
            quality_rows.append(
                {"time": session, "asset_id": asset_id, "value": round(quality, 4)}
            )

    return (
        pl.DataFrame(price_rows),
        pl.DataFrame(book_rows),
        pl.DataFrame(quality_rows),
    )


def main() -> None:
    calendar = pl.date_range(
        date(2024, 1, 1),
        date(2024, 12, 31),
        interval="1d",
        eager=True,
    )
    assets = [f"ASSET{i:03d}" for i in range(1, 51)]
    membership = _build_dynamic_membership(calendar, assets)
    domain = Domain(calendar=calendar, universe=membership)
    price_df, book_df, quality_df = _build_factor_inputs(calendar, assets)

    price = Panel.from_domain(price_df, domain, name="price")
    book = Panel.from_domain(book_df, domain, name="book")
    quality = Panel.from_domain(quality_df, domain, name="quality")

    bm_ratio = div(book, price, name="bm_ratio")
    value_factor = rank(zscore(winsorize(bm_ratio)), name="value_factor")
    quality_factor = rank(zscore(quality), name="quality_factor")
    composite = weighted_mean(
        value_factor,
        quality_factor,
        weights=[0.65, 0.35],
        name="composite",
    )
    signal = rolling_mean(composite, window=20, min_periods=5, name="signal")
    signal.compute()

    print(f"calendar sessions: {len(calendar)}")
    print(f"assets in dynamic universe: {len(domain.asset_ids)}")
    print(f"active membership rows: {membership.height}")
    for row in signal.output.data.tail(20).to_dicts():
        print(row)


if __name__ == "__main__":
    main()
