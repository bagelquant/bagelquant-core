from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import polars as pl

from bagelquant_core import Domain, ExecutionRuntime, Panel
from bagelquant_core.composer import add, rolling_corr, rolling_ols
from bagelquant_core.transformer import ewm_mean, rolling_mean, rolling_rank, zscore


def make_panel(rows: int, assets: int) -> Panel:
    periods = max(rows // assets, 1)
    dates = pl.date_range(
        start=pl.date(2020, 1, 1),
        end=pl.date(2020, 1, 1) + pl.duration(days=periods - 1),
        interval="1d",
        eager=True,
    )
    asset_ids = [f"asset_{index:05d}" for index in range(assets)]
    domain = Domain(calendar=dates, universe=asset_ids)
    frame = domain.membership.select("time", "asset_id").with_columns(
        pl.Series("value", np.random.default_rng(0).normal(size=periods * assets))
    )
    return Panel.from_domain(frame, domain, name="benchmark")


def measure(label: str, func: Callable[[], object], repeats: int) -> None:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    best = min(timings)
    avg = sum(timings) / len(timings)
    print(f"{label:28s} best={best:8.4f}s avg={avg:8.4f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="BagelQuant efficiency smoke benchmark")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--assets", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    base = make_panel(args.rows, args.assets)
    other = Panel.from_domain(
        base.data.with_columns((pl.col("value") * 1.5 + 2.0).alias("value")),
        base.domain,
        name="other",
    )
    runtime = ExecutionRuntime()

    print(f"rows={base.data.height} assets={args.assets} repeats={args.repeats}")
    measure("domain materialization", lambda: Panel.from_domain(base.data, base.domain), args.repeats)
    measure("rolling_mean", lambda: rolling_mean(base, window=20).compute(), args.repeats)
    measure("rolling_rank", lambda: rolling_rank(base, window=20).compute(), args.repeats)
    measure("ewm_mean", lambda: ewm_mean(base, alpha=0.2).compute(), args.repeats)
    measure("zscore(add)", lambda: zscore(add(base, other)).compute(), args.repeats)
    measure("rolling_corr", lambda: rolling_corr(base, other, window=20).compute(), args.repeats)
    measure("rolling_ols", lambda: rolling_ols(other, base, window=20).compute(), args.repeats)
    cached = zscore(add(base, other), name="cached_zscore")
    measure("runtime cache miss", lambda: cached.compute(runtime=runtime), 1)
    measure("runtime cache hit", lambda: cached.compute(runtime=runtime), args.repeats)


if __name__ == "__main__":
    main()
