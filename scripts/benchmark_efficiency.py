from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bagelquant_core import Domain, ExecutionRuntime, Graph, Panel
from bagelquant_core.composer import (
    add,
    coalesce,
    orthogonalize,
    rolling_corr,
    rolling_cov,
    rolling_elastic_net,
    rolling_lasso,
    rolling_ols,
    rolling_ridge,
    sum_frames,
    weighted_sum,
)
from bagelquant_core.transformer import (
    abs_value,
    constant,
    ewm_mean,
    negate,
    rolling_mean,
    rolling_percentile,
    rolling_rank,
    rolling_zscore,
    rank as cross_section_rank,
    winsorize,
    zscore,
)


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    rows: int
    assets: int
    repeats: int
    windows: tuple[int, ...]


PROFILES = {
    "smoke": BenchmarkProfile(
        rows=100_000,
        assets=500,
        repeats=3,
        windows=(20,),
    ),
    "comparison": BenchmarkProfile(
        rows=500_000,
        assets=2_000,
        repeats=3,
        windows=(20,),
    ),
    "cn-daily": BenchmarkProfile(
        rows=6_250_000,
        assets=5_000,
        repeats=1,
        windows=(20, 60, 252),
    ),
}

DEFAULT_CASES = (
    "domain_materialization",
    "rolling_mean",
    "rolling_rank",
    "rolling_percentile",
    "rolling_zscore",
    "ewm_mean",
    "zscore_add",
    "rolling_corr",
    "rolling_ols",
    "sum_10",
    "runtime_cache_miss",
    "runtime_cache_hit",
)
AVAILABLE_CASES = (
    *DEFAULT_CASES,
    "rolling_ols_3f",
    "rolling_ridge",
    "rolling_lasso",
    "rolling_elastic_net",
    "orthogonalize_3f",
    "traced_rolling_mean",
    "traced_multi_output_5",
    "eager_rank_pair",
    "rolling_chain_10",
    "pointwise_chain_20",
    "rolling_chain_20",
    "ewm_chain_20",
    "mixed_chain_20",
    "semantic_cse_5",
    "dense_exact_output",
    "sparse_alignment",
    "rolling_ols_sparse_3f",
    "rolling_ridge_sparse",
    "orthogonalize_sparse_3f",
    "rolling_lasso_1f",
    "rolling_lasso_8f",
    "rolling_lasso_iter_1",
    "rolling_lasso_iter_1000",
    "rolling_elastic_net_1f",
    "rolling_elastic_net_8f",
    "same_key_weighted_10",
    "same_key_coalesce_10",
    "sparse_sum_10",
    "rolling_cov",
    "cross_section_chain_20",
    "eager_cse_ols_5",
    "eager_cse_lasso_5",
    "eager_cse_orthogonalize_5",
)
WINDOW_CASES = {
    "rolling_mean",
    "rolling_rank",
    "rolling_percentile",
    "rolling_zscore",
    "rolling_corr",
    "rolling_ols",
    "rolling_ols_3f",
    "rolling_ridge",
    "rolling_lasso",
    "rolling_elastic_net",
    "traced_rolling_mean",
    "traced_multi_output_5",
    "eager_rank_pair",
    "rolling_chain_10",
    "rolling_chain_20",
    "mixed_chain_20",
    "semantic_cse_5",
    "dense_exact_output",
    "rolling_ols_sparse_3f",
    "rolling_ridge_sparse",
    "rolling_lasso_1f",
    "rolling_lasso_8f",
    "rolling_lasso_iter_1",
    "rolling_lasso_iter_1000",
    "rolling_elastic_net_1f",
    "rolling_elastic_net_8f",
    "rolling_cov",
    "eager_cse_ols_5",
    "eager_cse_lasso_5",
}


def make_panels(
    rows: int,
    assets: int,
    *,
    dynamic: bool,
) -> tuple[Panel, Panel]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if assets <= 0:
        raise ValueError("assets must be positive")
    periods = max(rows // assets, 1)
    dates = pl.date_range(
        start=pl.date(2010, 1, 1),
        end=pl.date(2010, 1, 1) + pl.duration(days=periods - 1),
        interval="1d",
        eager=True,
    )
    asset_ids = [f"asset_{index:05d}" for index in range(assets)]
    static_domain = Domain(calendar=dates, universe=asset_ids)
    keys = static_domain.grid_lazy().collect().with_row_index("__row")
    if dynamic:
        membership = keys.select(
            "time",
            "asset_id",
            ((pl.col("__row") * 7 + 3) % 10 < 7).alias("active"),
        )
        domain = Domain(calendar=dates, universe=membership)
        keys = domain.grid_lazy().collect().with_row_index("__row")
    else:
        domain = static_domain

    values = np.random.default_rng(0).normal(size=len(keys))
    values[::29] = 0.0
    values[::97] = np.nan
    frame = keys.select("time", "asset_id").with_columns(
        pl.Series("value", values)
    )
    base = Panel.from_domain(frame, domain, name="benchmark")
    other = Panel.from_domain(
        frame.with_columns((pl.col("value") * 1.5 + 2.0).alias("value")),
        domain,
        name="other",
    )
    return base, other


def _case_runner(
    case: str,
    base: Panel,
    other: Panel,
    *,
    window: int,
) -> Callable[
    [],
    tuple[Panel | Mapping[str, Panel], ExecutionRuntime | None],
]:
    min_periods = max(1, (window * 4 + 4) // 5)
    square = Panel.from_domain(
        base.lazy(dense=False).with_columns(
            (pl.col("value") ** 2).alias("value")
        ),
        base.domain,
        name="square",
    )
    cube = Panel.from_domain(
        base.lazy(dense=False).with_columns(
            (pl.col("value") ** 3).alias("value")
        ),
        base.domain,
        name="cube",
    )
    sparse_square = Panel.from_domain(
        square.lazy(dense=False).filter(
            pl.col("value").is_not_null()
            & ((pl.col("value").abs() * 100).cast(pl.Int64) % 3 != 0)
        ),
        base.domain,
        name="sparse_square",
    )
    sparse_cube = Panel.from_domain(
        cube.lazy(dense=False).filter(
            pl.col("value").is_not_null()
            & ((pl.col("value").abs() * 100).cast(pl.Int64) % 5 != 0)
        ),
        base.domain,
        name="sparse_cube",
    )
    independent_factors: tuple[Panel, ...] = ()
    if case in {
        "rolling_lasso_8f",
        "rolling_elastic_net_8f",
    }:
        keys = base.data.select("time", "asset_id")
        factors: list[Panel] = []
        for index in range(8):
            factor_values = np.random.default_rng(100 + index).normal(
                size=len(keys)
            )
            factor_values[(index + 17) :: 131] = np.nan
            factors.append(
                Panel.from_domain(
                    keys.with_columns(pl.Series("value", factor_values)),
                    base.domain,
                    name=f"independent_factor_{index}",
                )
            )
        independent_factors = tuple(factors)
    traced = Panel.from_domain(
        base.lazy(dense=False).with_columns(
            pl.col("time").alias("observation_date"),
            pl.col("time").alias("base_available_date"),
        ),
        base.domain,
        name="traced",
        trace_columns=("observation_date", "base_available_date"),
    )
    if case == "runtime_cache_hit":
        graph = zscore(add(base, other), name="cached_zscore")
        runtime = ExecutionRuntime()
        graph.compute(runtime=runtime)

        def cached() -> tuple[Panel, ExecutionRuntime]:
            return graph.compute(runtime=runtime), runtime

        return cached

    def execute() -> tuple[
        Panel | Mapping[str, Panel],
        ExecutionRuntime | None,
    ]:
        if case == "domain_materialization":
            output = Panel.from_domain(base.data, base.domain).output
            output.data
            return output, None

        runtime = ExecutionRuntime()
        if case == "rolling_mean":
            graph = rolling_mean(
                base,
                window=window,
                min_periods=min_periods,
            )
        elif case == "rolling_rank":
            graph = rolling_rank(
                base,
                window=window,
                min_periods=min_periods,
            )
        elif case == "rolling_percentile":
            graph = rolling_percentile(
                base,
                window=window,
                min_periods=min_periods,
            )
        elif case == "rolling_zscore":
            graph = rolling_zscore(
                base,
                window=window,
                min_periods=min_periods,
            )
        elif case == "ewm_mean":
            graph = ewm_mean(base, alpha=0.2)
        elif case == "zscore_add":
            graph = zscore(add(base, other))
        elif case == "rolling_corr":
            graph = rolling_corr(
                base,
                other,
                window=window,
                min_periods=min_periods,
            )
        elif case == "rolling_ols":
            graph = rolling_ols(other, base, window=window)
        elif case == "rolling_ols_3f":
            graph = rolling_ols(
                other,
                base,
                square,
                cube,
                window=window,
            )
        elif case == "rolling_ols_sparse_3f":
            graph = rolling_ols(
                other,
                base,
                sparse_square,
                sparse_cube,
                window=window,
            )
        elif case == "rolling_ridge":
            graph = rolling_ridge(other, base, square, window=window)
        elif case == "rolling_ridge_sparse":
            graph = rolling_ridge(
                other,
                base,
                sparse_square,
                window=window,
            )
        elif case == "rolling_lasso":
            graph = rolling_lasso(
                other,
                base,
                square,
                window=window,
                max_iter=100,
            )
        elif case in {
            "rolling_lasso_1f",
            "rolling_lasso_iter_1",
            "rolling_lasso_iter_1000",
        }:
            iterations = {
                "rolling_lasso_1f": 100,
                "rolling_lasso_iter_1": 1,
                "rolling_lasso_iter_1000": 1000,
            }[case]
            graph = rolling_lasso(
                other,
                base,
                window=window,
                max_iter=iterations,
            )
        elif case == "rolling_lasso_8f":
            graph = rolling_lasso(
                other,
                *independent_factors,
                window=window,
                max_iter=100,
            )
        elif case == "rolling_elastic_net":
            graph = rolling_elastic_net(
                other,
                base,
                square,
                window=window,
                max_iter=100,
            )
        elif case == "rolling_elastic_net_1f":
            graph = rolling_elastic_net(
                other,
                base,
                window=window,
                max_iter=100,
            )
        elif case == "rolling_elastic_net_8f":
            graph = rolling_elastic_net(
                other,
                *independent_factors,
                window=window,
                max_iter=100,
            )
        elif case == "orthogonalize_3f":
            graph = orthogonalize(other, base, square, cube)
        elif case == "orthogonalize_sparse_3f":
            graph = orthogonalize(
                other,
                base,
                sparse_square,
                sparse_cube,
            )
        elif case == "eager_cse_ols_5":
            graph = Graph(
                outputs=[
                    rolling_ols(
                        other,
                        base,
                        square,
                        window=window,
                        name=f"eager_ols_{index}",
                    )
                    for index in range(5)
                ]
            )
        elif case == "eager_cse_lasso_5":
            graph = Graph(
                outputs=[
                    rolling_lasso(
                        other,
                        base,
                        square,
                        window=window,
                        max_iter=100,
                        name=f"eager_lasso_{index}",
                    )
                    for index in range(5)
                ]
            )
        elif case == "eager_cse_orthogonalize_5":
            graph = Graph(
                outputs=[
                    orthogonalize(
                        other,
                        base,
                        square,
                        cube,
                        name=f"eager_orthogonalize_{index}",
                    )
                    for index in range(5)
                ]
            )
        elif case == "traced_rolling_mean":
            graph = rolling_mean(
                traced,
                window=window,
                min_periods=min_periods,
            )
        elif case == "traced_multi_output_5":
            shared = rolling_mean(
                traced,
                window=window,
                min_periods=min_periods,
            )
            graph = Graph(
                outputs=[
                    add(
                        shared,
                        constant(shared, value=float(index)),
                        name=f"traced_output_{index}",
                    )
                    for index in range(5)
                ]
            )
        elif case == "eager_rank_pair":
            graph = Graph(
                outputs=[
                    rolling_rank(
                        base,
                        window=window,
                        min_periods=min_periods,
                        name="rank",
                    ),
                    rolling_percentile(
                        base,
                        window=window,
                        min_periods=min_periods,
                        name="percentile",
                    ),
                ]
            )
        elif case == "rolling_chain_10":
            graph = base
            for _ in range(10):
                graph = rolling_mean(
                    graph,
                    window=window,
                    min_periods=min_periods,
                )
        elif case == "pointwise_chain_20":
            graph = base
            for index in range(20):
                graph = (
                    negate(graph)
                    if index % 2 == 0
                    else abs_value(graph)
                )
        elif case == "rolling_chain_20":
            graph = base
            for _ in range(20):
                graph = rolling_mean(
                    graph,
                    window=window,
                    min_periods=min_periods,
                )
        elif case == "ewm_chain_20":
            graph = base
            for _ in range(20):
                graph = ewm_mean(graph, alpha=0.2)
        elif case == "mixed_chain_20":
            graph = base
            for index in range(20):
                if index % 4 == 0:
                    graph = rolling_mean(
                        graph,
                        window=window,
                        min_periods=min_periods,
                    )
                elif index % 4 == 1:
                    graph = negate(graph)
                elif index % 4 == 2:
                    graph = ewm_mean(graph, alpha=0.2)
                else:
                    graph = abs_value(graph)
        elif case == "semantic_cse_5":
            graph = Graph(
                outputs=[
                    rolling_mean(
                        base,
                        window=window,
                        min_periods=min_periods,
                        name=f"semantic_cse_{index}",
                    )
                    for index in range(5)
                ]
            )
        elif case == "dense_exact_output":
            graph = rolling_mean(
                base,
                window=window,
                min_periods=min_periods,
            )
        elif case == "sparse_alignment":
            graph = add(sparse_square, sparse_cube)
        elif case == "same_key_weighted_10":
            graph = weighted_sum(
                *([base, other] * 5),
                weights=[0.1] * 10,
            )
        elif case == "same_key_coalesce_10":
            graph = coalesce(*([base, other] * 5))
        elif case == "sparse_sum_10":
            graph = sum_frames(*([sparse_square, sparse_cube] * 5))
        elif case == "rolling_cov":
            graph = rolling_cov(
                base,
                other,
                window=window,
                min_periods=min_periods,
            )
        elif case == "cross_section_chain_20":
            graph = base
            for index in range(20):
                graph = (
                    cross_section_rank(graph)
                    if index % 2 == 0
                    else winsorize(graph)
                )
        elif case == "sum_10":
            graph = sum_frames(*([base, other] * 5))
        elif case == "runtime_cache_miss":
            graph = zscore(add(base, other), name="cached_zscore")
        else:
            raise ValueError(f"unknown benchmark case: {case}")
        return graph.compute(runtime=runtime), runtime

    return execute


def _panel_summary(output: Panel) -> dict[str, int | float | None]:
    frame = output.collect(dense=False)
    values = frame.get_column("value")
    numeric = values.fill_nan(None)
    return {
        "rows": frame.height,
        "nulls": numeric.null_count(),
        "sum": numeric.sum(),
        "mean": numeric.mean(),
    }


def _output_summary(
    output: Panel | Mapping[str, Panel],
) -> dict[str, Any]:
    if isinstance(output, Panel):
        return _panel_summary(output)
    return {
        name: _panel_summary(panel)
        for name, panel in output.items()
    }


def _plan_sort_count(output: Panel | Mapping[str, Panel]) -> int:
    panels = [output] if isinstance(output, Panel) else output.values()
    return sum(
        panel.lazy(dense=False)
        .explain(optimized=True)
        .upper()
        .count("SORT BY")
        for panel in panels
    )


def _peak_rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)

    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _run_worker(args: argparse.Namespace) -> None:
    base, other = make_panels(
        args.rows,
        args.assets,
        dynamic=args.worker_universe == "dynamic",
    )
    runner = _case_runner(
        args.worker_case,
        base,
        other,
        window=args.worker_window,
    )
    timings: list[float] = []
    output: Panel | Mapping[str, Panel] | None = None
    runtime: ExecutionRuntime | None = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        output, runtime = runner()
        timings.append(time.perf_counter() - start)
    assert output is not None
    payload = {
        "name": args.worker_case,
        "universe": args.worker_universe,
        "window": (
            args.worker_window
            if args.worker_case in WINDOW_CASES
            else None
        ),
        "best_seconds": min(timings),
        "median_seconds": statistics.median(timings),
        "mean_seconds": statistics.fmean(timings),
        "peak_rss_bytes": _peak_rss_bytes(),
        "materializations": 0 if runtime is None else runtime.materializations,
        "eager_barriers": 0 if runtime is None else runtime.eager_barriers,
        "sort_nodes": _plan_sort_count(output),
        "runtime_diagnostics": (
            {}
            if runtime is None
            else dict(runtime._diagnostics)
        ),
        "output": _output_summary(output),
    }
    print(json.dumps(payload, sort_keys=True))


def _run_isolated(
    *,
    case: str,
    universe: str,
    window: int,
    rows: int,
    assets: int,
    repeats: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        case,
        "--worker-universe",
        universe,
        "--worker-window",
        str(window),
        "--rows",
        str(rows),
        "--assets",
        str(assets),
        "--repeats",
        str(repeats),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _selected_universes(value: str) -> tuple[str, ...]:
    return ("static", "dynamic") if value == "both" else (value,)


def _selected_cases(values: Sequence[str] | None) -> tuple[str, ...]:
    selected = DEFAULT_CASES if not values else tuple(values)
    unknown = sorted(set(selected) - set(AVAILABLE_CASES))
    if unknown:
        raise ValueError(f"unknown benchmark cases: {unknown}")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BagelQuant isolated computation benchmarks"
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="smoke",
    )
    parser.add_argument("--rows", type=int)
    parser.add_argument("--assets", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--windows", type=int, nargs="+")
    parser.add_argument(
        "--universe",
        choices=("static", "dynamic", "both"),
        default="static",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="benchmark one named case; repeat to select multiple cases",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable result",
    )
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-universe",
        choices=("static", "dynamic"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-window", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    profile = PROFILES[args.profile]
    args.rows = profile.rows if args.rows is None else args.rows
    args.assets = profile.assets if args.assets is None else args.assets
    args.repeats = profile.repeats if args.repeats is None else args.repeats
    windows = profile.windows if args.windows is None else tuple(args.windows)
    if args.worker_case:
        if args.worker_universe is None or args.worker_window is None:
            raise ValueError("worker universe and window are required")
        _run_worker(args)
        return

    measurements: list[dict[str, Any]] = []
    for universe in _selected_universes(args.universe):
        for case in _selected_cases(args.case):
            case_windows = windows if case in WINDOW_CASES else (windows[0],)
            for window in case_windows:
                measurements.append(
                    _run_isolated(
                        case=case,
                        universe=universe,
                        window=window,
                        rows=args.rows,
                        assets=args.assets,
                        repeats=args.repeats,
                    )
                )

    payload = {
        "profile": args.profile,
        "requested_rows": args.rows,
        "assets": args.assets,
        "repeats": args.repeats,
        "windows": list(windows),
        "measurements": measurements,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        f"profile={args.profile} requested_rows={args.rows} "
        f"assets={args.assets} repeats={args.repeats}"
    )
    for item in measurements:
        window = "" if item["window"] is None else f" window={item['window']:3d}"
        diagnostics = item["runtime_diagnostics"]
        print(
            f"{item['name']:24s} {item['universe']:7s}{window:11s} "
            f"median={item['median_seconds']:8.4f}s "
            f"rss={item['peak_rss_bytes'] / (1024**2):8.1f}MiB "
            f"mat={item['materializations']} eager={item['eager_barriers']} "
            f"sort={item['sort_nodes']} "
            f"elided_sort={diagnostics.get('sorts_elided', 0)} "
            f"elided_align={diagnostics.get('alignments_elided', 0)} "
            f"cse={diagnostics.get('semantic_cse_hits', 0)} "
            f"pos={diagnostics.get('positional_composer_hits', 0)} "
            f"eager_cse={diagnostics.get('eager_cse_hits', 0)} "
            f"solver_batches={diagnostics.get('solver_batches', 0)}"
        )


if __name__ == "__main__":
    main()
