# Performance Notes

Benchmarks are intentionally lightweight and reproducible:

```bash
uv run python scripts/benchmark_efficiency.py --rows 100000 --assets 500 --repeats 5 --json
```

The JSON output records same-machine medians for regression comparison.
Time-based thresholds should remain loose in CI; structural tests assert that
lazy graphs have one final materialization and eager graphs add exactly their
declared barriers.

```text
rolling_rank best=0.3035s
runtime cache hit best=0.0001s
```

The previous local baseline for `rolling_rank` was about `1.59s` on the same
100k-row synthetic panel. The improvement comes from replacing per-window
Python callbacks with a per-asset NumPy pass for `rolling_rank` and
`rolling_percentile`.
