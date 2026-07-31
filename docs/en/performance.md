# Performance Notes

Benchmarks run each case in an isolated child process so setup state, cached
graphs, and peak resident memory are attributable to one operation. Synthetic
inputs use a fixed seed and include nulls and repeated values.
Rolling cases use `min_periods` equal to 80% of the selected window so the
null-bearing input still produces representative output.

```bash
uv run python scripts/benchmark_efficiency.py --profile smoke --json
uv run python scripts/benchmark_efficiency.py --profile comparison --json
uv run python scripts/benchmark_efficiency.py \
  --profile cn-daily \
  --case rolling_zscore \
  --case rolling_rank \
  --case rolling_ols \
  --json
uv run python scripts/benchmark_efficiency.py \
  --profile comparison \
  --case sum_10
```

Use `--universe dynamic` or `--universe both` to exercise sparse dynamic
membership. The `cn-daily` profile represents 6.25 million requested rows,
5,000 assets, and 20-, 60-, and 252-session windows. `--rows`, `--assets`,
`--repeats`, and `--windows` override profile defaults.

The JSON output records same-machine best, median, and mean wall time; peak RSS;
output row/null/value summaries; and runtime materialization and eager-barrier
counts. Time-based thresholds remain outside CI. Structural tests assert that
lazy graphs have one final materialization and eager graphs add exactly their
declared barriers.

## July 2026 comparison

The 500,000-row, 2,000-asset comparison profile produced these local medians
on an Apple M1 Max with 32 GB of memory, macOS 26.5.2, Python 3.13.14,
NumPy 2.4.6, and Polars 1.41.2:

| Operation | Previous baseline | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| `rolling_zscore(window=20)` | 8.6–10.9s | 0.101s | at least 85x |
| `rolling_rank(window=20)` | 3.20s | 0.462s | 6.9x |
| `rolling_percentile(window=20)` | approximately 3.20s | 0.510s | approximately 6.3x |
| `rolling_ols(window=20, factors=1)` | 10.62s | 0.619s | 17.2x |
| `sum_10` | 0.167s | 0.143s | 1.2x |

These values are regression references, not portable performance guarantees.
The z-score path uses native Polars rolling expressions. Rank and percentile
use bounded sliding-window NumPy batches. Multi-factor OLS uses bounded batched
least-squares evaluation. Single-factor OLS uses rolling sufficient statistics
and falls back to NumPy least squares for rank-deficient or numerically
unstable windows. Each vectorized path limits its explicit temporary numeric
work buffers to approximately 64 MiB.

The 6.25-million-row, 5,000-asset profile completed single-factor OLS with a
252-session window in 4.88 seconds with approximately 3.4 GiB peak RSS. Its
temporary numeric arrays scale with input rows plus the bounded batch, rather
than with total rows multiplied by window size.

Wide n-ary aggregations build a balanced inner-join tree. This reduces lazy
plan depth from linear to logarithmic while preserving sparse inner alignment,
output ordering, and public composer behavior. Benefits grow with graph width;
the ten-input comparison case above is intentionally included as a regression
reference.
