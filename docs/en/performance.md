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
uv run python scripts/benchmark_efficiency.py \
  --profile comparison \
  --case rolling_ols_3f \
  --case rolling_ridge \
  --case rolling_lasso \
  --case rolling_elastic_net \
  --case orthogonalize_3f \
  --case traced_multi_output_5 \
  --case eager_rank_pair
```

Use `--universe dynamic` or `--universe both` to exercise sparse dynamic
membership. The `cn-daily` profile represents 6.25 million requested rows,
5,000 assets, and 20-, 60-, and 252-session windows. `--rows`, `--assets`,
`--repeats`, and `--windows` override profile defaults.

The JSON output records same-machine best, median, and mean wall time; peak RSS;
output row/null/value summaries; runtime materialization and eager-barrier
counts; and sort counts for inspectable lazy output plans. Time-based thresholds
remain outside CI. Structural tests assert that lazy graphs have one final
materialization and eager graphs add exactly their declared barriers.

## July 2026 comparison

The 500,000-row, 2,000-asset comparison profile produced these local medians
on an Apple M1 Max with 32 GB of memory, macOS 26.5.2, Python 3.13.14,
NumPy 2.4.6, and Polars 1.41.2:

| Operation | Previous baseline | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| `rolling_zscore(window=20)` | 8.6–10.9s | 0.101s | at least 85x |
| `rolling_rank(window=20)` | 3.20s | 0.462s | 6.9x |
| `rolling_percentile(window=20)` | approximately 3.20s | 0.510s | approximately 6.3x |
| `rolling_ols(window=20, factors=1)` | 10.62s | 0.373s | 28.5x |
| `sum_10` | 0.167s | 0.143s | 1.2x |

These values are regression references, not portable performance guarantees.
The z-score path uses native Polars rolling expressions. Rank and percentile
use bounded sliding-window NumPy batches. Multi-factor OLS uses bounded batched
least-squares evaluation. Single-factor OLS uses rolling sufficient statistics
and falls back to NumPy least squares for rank-deficient or numerically
unstable windows. Each vectorized path limits its explicit temporary numeric
work buffers to approximately 64 MiB.

The 6.25-million-row, 5,000-asset profile completed single-factor OLS with a
252-session window in 4.61 seconds with approximately 3.4 GiB peak RSS. Its
temporary numeric arrays scale with input rows plus the bounded batch, rather
than with total rows multiplied by window size.

Wide n-ary aggregations build a balanced inner-join tree. This reduces lazy
plan depth from linear to logarithmic while preserving sparse inner alignment,
output ordering, and public composer behavior. Benefits grow with graph width;
the ten-input comparison case above is intentionally included as a regression
reference.

## July 2026 second pass

The second pass removed the remaining row-wise regression, cross-sectional,
and rolling-rank paths. Same-case local medians were:

| Operation | Previous | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| two-factor `rolling_ols`, 500,000 rows | 2.665s | 0.811s | 3.3x |
| one-factor `rolling_ridge`, 100,000 rows | 3.226s | 0.112s | 28.8x |
| one-factor `rolling_lasso`, 20,000 rows, 100 iterations | 1.697s | 0.063s | 26.9x |
| one-factor `rolling_elastic_net`, 20,000 rows, 100 iterations | 1.674s | 0.059s | 28.4x |
| two-factor `orthogonalize`, 500,000 rows | 0.664s | 0.110s | 6.0x |
| `rolling_rank(window=20)`, 500,000 rows | 0.462s | 0.217s | 2.1x |
| `rolling_percentile(window=20)`, 500,000 rows | 0.510s | 0.216s | 2.4x |
| traced five-output rolling graph, 500,000 rows | 0.631s | 0.336s | 1.9x |

Multi-factor OLS and ridge now use bounded rolling Gram statistics. Lasso and
elastic-net run coordinate descent on those statistics while retaining the
existing coefficient update and convergence order. Ill-conditioned OLS and
ridge windows fall back to the exact NumPy reference implementations.

Rolling rank and percentile share a contiguous-array grouping path, rolling
valid counts, and forward last-valid indices. Their comparisons remain exact,
including average tie ranks. At 6.25 million rows and a 252-session window,
rank, percentile, three-factor OLS, and two-factor ridge completed in 4.50,
4.60, 13.06, and 9.62 seconds respectively. These runs did not allocate a
row-count-by-window-size design tensor.

The execution runtime deduplicates eager inputs within one run and collects
shared trace plans once per domain, key lineage, trace identity, and column
set. A rank-plus-percentile graph now uses two materializations instead of
three, while dense multi-output graphs retain one final collection boundary.
