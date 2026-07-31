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
uv run python scripts/benchmark_efficiency.py \
  --profile comparison \
  --case rolling_chain_20 \
  --case ewm_chain_20 \
  --case mixed_chain_20 \
  --case semantic_cse_5 \
  --case dense_exact_output \
  --case sparse_alignment
uv run python scripts/benchmark_efficiency.py \
  --profile cn-daily \
  --windows 252 \
  --universe both \
  --case rolling_rank \
  --case eager_rank_pair \
  --case rolling_ols_3f \
  --case rolling_ridge \
  --case orthogonalize_3f \
  --case traced_multi_output_5
uv run python scripts/benchmark_efficiency.py \
  --profile comparison \
  --case rolling_lasso_1f \
  --case rolling_lasso \
  --case rolling_lasso_8f \
  --case rolling_elastic_net_1f \
  --case rolling_elastic_net \
  --case rolling_elastic_net_8f \
  --case eager_cse_lasso_5
uv run python scripts/benchmark_efficiency.py \
  --profile cn-daily \
  --windows 252 \
  --universe both \
  --case sum_10 \
  --case rolling_corr \
  --case rolling_lasso \
  --case pointwise_chain_20
```

Use `--universe dynamic` or `--universe both` to exercise sparse dynamic
membership. The `cn-daily` profile represents 6.25 million requested rows,
5,000 assets, and 20-, 60-, and 252-session windows. `--rows`, `--assets`,
`--repeats`, and `--windows` override profile defaults.

The JSON output records same-machine best, median, and mean wall time; peak RSS;
output row/null/value summaries; runtime materialization and eager-barrier
counts; sort counts for inspectable lazy output plans; and internal counts for
elided sorts, elided alignments, membership applications, and semantic CSE
hits. Time-based thresholds remain outside CI. Structural tests assert that
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

## July 2026 execution-plan and memory pass

The third pass carries private key coverage, membership, and ordering lineage
through the runtime. Proven exact-domain plans skip final alignment, and
per-asset rolling, EWM, fill, difference, and rate kernels reuse existing
monotonic order. Unknown or custom operations retain the conservative
membership, validation, alignment, and sort path. Public operation calls
continue to return `(time, asset_id)` order.

Same-machine 500,000-row medians with a 20-session window were:

| Case | Previous pass | Third pass | Change |
| --- | ---: | ---: | ---: |
| `rolling_rank` | 0.217s | 0.125s | 43% faster |
| `rolling_percentile` | 0.216s | 0.121s | 44% faster |
| rank + percentile sibling graph | approximately 0.401s | 0.122s | 70% faster |
| three-factor `orthogonalize` | 0.142s | 0.120s | 15% faster |
| traced five-output rolling graph | 0.336s | 0.247s | 26% faster |
| ten-layer rolling chain | approximately 0.215s | 0.092s | 57% faster |
| five semantically equal rolling outputs | five independent kernels | 0.029s | four CSE hits |

The ten-layer rolling chain contains no physical sort node, down from eleven
before order lineage was introduced. A 20-layer rolling chain similarly
contains no sort and completed in 0.171 seconds. Semantic CSE excludes output
names and metadata from physical identity while keeping public Panel
identities, names, metadata, GraphSpec, and runtime cache keys distinct.

For eager operations, exact same-key parents now collect one immutable key
layout and value-only sibling columns. Static exact domains use time-major
arrays with strided per-asset views; dynamic and sparse inputs reuse a stable
asset permutation inside one runtime. Rank and percentile share one exact
comparison kernel while retaining two logical eager barriers. Regression
inputs bypass wide DataFrame joins when key lineage proves positional
alignment.

The 6.25-million-row, 5,000-asset, 252-session static profile showed the
following peak-RSS reductions relative to the second pass:

| Case | Previous peak RSS | Third-pass peak RSS | Reduction | Third-pass time |
| --- | ---: | ---: | ---: | ---: |
| `rolling_rank` | 2.34 GiB | 1.48 GiB | 37% | 3.09s |
| three-factor `rolling_ols` | 6.11 GiB | 3.28 GiB | 46% | 8.16s |
| two-factor `rolling_ridge` | 4.34 GiB | 2.02 GiB | 53% | 5.43s |
| traced five-output graph | 8.22 GiB | 3.50 GiB | 57% | 3.58s |

The same dynamic-universe profile also completed: rank in 2.93 seconds,
three-factor OLS in 7.35 seconds, ridge in 4.82 seconds, and traced five-output
execution in 2.78 seconds. Explicit rolling numeric workspaces remain bounded;
none of these cases creates a materialized total-row-count by window-size
array.

## July 2026 compatible-kernel pass

The fourth pass batches regularized regression windows across assets, uses
positionally aligned lazy value columns for exact same-key composers, and
extends physical CSE to deterministic eager operations. Public operation
calls, GraphSpec, trace rules, and finite coordinate-descent iteration
semantics are unchanged.

Same-machine 500,000-row medians with a 20-session window were:

| Case | Third pass | Fourth pass | Change |
| --- | ---: | ---: | ---: |
| two-factor `rolling_lasso` | 6.23s | 0.70s | 8.9x faster |
| two-factor `rolling_elastic_net` | 6.81s | 0.97s | 7.0x faster |
| `sum_10` | 0.135s | 0.006s | 22.5x faster |
| `rolling_corr` | 0.131s | 0.076s | 42% faster |
| 20-layer pointwise chain | 0.013s | 0.008s | 40% faster |

Coordinate descent retains zero initialization and coefficient update order,
but packs independent windows from many assets into bounded solver calls. A
contiguous active workspace is compacted only after a complete coordinate
sweep. Benchmarks cover one, two, and eight factors plus one, 100, and 1,000
iteration limits.

Exact same-key arithmetic, aggregation, weighting, coalesce, mask, scaling,
and rolling-pair composers use lazy horizontal value plans instead of key hash
joins. Sparse or unproven inputs retain the balanced join path. At 6.25
million rows, `sum_10` improved from 2.73 seconds and 5.69 GiB peak RSS to
0.045 seconds and 1.52 GiB. Rolling correlation improved from 1.74 to 0.85
seconds and its physical sort count fell from two to zero.

Deterministic eager physical results are shared for the duration of one
runtime execution. Five equivalent OLS, lasso, or orthogonalize outputs retain
five logical eager barriers and distinct public identities, names, metadata,
and traces, while the numeric kernel runs once. Diagnostics report positional
composer hits, eager CSE hits, solver batches, and active-window iterations.

The 6.25-million-row, 252-session regularized paths completed in 5.57 seconds
for lasso and 5.98 seconds for elastic-net on the static domain. The dynamic
domain completed in 5.02 and 5.91 seconds respectively. Pointwise and
cross-sectional 20-layer chains contain no runtime-generated sort nodes.
