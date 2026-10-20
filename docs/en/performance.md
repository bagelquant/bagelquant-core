# Performance Notes

An update's existing `ExecutionPreparation` budget also admits immutable store
metadata: decoded computation indexes, serialized manifests and successful
original source bindings. The adapter shares the numerical preparation LRU;
it adds no separate allowance. Keys bind both store roots and exact identities,
entries carry conservative encoded-size charges, and metadata hits are reported
in `preparation_usage`. Public manifest calls return fresh trees. Missing proofs
are never cached; eviction and scope exit require an ordinary reread.

An explicit finite store read context remains authoritative when present.
Explicit original-byte audits and closing publication checks still validate
fresh authority; mutable revision/state guards are not memoized. This reduces
repeated preparation without changing canonical 32-session certificates,
numerical identities, historical receipts or accepted update intervals.

The preparation reserve scales to one quarter of the caller's `cache_mib`,
without a fixed 64 MiB cap. It is deducted from the same runtime cache allowance;
it adds no memory budget. For example, an admitted 896 MiB cache reserves
224 MiB for preparation and leaves 672 MiB for runtime caching. A smaller
worker admission restricts the shared preparation reserve for the rest of the
operation. Zero-cache execution retains its ordinary bounded reads. Cache
hits/evictions and decoded parent rows measure useful reuse; a larger reserve
does not change canonical output work, native pool size or mathematical policy.

Built-in `lag` uses the exact numerical-window path. Its runtime shifts keys on
the complete parent calendar, including dates with no member observations, so
numerical reads need only the missing output interval and `periods` preceding
calendar sessions. Sparse observations and reentry do not require older rows
for this kernel. It has no floating accumulator or membership-index reservation.
Canonical proof halos, 32-session publication boundaries, input availability
traces and materialization identities remain unchanged. Nested lag dependencies
still execute through the shared global DAG. This does not change conservative
historical repair decisions or enable generic rolling/stateful partitioning.

Cached stored-parent slabs begin at the original 32-session calendar boundary,
even when a numerical window begins inside that slab. This lets overlapping
consumers reuse one retained slab instead of decoding separate partial prefixes.
Admission includes the extra historical coordinates; nonadmitted and external
input reads keep the exact requested range. The final slab never extends beyond
the requested end, and earlier retained rows are removed before numerical
execution. Output windows, canonical proofs and typed availability remain unchanged.

Preparation diagnostics distinguish per-kind builds, hits, coalesced waits,
evictions and evicted bytes. Oversize build counts/bytes describe entries that
were built but could not be retained. `parent_window_bulk_bypasses` counts the
separate early decision to read a window in bulk when its estimated size exceeds
half the preparation reserve; `parent_window_bulk_estimated_bytes` sums those
estimates, not measured resident memory or physical I/O. Builds include first use
and rebuilds, so aggregate counts alone cannot identify duplicate decoding.

Finite percentage-change blocks can use an optional private dynamic membership
view after their unchanged bounded input read and complete typed dense
normalization. The view retains every missing output coordinate and each output
asset's last H admitted preceding coordinates, including implicit/null keys.
Its calendar is unchanged; its active asset union can contract. Dynamic dense
alignment uses only those active keys, and the same registered lazy percentage
kernel and current/shift availability rule run on the smaller table. Original
canonical halos, proof/read windows, output Domains, receipts and historical
references remain authoritative. Static inputs, non-Float64/type/trace channels,
custom/auxiliary kernels and smooth/other rolling computations keep their
existing paths. An empty whole block skips the view; an empty missing tail after
partial-prefix reuse keeps the original scoped input and runtime.

The view reserves working headroom from the already divided worker limit after
the compact coordinate index's cache debit, before any view-only dense/key/Domain
allocation. Its conservative byte estimate is
`6*N*(64+max_asset_UTF8_bytes+8*Date_trace_count)+8*A+256*S+65536`, where N is the
full selected Domain's active coordinates, A its asset count and S its full
calendar length. The S term includes known public calendar constructor Python
date-list scratch. Round upward to MiB, admit at most one quarter of current
noncache headroom, and lower the inner memory target by that reservation. The
original reader's limits and cache allowance stay unchanged; no second worker
division or new cache is introduced. Nonadmission keeps the original pipeline,
without a MemoryError retry. The reservation lasts through ordinary publication;
own aliases are cleared before scope restoration. Foreign constructor/library/
caller exception frames can retain arguments and are not cleared or rewritten.
This is a soft native-allocation selection estimate, not a hard RSS promise.

Existing `input_sessions` and `kernel_sessions` count selected calendar lengths
per input/block, not active rows or native instructions. They remain unchanged
when the private view keeps the calendar. External test/probe spies can report
actual prepared input rows without introducing a new public work metric.
Exact bit/trace/receipt/history, tight-budget and whole-node cost validation is
required before selecting this implementation; fewer rows alone does not prove
an application latency gain. In particular, smooth's weighted rolling sums have
not been admitted to this prefix-trimming argument.

Benchmarks run each case in an isolated child process so setup state, cached
graphs, and peak resident memory are attributable to one operation. Synthetic
inputs use a fixed seed and include nulls and repeated values.
Rolling cases use `min_periods` equal to 80% of the selected window so the
null-bearing input still produces representative output.

Persisted parents keep already-canonical `Date` columns unchanged during input
normalization. A redundant `Date` cast prevents Polars from pushing a finite
window predicate into a Parquet scan, causing repeated historical decompression
for 32-session blocks. The identity expression permits row-group pruning while
preserving actual String/Datetime conversion, membership, values, traces and
materialization identities. File metadata inspection and required proof work
remain; this optimization does not replace full integrity verification.

`CoreStore.read_values(start=..., end=...)` also pushes observation bounds into
the persisted calendar and dynamic membership scans before constructing and
validating the selected Domain. A small value window therefore avoids sorting
and checking every historical membership coordinate. Canonical footer types and
selected key uniqueness remain validated; null observation keys are retained
so pruning cannot hide an invalid date. Static asset vectors remain complete.
An empty or calendar-gap view returns typed empty values and traces without a
synthetic calendar. Unbounded reads and explicit full integrity audits retain
the complete Domain and original-byte checks. Receipts and numerical identities
are unchanged; this is a bounded read improvement, not a new cache.

Ordinary finite updates use that public bounded value reader when a completed
CoreStore parent matches the frozen planned receipt and expected Node. This
avoids rebuilding a scan of every retained physical reference for each missing
block. The existing numerical read start, canonical observation halos, values,
types, traces and local proof rebinding remain authoritative. Recently published
finite parents are eligible; unsaved or differently identified in-memory Nodes
and custom stores keep their existing path. Checkpoint, period Prediction,
causal replay and generic execution retain their loading rules.

Bounded parent consumption validates selected saved Domain/schema/keys and typed
values rather than treating a deferred plan as a full validation boundary. It
can reject malformed input earlier or report the public reader's wrapped error.
No implicit byte audit, full-history replay, numerical-window cache or separate
memory reservation is added. Selected Domain/normalization temporaries still
share the caller's admitted worker budget; a bounded visible frame size is not
a hard bound on backing Arrow allocations or process RSS. Exact valid results
and retained receipts require integration tests; a direct read microbenchmark
does not establish whole-update or application latency gains.

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
output ordering, and public operator behavior. Benefits grow with graph width;
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
names and metadata from physical identity while keeping public Node
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
positionally aligned lazy value columns for exact same-key operators, and
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
and rolling-pair operators use lazy horizontal value plans instead of key hash
joins. Sparse or unproven inputs retain the balanced join path. At 6.25
million rows, `sum_10` improved from 2.73 seconds and 5.69 GiB peak RSS to
0.045 seconds and 1.52 GiB. Rolling correlation improved from 1.74 to 0.85
seconds and its physical sort count fell from two to zero.

Deterministic eager physical results are shared for the duration of one
runtime execution. Five equivalent OLS, lasso, or orthogonalize outputs retain
five logical eager barriers and distinct public identities, names, metadata,
and traces, while the numeric kernel runs once. Diagnostics report positional
operator hits, eager CSE hits, solver batches, and active-window iterations.

The 6.25-million-row, 252-session regularized paths completed in 5.57 seconds
for lasso and 5.98 seconds for elastic-net on the static domain. The dynamic
domain completed in 5.02 and 5.91 seconds respectively. Pointwise and
cross-sectional 20-layer chains contain no runtime-generated sort nodes.


## Stage 3 resource ownership

Core accepts explicit ResourceLimits and executes admitted nodes. Workbench owns
machine detection, DAG admission and memory-pressure response. Historical
measurements above are not a performance claim for the refactored storage path.

## Bounded persistent execution

An admitted update shares immutable Domain windows, halo schedules, successful
interval proofs and fitting parent slabs. All requested halo histories on one
actual Domain share one membership sort. Parent slabs retain each canonical
32-session kernel reset; generic rolling means, sums and standard deviations
keep their original warm prefix because shortening it changes floating-point
arithmetic. A prefix larger than the preparation budget keeps its bulk read path
rather than thrashing a smaller slab cache. Per-key single-flight collection
allows unrelated ready work to overlap. `plan.preparation_usage` reports actual
decodes, decoded sessions/rows, hits, evictions and charged resident/peak bytes.
The preparation budget reserves a quarter of cache memory without a fixed cap,
is restricted by smaller worker admissions, and expires after successful publication.

Schema 4 pointwise regions publish requested/shared/explicit boundaries and retain
exact transient interval indexes. Required transient output plans are collected
together through the existing multi-output collector. Sparse values join traces
by their full coordinate keys before collection, then retain their validated
operation-owned frames for exact proof reads; dense caches never supply sparse
row presence. Per-day proof hashing shares one physical row-hash evaluation and
partitions the hashes in original row order, preserving every original schema,
height, width and SHA byte. Region certificates are committed with
the durable block, and seals establish complete logical coverage without claiming
that transient numerical payloads exist. Physical persistence hints change neither
canonical block keys nor mathematical execution policies. In a synthetic five-node,
70-session region, this reduces durable receipts from 20 to 4 with exact keys,
values, types and traces. It is a structural count, not an elapsed-time claim.
Auxiliary evidence traverses the seal's original transient keys to retained
durable ancestors, preserving model audits, operator checkpoints and artifacts
in root evidence descriptions and selected reads.
Before reading a missing interval, finite/region execution estimates all scoped
input frames and every retained output frame, including key/trace/object overhead,
then reserves eight copies for computation, validation, transport and proof work
against the actual admitted node memory. Oversized work raises `MemoryError`
before source-window reads or kernels; it cannot bypass admission by running
inline. Existing receipts remain usable under smaller budgets, and append charges
only the missing tail. Increase the admitted memory or choose explicit
`persist_nodes` boundaries before initializing a fresh graph; admission never
replays completed history to insert boundaries. This remains a conservative
working-set estimate, not an operating-system memory quota.

Finite work descriptors permit fresh process dispatch without passing mutable
plans, callbacks or full frames through Python pickling. Core stages Arrow files,
validates returned types/traces, and retains all publication/completion authority
in the coordinator. Transport bytes and dispatched kernel counts are reported.
Workers start only for admitted missing work; no-op plans do not stage data or
call a dispatcher. Process/native setup and one aggregate resource budget remain
the caller's responsibility. No real-data speedup has been established by these
synthetic acceptance tests.

## Typed preparation and composed metadata

Source and consumed-value validation carry availability traces alongside their
values through one key normalization, uniqueness check and sort. Unique typed
coordinates establish row correspondence; separating traces and joining them
back repeats work at every finite block. Date conversion, invalid-key rejection,
category types, NaN-to-null normalization, sparse row presence and trace values
retain their original rules.

`Domain.window` selects immutable coordinates already validated by its parent
Domain. It retains sorted unique Date/String keys and computes the same canonical
calendar, asset and membership hashes as public construction, including windows
with no active members. Public Domain construction still validates arbitrary
inputs. An opaque parent identity never substitutes for selected coordinates.

Within `CoreStore.read_context`, successful original-source binding checks share
the existing byte-bounded immutable metadata memo. Reading composed receipts with
shared ancestry checks that ancestry once in the finite context. Entries bind
original manifest content and expire on exit; unsuccessful checks are never
retained. Full audits and the closing publication transaction traverse current
original metadata independently of that memo. These internal optimizations
preserve the current storage schema, numerical receipts, canonical block policies and stored bytes.

The manifest entry retains the verified original serialized JSON string, not a
shared mutable parsed tree. Each public manifest hit parses that immutable string
after the memo get has released its lock, returning a fresh dict/list/scalar tree.
Caller mutation, including nested fields returned by descriptions, cannot affect
later reads. A cold read still parses and validates original metadata, rebinding
and source authority before retaining the exact original string; it does not
reserialize or canonicalize JSON for this cache. The same UTF-8 byte charge,
eviction, metadata events, finite lifetime and generic memo copying policy apply.
Audits and publication guards retain their existing fresh validation paths. This
internal representation change has no artifact, identity, schema or formula effect;
it does not by itself establish a whole-calculation speedup or memory peak.

## Selective artifact verification

Metadata inspection and ordinary selected value reads perform zero artifact
hashes. `read_values` decodes the selected Domain and value partitions and applies
the public typed-value validation; it does not establish a fresh SHA proof.
Unselected partition damage can remain unobserved until that partition is
consumed or an explicit integrity audit checks the complete materialization.
Explicit `verify_identity` and full integrity checks retain byte validation;
successful checksum proofs expire with their finite `read_context`. These
structural contracts are tested; they are not a wall-time guarantee or a change
to numerical kernels.

## Finite membership date queries

Dynamic finite `pct_change` and `smooth` execution can retain a compact coordinate
index for one node call. The existing canonical-halo sort supplies asset-grouped
int32 dates, int64 group offsets and an owned int32 calendar. Canonical proof
halos still consider every asset's preceding observations; numerical read starts
still consider only assets occurring in the missing output. Admitted coordinates
include null-valued observations and preserve sparse reentry. The selected dates,
32-session blocks, source proofs, typed values and traces, numerical kernels and
receipt identities do not change. No numerical membership trimming is performed.

Admission uses the worker allocation already divided by the caller. For `N`
membership rows, `A` assets and `S` calendar sessions, the retained charge is
`4*N + 8*(A+1) + 4*S + 4096` bytes, rounded up to whole MiB and reserved from
the existing numerical cache. The same remaining limits cover Runtime and both
full and bounded input loaders. A separate conservative sort/scratch estimate
must fit one eighth of the original memory target minus cache budget. That
estimate is an admission heuristic, not a bound on opaque native allocations.
The existing CoreStore metadata memo has its separate unchanged scope.

Static, empty, history-zero, nonexact or unadmitted calls retain the original
queries. Allocation, validation and cancellation errors propagate without a
second attempt. Buffers own no values, String frame or Domain membership view.
Core clears its own buffer fields and temporary argument/view aliases before
the worker limits restore, on success and failure. Direct ndarray methods avoid
unnecessary Python wrapper frames around date queries and offset accumulation.
This does not promise that arbitrary foreign library or caller traceback locals
release their own references; Core does not clear those frames or rewrite
propagated exceptions.
Completed whole-receipt reuse bypasses this allocation. Interrupted reassembly
of already-complete blocks can still allocate the index despite zero kernels
and zero window reads. Existing public work-counter keys remain unchanged;
private focused checks account for reserved and retained bytes separately.

## Bounded cold finite invocations

An optional private initialization preparation can share one ordinary numerical
Graph/runtime call across two to eight consecutive canonical blocks. It requires
an exact concrete CoreStore, no materialization history for the current logical
node, exact completed Core direct-parent keys, existing interval indexes and
nonempty original per32 input proofs. External input loaders, missing optional
indexes, retained candidates, append, repair and interrupted retries keep the
individual-block path. No prerequisite, currentness check or metadata-only repeat
resolves a new footer, reads values or constructs this batch.

The closed path is limited to the original built-in Float64 arithmetic, rank,
percentage-change240, constant100, project-domain, canonical14 and fixed smooth
kernels under Polars 1.42.1. Registered operation, plan and contract objects must
match; other supported versions and custom implementations remain on the existing
path. Sparse admitted observations, implicit nulls, exit/reentry, the full
cross-section at each output date and up to two Date traces remain unchanged.
The unchanged smooth expression reduces each weighted ten-observation window in
its original operand order; this admission does not generalize to arbitrary
stateful rolling kernels or assert binary equivalence without acceptance tests.

Selection first uses existing typed metadata only. With worker limits already
divided and membership-index debits already applied, planning charge P includes
twice each parent/output membership size times its key/string allowance,
256 bytes per calendar session, 4096 bytes per registered parent partition,
a 64 KiB allowance and a separate 4 KiB scalar-memo/axis allowance. It must fit
one quarter of the original noncache working headroom M-C. Candidate output
coordinates are counted from the resident sorted active Date column with scalar
lower/upper binary searches, including implicit-null values and active keys on
dates outside the calendar. Row overflow is rejected
before asset-local history queries or candidate Domain construction/hashing.
Original per32 proof Domains remain unchanged and are not skipped.

Within one immutable node invocation, at most eight successful numerical-start
Dates are retained under exact canonical-origin/own-end keys. Candidate attempts
and adjacent origins can reuse these scalar answers. The callback, exceptions,
Nodes, values and frames are never memo entries. A partial final block has its
actual endpoint; first-origin monotonicity is not assumed, because later asset
reentry can require older admitted observations. The bounded LRU and parent-axis
metadata are cleared on normal/error/cancellation exit; Core drops its own new
callback aliases without clearing foreign traceback frames or rewriting errors.

For each candidate of at most eight blocks, R includes P plus an eightfold
allowance for output rows and the greater of each parent's conservative dense
rectangle or intersecting physical partition rows. Global parent asset counts
and maximum UTF8 key lengths bound any selected window; held calendar axes give
the selected session count. Unknown physical bounds still count. These upper
bounds can decline extra sparse batches but cannot reduce the former reserve.
Candidate estimation constructs no temporary Domain; after admission the same
original public Domain construction/identity path runs under R. No coordinate,
proof, schema, value or trace validation is waived. Output membership is capped by batch_rows;
summed parent rows are capped by eight times batch_rows. R must also fit one
quarter of M-C, and the same immutable limits with memory_target_mib=M-R cover
the batch. After the first batch is selected, P remains reserved between batches
until held schemas/metadata close, including an individual final block. If no
first batch is admitted, selection metadata is discarded before the original
body path. Unsupported schema releases P and its held state before that path.
Cache, threads and all other fields remain unchanged. There is no
second worker division or allocation-failure retry.

Private count-only preparation uses at most twice the Date axis length's bit
length in scalar item visits. The normalized resident calendar is sorted and
unique; active membership is sorted by `(time, asset_id)`, so dates may repeat
across assets. The lower bound excludes dates before the start and the upper
bound includes every date equal to the end. Dynamic counts use membership rows,
not calendar membership or a calendar-by-assets rectangle. Static counts use
calendar sessions times existing assets. Empty/reversed key ranges count zero.

This private helper accepts exact normalized `datetime.date` bounds. Public
`Domain.window` keeps its original input and empty-calendar behavior. The cold
planner reuses its existing calendar list once per candidate after proving all
parent calendars equal. Exactly three finite session counters also use the
already-owned calendar list, retaining the empty-calendar error. Required
publication, proof, typed Node and read-window Domains are still constructed
and hashed through the original path. Counts never establish identity or key
equivalence, even when endpoints, counts or caller-supplied identities match.

No new array, retained view, time index or cache is created. A borrowed resident
Series wrapper and scalar lookup alias are released on normal/error exit; no
full-column conversion, expression collection, rechunk or native `search_sorted`
is requested. The scalar Polars API is not an operating-system allocation/RSS
guarantee. Existing P/R/A reservations, coordinate-index cache charges, worker
division, kernel math, work counters and canonical receipts remain unchanged.
This isolated source daughter requires Root validation and measurement; it has
no measured gain or CPU-cost attribution yet.

After the original per32 proofs, the admitted reservation covers an explicit
footer/schema preflight of each plain immutable parent Parquet plan. This is
consumption validation inside authorized initialization, not a file-free
metadata gate. Each parent's schema is resolved at most once in the invocation;
the private resolution count does not measure physical footer I/O. Only the
exact Date/String/Float64/Date-channel schema is admitted. Unsupported schemas
select the original path before value bodies; footer failures and cancellation
propagate. Body consumption still uses the original public selected-value reader,
full typed/coordinate validation, public Node construction and matching held
schema. Ordinary reads establish no fresh full-file SHA audit.

The result is sliced and published using each original 32-session Domain, proof,
key, trace identity, policy and publisher, followed by the original composed
receipt. There is no batch receipt or new persisted schema/API. Actual runtime
calls, input_sessions and kernel_sessions can decrease; computed_blocks and
computed_output_sessions count only canonical receipts actually published.
Their old larger numerical-work totals are never synthesized.

An admitted native call may finish several blocks before the first new receipt
can be published. This changes initialization failure/cancellation granularity,
not the canonical blocks or durable history. Cancellation and revision guards
run around consumption and every publication. An optimistic inventory guard
rejects unexpected own-node candidates before continuing; it is not an atomic
cross-process lock spanning native execution. Every already committed receipt
survives an error, and its presence vetoes batching on retry. Owned batch frames,
slices, Graph/runtime/binding and temporary Domain aliases are dropped before R
restores on success or failure. Foreign library/caller traceback locals remain
untouched and can retain their own references. P/R and row ceilings are
conservative admission heuristics, not hard native scratch, RSS or cancellation
latency bounds. This unselected experiment has no measured speed claim.
