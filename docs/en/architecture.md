# Core: typed computation and persistent graphs

Core 0.11 is a standalone Python 3.13 package. It imports no other BagelQuant
package. Callers supply data, Domain coordinates, immutable source evidence,
SQLite and artifact paths, observation dates, information cutoff and resource
limits. Workbench owns machine inspection, global scheduling and China semantics.

## Objects and types

- `Domain(calendar=..., universe=...)` defines trading sessions and static or
  dynamic membership. `window(start, end)` selects observation coordinates.
- `Node.from_domain(frame, domain, source_key="close", identity="immutable-v1")`
  accepts a Polars DataFrame or LazyFrame in sparse `(time, asset_id, value)` form.
  Values outside membership are removed. `value_type` is `numeric`, `category`,
  `prediction` or `weights`; availability columns are explicit traces.
- `Operator` calls accept Nodes, named auxiliary Nodes and separate scalar
  configuration, returning deferred Nodes. All operators live in
  `bagelquant_core.operator` under one registry and mathematical modules.
  `node.compute()` materializes a dependency closure; collecting a deferred
  node before computation raises an error.
- `Graph` supports Python construction, safe AST DSL, local/global merging,
  dependency resolution and node lifecycle. It never evaluates arbitrary Python.

Use explicit permanent `source_key` values for a graph reused across sessions.
Display names are not logical source identities. Eager frames can derive immutable
content evidence; lazy frames need caller-provided immutable identities for durable
execution. A data receipt identity must change when values or provenance change.

Peer prediction inputs cannot mix with ordinary numeric Nodes. Operators preserve
prediction type when their contract allows it. `prediction_signal(x)` explicitly
produces predictions; model operators produce predictions and portfolio operators
produce weights. Group/category, risk, label, availability, calendar and reference
roles are validated individually and retained in the full dependency graph.
Python, bound DSL, saved graphs and cache reads share the same validation paths.
Unbound authoring templates defer source types until actual Nodes are bound.

## Standalone persistent example

```python
from datetime import date
import polars as pl
from bagelquant_core import CoreStore, Domain, Graph, Node

store = CoreStore("scratch/core.sqlite", "scratch/artifacts")
days = [date(2024, 1, 2), date(2024, 1, 3)]
domain = Domain(calendar=days, universe=["A", "B"])
x = Node.from_domain(domain.grid_lazy().with_columns(pl.lit(1.).alias("value")),
    domain, source_key="close", identity="fixture-close-v1")
local = Graph.from_dsl("output = zscore(rolling_mean(close, window=2))", inputs={"close": x})
global_graph = Graph(store=store, graph_id="research")
merged = global_graph.merge(local)
global_graph.register_context("universe", anchor=days[0], information_cutoff=days[-1])
plan = global_graph.plan_update({"universe": {"close": x}}, through=days[-1])
while not plan.complete:
    for node_id in plan.ready("universe"):
        plan.execute("universe", node_id)  # The caller chooses admission and budgets.
receipt = plan.publish()
root = next(iter(merged.outputs.values()))
values = store.read_values(receipt["results"]["universe"][root])
```

Reopen `CoreStore` at the same two paths and `Graph` with the same graph ID to reuse
saved definitions, states and results. No default storage discovery or service is
installed. The new schema rejects incompatible databases; no migration is provided.

`CoreStore.inspect()` reports schema readiness without initializing, recovering
or changing original storage/sidecars; its offline metadata snapshot rejects
active/hot rollback journals. Live applications may explicitly use
`inspect(runtime=True)` to read one committed SQLite transaction with ordinary
WAL/SHM coordination during concurrent writes, without copying metadata.
`inspection.open_metadata_snapshot(path, runtime=True)` exposes the same generic
read-view primitive to downstream backends. Runtime views respect committed WAL,
permit coordination sidecars, forbid writes and fail closed when an abandoned hot
journal would require recovery. Readiness is separate from artifact integrity.

## Graph and identity

`Graph.from_dsl` builds a local graph. `add_dsl` validates and atomically merges
into a global graph, returning outputs and local-to-global mappings without
computing. `merge`, `resolve_local`, `bind`, `upstream` and `downstream` expose
closures, including auxiliary roles; resolving does not copy result data.

Logical identity includes operator, normalized defaults, ordered dependencies and
types. Names, app IDs, Domain, input versions and workers are excluded. Material
identity adds Domain, source/trace evidence, implementation version and numerical
context. Pure identity is eliminated and verified binary add/multiply operands
are canonicalized in the actual execution definition. No arbitrary reassociation,
distribution or division cancellation is performed; custom operators do not rewrite.

## Cache, incremental work and evidence

CoreStore owns immutable Parquet generations, SQLite metadata, monthly values,
Domain, types, traces, artifacts, training audits and checkpoints. `query` reports
exact hit, partial coverage or miss. Exact hits execute no kernel and require no
numerical source reads. Partial coverage alone never proves reusable results.
`describe`, `inventory`, `read_values`, `evidence`, `check_integrity`, `cleanup_plan`
and `apply_cleanup` are the public application boundary. Cleanup reclaims verified
unreferenced completed generations; committed historical receipts remain retained.

Finite-history operators use fixed 32-session mathematical blocks with a supplied
anchor. Dynamic membership halos count admitted observations. Block proofs include
all parent values, traces, coordinates and membership. Append and revisions reuse
only proven blocks; cross-sectional changes affect complete dates. Unsupported
projections and unknown histories conservatively use required full history.

Declared stateful operators can restore checkpoints after all parent prefixes
are proven unchanged. Revisions, maturity and membership changes invalidate that
proof. Full observation calendars remain available to training windows and trace
propagation; checkpoints skip completed state transitions. Damaged optional
checkpoint candidates fall back to computation; explicitly reading corrupt
historical evidence still fails. Physical batches and worker counts do not change
mathematical policy, identity or results. Observation endpoints never replace the
frozen information cutoff.

## Versions and sleep

A global plan freezes graph revision, state revision and every registered context,
then computes every active derived node. All must succeed before `publish` moves
the global pointer. A branch plan takes `roots=` and publishes a branch receipt
without advancing the global version. Failed/canceled/conflicting plans retain
completed immutable cache entries but cannot publish a new global version.

`sleep(node)` atomically sleeps the complete downstream closure across all contexts.
New branch growth, admission, execution and publication check the state; an exact
cache hit cannot bypass sleep. `wake(nodes)` changes only explicitly selected nodes
and atomically requires every upstream node to be active. Historical evidence
remains readable. `reference_publication(receipt_id)` fences state writers while
an application commits its own receipt reference; Core's immutable commit remains
independent. `find_update(graph_id, request_id)` supports interrupted binding recovery.

## Validation and numerical extensions

Run `uv run ruff check .`, `uv run python -m pytest` and
`uv run python scripts/generate_operator_reference.py --check` from Core. Tests use
synthetic inputs and temporary stores. Register custom operators with explicit
input/output/auxiliary types, version and OperationContract (execution, density,
trace propagation, history and checkpoint eligibility). Unknown histories require
full computation. Optional ML/optimizer dependencies are imported only as needed.

See the [generated Operator catalog](reference/operators/index.md).

Paired temporal operations such as `diff` and `pct_change` require current trace
support before combining current and shifted evidence. A missing current
coordinate cannot inherit a prior-only availability trace. Known null values
keep their current evidence, including the first undefined change. The
`logical_runtime.v2` implementation identity prevents reuse of older trace
materializations without altering logical identities or historical receipts.

## Published integrity and reads

CoreStore owns immutable manifests and committed graph/update receipts. `describe`, `describe_update` and `describe_evidence` check metadata identity without hashing artifact bytes. Publication and explicit integrity audits still verify every retained channel. `read_values` validates the Domain and only monthly partitions intersecting the requested dates, preserving sparse values, types and traces. `evidence` reads auxiliary evidence without loading the value history.

`read_context` shares bounded, thread-safe checksum proofs within one operation. Proofs bind receipt, expected checksum and file identity; replacement, modification or context exit discards reuse. They are not a persistent global validity label. Inspection and graph metadata reads create no missing store or graph. Explicit initialization enables SQLite WAL; readers use query-only committed transactions, while publication retains fresh revision/CAS checks.
