# Core: typed computation and persistent graphs

Core 0.11 is a standalone Python 3.13 package. It imports no other BagelQuant
package. Callers supply data, Domain coordinates, immutable source evidence,
SQLite and artifact paths, observation dates, information cutoff and resource
limits. Workbench owns machine inspection, global scheduling and China semantics.

Core metadata schema 4 retains one small successful-node row per graph, execution
context and durable node. `CoreStore.node_completions(graph_id, context_id)` reads
the definition/upstream digests, completed observation/check dates and immutable
result reference in one metadata query. Exact repeat plans are already complete:
they invoke no value lookup, source proof callback or numerical kernel. Missing
children load only their direct completed parents as lazy typed plans. Each node
advances its completion after durable result publication, so an interrupted run
resumes its remaining frontier. Global/branch publication still requires complete
coverage and revision fences; historical updates preserve a later completion row.
Updates store their scalar header, admitted contexts (including empty contexts)
and node-result relations once. Indexed request/result lookups reconstruct the
same public receipt and identity without scanning historical receipt JSON.
`CoreStore.materialization_candidates(node_id, implementation_id, ...)` selects
compact policy, parameter and date headers before opening original manifests or
checkpoints. Finite and causal blocks consume headers; checkpoint selection stops
at the newest valid boundary. Canonical 32-session mathematics is unchanged.
`CoreStore.read_context(max_metadata_bytes=..., metadata_observer=...)` shares a
bounded immutable metadata memo across workers and expires on exit. The optional
observer receives `(event_name, amount)` on the calling worker; callers provide a
thread-safe collector. Revision guards always read fresh state; explicit audits
read original metadata and numerical evidence. A separate finite preparation cache
shares actual Domain windows, halo schedules, successful proofs and fitting parent
slabs within an admitted update, then expires after publication.
Use a fresh store for schema 4; older stores, including schema 3, are rejected
without a migration/compatibility reader.

Deterministic zero-history lazy built-ins execute in canonical, bounded regions.
Requested roots, shared operator boundaries, explicit `persist_nodes`, rolling,
stateful, model, causal, custom and eager nodes remain durable. Exact intermediate
interval indexes are saved with each boundary block; they are logical evidence,
not numerical artifacts. Whole-region seals bind coverage to the durable root.
Update receipts expose `results` containing real artifact IDs and `coverage`
mapping each transient logical node to its durable boundary in the same context.
`CoreStore.update_covers(receipt, context, node_ids=...)` checks registered evidence.
Direct internal-output requests promote those nodes without invalidating already
completed roots. Canonical policies, block keys, types, traces and resets stay equal
to execution with every operator in `persist_nodes`.

`Graph.plan_update(..., kernel_dispatch=...)` separates finite/region math from
publication. Core stages explicit Arrow inputs and immutable `FiniteKernelWork`;
the caller can send it to `compute_finite_work` in a fresh process and return a
`FiniteKernelResult`. Workers have no store or completion authority. The coordinator
consumes typed results under current cancellation/revision fences and deletes the
staging files. `native_admission` wraps each complete execution unit; Workbench
chooses native profiles, worker admission and aggregate budgets before imports.

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
projections and unknown histories still have implementation gaps in the stateful
path described below. Finite execution with a completed matching graph/context
requires explicit interval-index maintenance when its reuse proof is unknown.

`Graph.plan_update(input_window_loader=...)` may load only each missing block's
direct input window. Core validates the full frozen identity, type and traces,
and the window's actual calendar/membership coordinates. A completed canonical
partial block continues by publishing its new output tail. Pointwise kernels
consume new sessions only. The built-in `pct_change` and `smooth` kernels read
only the required admitted lookback of assets occurring in the missing output,
including nulls and historical observations needed after reentry. Retained
canonical proof scopes and material keys stay unchanged. Other rolling kernels
reconstruct the canonical warm state to preserve floating-point arithmetic.
`computed_output_sessions`, `kernel_sessions` and
`input_sessions` report these distinct amounts of work.

`CoreStore.publish(value_references=...)` composes registered numerical partitions
without decoding, rehashing or rewriting old values. Indexed ordered calendar,
membership/type/trace and exact once-only coverage proofs are required. Reusable
Domain files are fenced again inside the publication writer transaction. Missing
optional indexes require explicit maintenance; they never trigger an audit or
backfill during publication. When several registered receipts share a Domain,
publication prefers a receipt with an index bound to its original content in
both preparation and the closing transaction. It still validates actual
coordinates; a damaged selected proof fails and never causes implicit repair.
`completed_materializations` reads successful scoped
update receipts, rather than interpreting interrupted cache fragments as a
completed context.

Declared stateful operators restore checkpoints after every parent's complete
prefix is proven unchanged. Missing proof or a missing valid preceding state
for completed history blocks with explicit maintenance; it never starts a cold
historical replay. Candidates validate numerical signatures and declared state
shape, trying earlier proven states when an optional candidate is malformed.
Revisions, maturity and membership changes invalidate their actual prefix proof.
Same-boundary enclosing receipts reuse values/state without input consumption.

Training `checkpoint_parent_scopes` declares each label auxiliary's consumed
observation prefix and the last scheduled fit's information cutoff. A caller's
`input_scope_proof` can prove economic endpoints were ineligible at that cutoff;
calendar extensions do not invalidate old fits for such endpoints. This does
not change the sampling window or maturity parameter. The original scoped
proof/cutoff is retained in the checkpoint policy; legacy proofs keep their
original scope and cannot be narrowed implicitly. Newly available labels
outside that prefix preserve old fits. Positional features retain their complete
prediction prefix, including a source also aliased as a label. The complete output
calendar/membership prefix is checked independently. Auxiliary sources with
fused trace proofs still use the full prefix; separate trace evidence and finer
fit-boundary checkpoints remain required for more precise historical repair.

Finite updates preflight completed blocks before any kernel. If changed opaque
proofs invalidate the entire completed prefix, execution blocks for explicit
impact resolution/proof maintenance. A proof-format change never permits
historical numerical replay or constitutes a numerical definition change.

`OperationContract.checkpoint_history` declares a continuation read halo: EWM
uses new sessions, weight state includes the preceding session, and training
uses its window plus label maturity. The full execution calendar retains fit
anchors and embargo semantics. Cumulative traces retain a per-asset checkpoint
carry; older states derive this carry from saved output traces when consumed.
Continuation publishes only its typed tail and references original partitions.
Relevant `$` parameter bindings enter state/block policy; unrelated parameters
do not invalidate completed work. Unprovable legacy policy/parameter scope needs
explicit maintenance. Reading corrupt historical values or auditing them still
fails; reference-only publication does not audit those bytes.
Physical batches and worker counts do not change
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
trace propagation, history and checkpoint eligibility). Unknown histories cannot
authorize replay of completed results. Optional ML/optimizer dependencies are imported only as needed.

See the [generated Operator catalog](reference/operators/index.md).

Paired temporal operations such as `diff` and `pct_change` require current trace
support before combining current and shifted evidence. A missing current
coordinate cannot inherit a prior-only availability trace. Known null values
keep their current evidence, including the first undefined change. The
`logical_runtime.v2` implementation identity prevents reuse of older trace
materializations without altering logical identities or historical receipts.

## Published integrity and reads

CoreStore owns immutable manifests and committed graph/update receipts. `describe`, `describe_update` and `describe_evidence` check metadata identity without hashing artifact bytes. Publication validates immutable identity and transaction fences; explicit integrity audits verify every retained channel and derived index. Bounded `read_values` selects persisted calendar and dynamic membership rows before Domain validation and reads only monthly partitions intersecting the requested dates, preserving sparse values, types and traces. Canonical Domain footer schemas and invalid null observation keys remain checked; empty selections return typed empty frames. Unbounded reads still validate the complete Domain. `evidence` reads auxiliary evidence without loading the value history.

`read_context` shares bounded, thread-safe checksum proofs within one operation. Proofs bind receipt, expected checksum and file identity; replacement, modification or context exit discards reuse. They are not a persistent global validity label. Inspection and graph metadata reads create no missing store or graph. Explicit initialization enables SQLite WAL; readers use query-only committed transactions, while publication retains fresh revision/CAS checks.

## Explicit input proof format maintenance

`CoreStore.plan_input_proof_rebind(spec, identity, **source_bindings)` validates
one original canonical finite block against frozen source-owner projections and
registers an immutable admission record. Despite its planning name, this is an
explicit metadata mutation, never a normal readiness check or graph update step.
The caller supplies actual original parameter bindings, direct parent Domains,
fully bound interval tokens, and `ConsumedInputEvidence` for both sources. Each
projection binds its immutable source receipt/digest, exact consumed content,
read bounds, Domain, value type and every consumed trace channel. Scalar-only
evidence cannot establish equivalence of a traced input. Output equality alone
does not establish input equivalence. Unknown original bindings require source
owner maintenance rather than a guessed current receipt.

`Node.source_interval_identity(source_token, start, end)` supplies the same Core
Domain/type/trace binding used by Graph for external source proofs. Bounds are
`datetime.date` values; the source owner proves the consumed channels. The
method reads only actual coordinates and metadata, without numerical reads or
publication. Graph memoization distinguishes actual immutable Domain objects,
even when callers supply equal opaque Domain identities.

`apply_input_proof_rebind(plan)` checks the exact registered plan before a cache
hit and in the publication transaction, then references original numerical
partitions. It does not execute historical kernels or rehash original files.
Original receipts, numerical keys/context, parameters and execution policy remain
unchanged; the replacement changes only proven input identities and may record
verified relevant parameter bindings. Reads and explicit audits validate the
entire retained original/admission chain. Missing or altered authority blocks
reuse; no normal path rebuilds it implicitly. Cancellation before admission
leaves no positive authority; cancellation afterward may retain planned metadata
without publishing a numerical result or advancing a graph pointer.


Rebalance checkpoint continuation uses the actual named calendar Node's complete
Domain date axis, including sessions with empty membership. The saved anchor and
complete consumed prefix must match before restoring state; only new sessions
are read and calculated. CoreStore.read_checkpoint reads the state channel alone.
CoreStore.compose_artifact references original uniquely dated decision artifacts
and appends a same-schema, nonoverlapping tail, preserving historical files.
Legacy completions without checkpoint/calendar or dated-artifact metadata require
explicit owner maintenance before continuation.

### External input proof windows

The five supervised `prediction:*` factories use independent-period checkpoints.
They retain original IC vectors or OLS coefficients/covariance and diagnostics,
sample identities, availability, pending periods and the complete Domain date
axis, including empty sessions. `OperationContract.checkpoint_periods` selects
new, revised and newly mature samples from day-level source proofs. Core's
checkpoint executor uses the same Runtime for a statistics phase and the exact
dependent prediction dates. This preserves arbitrary label delays without a
fixed lookback assumption. Historical fits and unchanged value partitions are
reused; committed statistics survive an interrupted prediction/publication.
Legacy results lacking original statistics require explicit evidence maintenance
from saved IC/regression records; output values cannot establish that evidence.
Missing proofs or changes affecting all completed history remain explicit blocks.

`UpdatePlan.input_proof_windows(context_id)` returns a typed Polars metadata table
with `context_id`, `input_key`, `consumer`, Date `start`/`end`, optional Date
`consumption_cutoff`, `strategy` and `status`. A null consumption cutoff means
no override of the caller's actual frozen global information cutoff. It uses the plan's actual bound
Domains and supported finite policies. Finite scopes use the execution engine's
32-session blocks and membership-aware observation halos, including retained
same-policy partial block endpoints. Named auxiliary external parents participate.
Checkpoint scopes use the same configuration/role/trace rules and relevant
parameter candidate selection as execution. Retained original `parent_ends` and
`parent_cutoffs` are preserved; absent legacy fields mean the original full prefix,
not a newly shortened consumed-label contract.

`required` describes current proof requests; `candidate` describes conditional
retained-prefix prerequisites, whose proof and checkpoint validity are still
checked during execution. `empty` consumed prefixes need no scalar token.
`unknown` identifies a generic or unsupported incremental strategy, with null
bounds; it never promises a finite replay window. Exact materialization hits
produce no input proof requests. The table describes direct external parents
only; completed operator parents are proven by Core's own receipts. Independent
period checkpoints return `strategy="checkpoint_periods"` and one required
Date interval per source period, including empty days, with no cutoff override.

This API reads metadata and actual coordinate state only. It does not invoke proof
callbacks, load source values, run kernels, audit original bytes, merge graphs,
publish records or warm indexes. Callers may explicitly prepare exactly these
scopes through the source owner's maintenance API, preserving real information
cutoffs and coordinate selectors. Candidate scopes are a conservative union,
not a claim that every candidate is read or reused. Inventory can grow through
other publications; new missing scope proofs remain an explicit execution block.
Monthly hashes cannot replace a different block/halo/Domain scope or be joined
into a new canonical hash. Scope planning is not live acceptance or authority to
perform maintenance.
