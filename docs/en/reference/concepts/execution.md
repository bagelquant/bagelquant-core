# Execution Model

## Overview

Graphs define what should be computed. Calling `Graph.compute()` builds one
sparse Polars plan and materializes public outputs only.

```python
signal.compute()
panel = signal.output
```

## Pipeline

```text
Graph construction
    -> validation
    -> dependency resolution
    -> operation-contract planning
    -> lazy fusion / required dense alignment
    -> optional eager barriers
    -> Panel output creation
    -> cache storage or reuse
    -> Graph.output population
```

## Reusable Compilation

Declarative graphs can be validated once and rebound repeatedly:

```python
compiled = Graph.compile(specification)
runtime = ExecutionRuntime()
january = compiled.compute(january_inputs, runtime=runtime)
february = compiled.compute(february_inputs, runtime=runtime)
```

## Current Semantics

- Execution is deterministic.
- Panels are immutable from the public API.
- Multi-input frames align on intersecting `(time, asset_id)` keys by default.
- Intermediate values are internal `PlanValue` lazy plans, not Panels.
- Shared DAG nodes are evaluated once per runtime invocation.
- Full input payloads are never hashed by the runtime. Explicit identities are
  suitable only when the caller can guarantee immutable input content.
- `materializations` and `eager_barriers` counters support structural tests.
- Independent eager kernels use a bounded ready frontier when
  `ResourceLimits.parallel_nodes > 1`; lazy plans remain fused.

## Persistent Materialization

An application implements `MaterializationStore.query(key)` and
`MaterializationStore.publish(value)` and passes it to `ExecutionRuntime`.
The query returns `MaterializationLookup` with HIT, PARTIAL or MISS. An exact
HIT restores the saved typed Panel and its traces, fit audits, operator state and
named numerical artifacts. PARTIAL is a candidate only: it is recomputed unless
the application has already proved a causal prefix and supplied its checkpoint.

`MaterializationKey` separates logical identity from implementation identity,
relevant input identities, Domain and execution context. Unrelated snapshot
changes need not change any of those relevant receipts. `Panel.from_domain`
accepts a permanent `source_key`; its `identity` and `trace_identity` must prove
immutable content. Anonymous instance identities cannot support reconstruction.

The existing sparse/eager planner remains the numerical executor. Pending node
outputs are collected together within the active resource budget, then reloaded
as immutable Panels from the adapter. A scan-backed adapter bounds resident
values without rerunning shared kernels. Large applications must also submit
bounded date blocks; a resource limit never changes training history or cadence.

`runtime.node_materializations` exposes saved records by logical node ID;
`runtime.node_artifacts` exposes named numerical outputs. `rebalance` retains
both sparse complete targets and every hold/unavailable decision. Checkpoints
are keyed by logical node ID, and restored state and the explicit execution
calendar participate in materialization identity. Callers own source proofs,
historical invalidation, atomic publication and retention. Core performs no I/O.

One Runtime instance has a single owner and rejects nested runs. Its frontier
scheduler collects shared eager inputs once, executes independent numerical
kernels in threads, and merges evidence and publishes receipts in stable graph
order on the owner thread. Worker checkpoint and audit contexts are isolated.
Each admitted worker receives `ResourceLimits.for_workers(actual_workers)`;
LightGBM uses that worker's thread and histogram shares. Applications configure
the shared Polars pool and native BLAS/OpenMP pool before importing their
numerical libraries; native pools must respect the maximum concurrent kernels.

Current resident-memory samples drive later admission and optional-cache
release. Numerical rank comparison temporaries also respect `batch_rows`.
Regression moment tiles keep their fixed arithmetic boundaries: changing their
cumulative-sum origin would change floating-point results under the same key.
`runtime.resource_usage` records observed resident memory, pressure reductions
and actual kernel concurrency. The memory target remains soft: one required
kernel and its inputs may exceed its estimate, and callers must bound its date
coverage rather than truncate required numerical history.

`runtime.plan_materialization_keys(graph)` plans operator keys without executing
operators or collecting input payloads; `runtime.node_domains` retains the
corresponding Domains. Planning uses the same full-calendar checkpoint context
as durable execution. This allows an application to query complete immutable
results before loading expensive auxiliary inputs.

`causal_history_requirements(logical_spec)` returns preceding observation counts
for supported finite closures and `None` for unknown or unbounded history.
Counts accumulate through primary and auxiliary dependencies. They are counts
of an asset's admitted coordinates, not calendar days: sparse dynamic membership
requires a membership-based halo proof. The helper does not establish source
prefix integrity or authorize stateful checkpoint restoration. Applications
retain those proofs, trim block halos and publish only complete full-key values.
New receipts use `materialization_trace_identity(key, trace_columns)`, a canonical
trace token based on the complete key and columns. Block halos and author aliases
therefore do not affect saved trace identity.

`ExecutionRuntime(node_contexts={logical_node_id: context_version})` assigns an
explicit numerical execution policy to selected nodes, with `context_identity`
as the default. Planning and execution use the same mapping, and changed parent
keys propagate into every consumer. An application may therefore combine a fixed
finite-block policy with full-history downstream kernels; hardware budgets never
belong in these context versions.
