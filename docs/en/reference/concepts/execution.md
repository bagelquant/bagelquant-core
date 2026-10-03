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
- Scheduling is sequential.

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

Parallel scheduling remains an application responsibility; one Runtime instance
has a single owner and rejects nested runs.
