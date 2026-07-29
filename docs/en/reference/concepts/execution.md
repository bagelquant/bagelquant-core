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

Parallel scheduling, persisted caches, and explicit invalidation remain future
extensions.
