# Execution Model

## Overview

Graphs define what should be computed. Calling `Graph.compute()` materializes
output panels and caches intermediate results during execution.

```python
signal.compute()
panel = signal.output
```

## Pipeline

```text
Graph construction
    -> validation
    -> dependency resolution
    -> panel alignment
    -> node evaluation
    -> Panel output creation
    -> cache storage or reuse
    -> Graph.output population
```

## Intermediate Outputs

Executing a downstream graph evaluates its dependencies. Every evaluated
derived node receives a panel output:

```python
signal.compute()
prediction_panel = prediction.output
signal_panel = signal.output
```

## Current Semantics

- Execution is deterministic.
- Panels are immutable from the public API.
- Multi-input frames align on intersecting indexes and columns by default.
- Intermediate cache values are panels.
- Shared DAG nodes are evaluated once per runtime invocation.
- Stored panel hashes are reused when alignment does not change an input frame.
- Scheduling is sequential.

Parallel scheduling, persisted caches, and explicit invalidation remain future
extensions.
