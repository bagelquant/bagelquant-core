# Execution Model

## Overview

Graphs define what should be computed. `ExecutionEngine` materializes output
panels and caches them.

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

## Cache Reuse

Pass an explicit engine when related runs should share cached panels:

```python
engine = ExecutionEngine()
signal.compute(engine)
another_signal.compute(engine)
```

## Intermediate Outputs

Executing a downstream graph evaluates its dependencies. Every evaluated
derived node receives a cached panel output:

```python
signal.compute(engine)
prediction_panel = prediction.output
signal_panel = signal.output
```

## Current Semantics

- Execution is deterministic.
- Panels are treated as immutable.
- Multi-input frames align on intersecting indexes and columns by default.
- Cache values are panels.
- Scheduling is sequential.

Parallel scheduling, persisted caches, and explicit invalidation remain future
extensions.
