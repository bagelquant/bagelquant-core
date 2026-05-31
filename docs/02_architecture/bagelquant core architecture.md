# BagelQuant Core Architecture

## Overview

BagelQuant separates concrete panel data from lazy graph logic.

```text
Panel inputs
    |
    v
Transformer and Composer functions
    |
    v
Graph logic chains
    |
    v
Internal execution runtime
    |
    v
Cached Panel outputs
```

## Panel

A `Panel` is an immutable numeric frame indexed by time and asset.

```python
price = Panel(price_df, name="price")
```

Panels are DAG leaves and execution outputs.

## Graph

A `Graph` represents lazy derived logic:

```python
bm_ratio = div(book, price, name="bm_ratio")
bm_factor = rank(zscore(bm_ratio), name="bm_factor")
```

Graph responsibilities:

- Collect dependencies
- Validate DAG structure
- Expose reproducible specs
- Delegate execution
- Expose the materialized `output` panel after execution

Graph does not own domain operations or raw input data.

## Transformer Functions

A transformer is unary:

```text
Panel | Graph -> Graph
```

```python
signal = rank(raw_factor, name="signal")
```

Custom transformers use `@transformer`.

## Composer Functions

A composer accepts one or more inputs:

```text
(Panel | Graph, ...) -> Graph
```

```python
bm_ratio = div(book, price, name="bm_ratio")
```

Custom composers use `@composer`.

## Internal Nodes

Calling an operation creates an internal node that stores:

- Parent nodes
- Qualified operation name
- Serializable configuration
- Node name and metadata
- Cached output panel after execution

Users do not construct internal nodes directly.

## Execution

Calling `Graph.compute()` invokes an internal runtime that recursively
evaluates dependencies, aligns multi-input frames, computes deterministic
cache keys, caches output panels during execution, and updates node outputs.

```python
signal.compute()
panel = signal.output
```

Scheduling is sequential in the current implementation. Parallel scheduling,
persisted caches, and incremental invalidation remain future work.
