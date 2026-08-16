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
Sparse PlanValue / Polars LazyFrame
    |
    v
Cached Panel outputs
```

## Panel

A `Panel` is an immutable numeric plan indexed by time and asset. Every input
is normalized through a `Domain`, which owns its trading sessions and compact
asset membership. Inputs remain sparse; `Panel.collect(dense=True)` is the explicit dense
dense, defensive-copy boundary.

```python
price = Panel.from_domain(price_df, domain, name="price")
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

Calling `Graph.compute()` compiles the DAG into lazy `PlanValue` objects.
Lazy nodes fuse into one Polars plan. Dense alignment is inserted
only for operations whose contracts require it, and NumPy/regression
operations create explicit eager barriers. Shared nodes execute once and
multi-output graphs use one final collection boundary.

```python
signal.compute()
panel = signal.output
```

`Graph.compile(spec)` validates a declarative graph once. Its
`CompiledGraph.compute(inputs, runtime=...)` method can be rebound to successive
input batches. Cache keys use input and Domain identities plus node
configuration; Core does not hash full payloads during graph execution.
