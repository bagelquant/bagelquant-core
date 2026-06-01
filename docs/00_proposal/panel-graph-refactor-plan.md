# Panel and Graph Architecture Refactor Plan

## Goal

Separate data from logic:

```python
some_input = Panel(dataframe, name="some_input")
some_factor = rank(zscore(some_input), name="some_factor")
some_factor.compute()
factor_panel = some_factor.output
```

## Public Boundary

```python
from bagelquant_core import Graph, Panel
from bagelquant_core.composer import composer
from bagelquant_core.transformer import transformer
```

- `Panel` is the explicit data input and output object.
- `Graph` stores lazy logic chains.
- Decorated transformer and composer functions build graphs.
- `Graph.compute()` evaluates graphs and materializes output panels.

## Completed Refactor

### Panel Data Boundary

- Users create raw inputs with `Panel(dataframe, name=...)`.
- Execution results are `Panel` objects.
- Panels validate numeric two-dimensional data.
- Panels copy input data and return defensive copies through `Panel.data`.

### Function-Style Operations

- Built-ins are plain functions such as `rank(...)`, `zscore(...)`, and
  `div(...)`.
- Users create custom operations with `@transformer` and `@composer`.
- Adding operations does not require editing `Graph`.

### Lazy Graph Logic

- Factors, predictions, and portfolio weights are graphs.
- Graphs collect dependencies, validate DAG structure, and expose specs.
- A graph output is available through `Graph.output` after execution.

### Runtime Cache

- The internal execution runtime caches computed panels during a run.
- Computing a downstream graph populates outputs for evaluated intermediate
  graphs.
- Shared DAG nodes are evaluated once per runtime invocation.
- Already-aligned input panels reuse their stored hashes.

## Future Work

- Add stable serialization and versioning for operation identifiers.
- Add persisted caches and explicit invalidation policies.
- Add operation modules for neutralization, portfolio construction,
  backtesting, and analytics.
