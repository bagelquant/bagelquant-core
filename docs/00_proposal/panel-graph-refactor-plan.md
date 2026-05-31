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
from bagelquant_core import ExecutionEngine, Graph, Panel
from bagelquant_core.composer import composer
from bagelquant_core.transformer import transformer
```

- `Panel` is the explicit data input and output object.
- `Graph` stores lazy logic chains.
- Decorated transformer and composer functions build graphs.
- `ExecutionEngine` evaluates graphs and caches output panels.

## Completed Refactor

### Panel Data Boundary

- Users create raw inputs with `Panel(dataframe, name=...)`.
- Execution results are `Panel` objects.
- Panels validate numeric two-dimensional data and are treated as immutable.

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

- The engine caches computed panels.
- Computing a downstream graph populates outputs for evaluated intermediate
  graphs.
- A shared engine reuses cached panels across related runs.

## Future Work

- Add stable serialization and versioning for operation identifiers.
- Add persisted caches and explicit invalidation policies.
- Add operation modules for neutralization, portfolio construction,
  backtesting, and analytics.
