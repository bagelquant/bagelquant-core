# Graph

## Overview

`Graph` represents a lazy chain of research logic.

```python
price = Panel.from_domain(price_df, domain, name="price")
signal = rank(zscore(price), name="signal")
```

The raw input is a `Panel`. The derived signal is a `Graph`.

## Responsibility

Graph manages:

- Logic-chain outputs
- Dependency collection
- DAG validation
- Topological ordering
- Graph specifications
- Runtime delegation
- Materialized output access

Validation rejects cycles, duplicate node names, invalid parent types, and
operation nodes with an invalid number of parents.

Graph does not store raw input frames and does not contain domain-specific
operation methods.

## Logical Definitions

`Graph.spec()` is a local authoring template with symbolic names.
`Graph.logical_spec()` converts it to the immutable `logical_dag.v1` storage
format. `canonicalize_graph(template, input_keys=...)` provides the same boundary
for declarative compilers. Permanent source keys and numerical input types,
registered operations, normalized scalar defaults, ordered primary inputs and
named auxiliary dependencies define a logical node ID. Labels, object metadata,
data receipts, Domain, operator implementation versions and snapshots do not.

```python
first = rolling_mean(price, window=20, name="first").logical_spec()
second = rolling_mean(price, window=20, min_periods=None, name="second").logical_spec()
union = first.union(second)
graph = Graph.from_logical_spec(union, inputs={price.source_key: price})
```

The union has one shared rolling operation and two output aliases. Portable
serialization verifies each node's content address, dependency order and roots.
The authoring template's provenance-only presentation edges are not numerical
dependencies; applications retain that evidence outside the Core input boundary.

Logical node parameters are immutable JSON values. Changing an operation or its
parameters creates a different node. `parameter_bindings` resolves explicit
execution placeholders without rewriting the stored logical definition.

`Graph.from_logical_spec(..., node_bindings={node_id: panel})` may bind
caller-proven intermediate operator values. Execution stops traversing parents
at those nodes, so their original sources need not be loaded. The complete
logical definition remains unchanged; the numerical Panel type is checked.
The application proves source integrity and coverage and must not publish a
block-specific receipt as a complete full-domain value.

## Output

Before execution, output access raises an error:

```python
signal.output
```

After execution, `Graph.output` is a panel:

```python
signal.compute()
signal_panel = signal.output
```

Computing a downstream graph also populates outputs for evaluated intermediate
graphs.

## Multi-Output Graphs

```python
strategy = Graph(outputs=[signal, prediction])
strategy.compute()
outputs = strategy.output
```

For a multi-output graph, `output` is a mapping from output name to panel.
