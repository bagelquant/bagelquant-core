# Graph

## Overview

`Graph` represents a lazy chain of research logic.

```python
price = Panel(price_df, name="price")
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

Graph does not store raw input frames and does not contain domain-specific
operation methods.

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
