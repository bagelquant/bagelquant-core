# Concepts

`bagelquant-core` models research logic as domain-aware data plus lazy
operation graphs.

## Domain

A `Domain` defines the sessions and asset universe for a research problem. It
does not download calendars or security masters; callers provide the calendar
and either a static asset list or a dynamic membership frame.

## Panel

A `Panel` is immutable time-by-asset numeric data aligned to a `Domain`.
Panels are raw inputs, graph leaves, materialized graph outputs, and cache
values during execution.

See [Panel](reference/concepts/panel.md).

## Graph

A `Graph` is a lazy chain of research logic. Transformers and composers return
graphs, and execution materializes output panels only when `compute()` is
called.

See [Graph](reference/concepts/graph.md).

## Transformers

Transformers are unary operations:

```text
Panel | Graph -> Graph
```

They cover ranking, normalization, rolling windows, missing-value handling,
outlier treatment, and other common factor transformations.

See [Transformer](reference/concepts/transformer.md).

## Composers

Composers combine multiple `Panel` or `Graph` inputs into one graph. They cover
arithmetic, cross-sectional grouping, rolling relationships, projection, and
weighted aggregation.

See [Composer](reference/concepts/composer.md).

## Execution

Execution validates the graph, computes dependencies in topological order,
caches intermediate outputs, and applies domain membership rules so inactive
cells do not leak into later operations.

See [Execution](reference/concepts/execution.md).
