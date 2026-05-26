# Execution Model

## Overview

The Execution Model defines how a BagelQuant Graph is computed.

While the Graph defines **what to compute**, the Execution Model defines **how to compute it**.

Execution transforms a declarative DAG into a deterministic sequence of computations over Panels, Transformers, and Composers.

---

## Core Principle

BagelQuant execution follows one rule:

```text
A Graph defines dependencies, not execution order.
```

The system is responsible for deriving execution order automatically.

---

## Execution Pipeline

A BagelQuant graph is executed through the following stages:

```text
Graph Construction
        ↓
Validation
        ↓
Dependency Resolution
        ↓
Topological Sorting
        ↓
Scheduling
        ↓
Execution
        ↓
Caching
```

---

## Step 1: Graph Construction

Users define a graph using Panels, Transformers, and Composers.

Example:

```text
Price
  ↓
Returns
  ↓
Momentum
```

At this stage, no computation occurs.

Only structure is defined.

---

## Step 2: Validation

The system validates the graph.

### Checks include:

- Cycles (must be DAG)
- Missing inputs
- Type consistency (Panel compatibility)
- Invalid node configurations

Example invalid graph:

```text
A → B → C → A
```

This is rejected.

---

## Step 3: Dependency Resolution

The system determines all upstream dependencies for each node.

Example:

```text
Momentum depends on Returns
Returns depends on Price
```

This produces a dependency graph.

---

## Step 4: Topological Sorting

The dependency graph is converted into an execution order.

Example graph:

```text
Price
  ↓
Returns
  ↓
Momentum
```

Execution order:

```text
1. Price
2. Returns
3. Momentum
```

Nodes are guaranteed to be executed only after all dependencies are satisfied.

---

## Step 5: Scheduling

The runtime assigns execution strategy.

Depending on configuration, execution may be:

- Sequential
- Parallel
- Incremental
- Cached reuse
- Distributed (future extension)

Example:

```text
Factor A ─┐
          ├──> Independent branches execute in parallel
Factor B ─┘
```

---

## Step 6: Execution

Each node is evaluated:

### Transformer execution

```text
Panel → Transformer → Panel
```

Example:

```text
Rank(Price)
```

---

### Composer execution

```text
(P₁, P₂, ..., Pₙ) → Composer → Panel
```

Example:

```text
Divide(BookValue, Price)
```

---

### Panel production

Each node produces a new immutable Panel.

```text
Output Panel = f(Input Panel(s))
```

---

## Step 7: Caching

Once a node is computed, its result may be cached.

Example:

```text
Returns = f(Price)
```

If Returns is already computed and Price has not changed:

```text
Reuse cached Returns
```

This avoids recomputation.

---

## Incremental Computation

One of the key advantages of BagelQuant execution is incremental updates.

### Example graph:

```text
Price
  ↓
Returns
  ↓
Momentum
```

If Price changes:

```text
Price Updated
```

Only affected downstream nodes are recomputed:

```text
Returns → Momentum
```

Unrelated parts of the graph are untouched.

---

## Parallel Execution

Independent branches can be executed in parallel.

Example:

```text
        Factor A
       /        \
      ↓          ↓
  Model 1     Model 2
      ↓          ↓
        Combined Output
```

Model 1 and Model 2 can run simultaneously.

---

## Execution Semantics

### Determinism

Given the same graph and inputs:

```text
Output is always identical
```

---

### Purity

Nodes are pure functions:

```text
output = f(input)
```

No hidden side effects are allowed.

---

### Immutability

Inputs are never modified in-place.

Each step produces a new Panel.

---

## Memory Model

Execution is designed around Panel immutability.

This enables:

- Safe caching
- Reproducibility
- Parallel execution
- Debugging simplicity

---

## Error Handling

Execution errors are isolated to nodes.

Example:

```text
ZScore(Invalid Panel)
```

Only this node fails.

Downstream nodes are not executed.

---

## Debugging Model

Because execution is graph-based, debugging is node-centric.

Users can inspect:

- Input Panel
- Output Panel
- Transformation logic
- Dependencies

Example:

```text
Inspect node: Momentum
```

Trace:

```text
Momentum → Returns → Price
```

---

## Future Extensions

The execution model is designed to support:

### Streaming execution

- Real-time data updates
- Incremental recomputation

---

### Distributed execution

- Multi-machine graph execution
- Partitioned computation

---

### GPU acceleration

- Matrix-based Panel operations
- ML Composer acceleration

---

### Lazy evaluation

- Nodes computed only when required

---

## Execution vs Graph

| Component | Responsibility |
|----------|----------------|
| Graph | Defines structure |
| Execution Model | Computes results |

Graph is declarative.

Execution is procedural.

---

## Design Goals

The execution system is designed to achieve:

### Determinism

Same input → same output

---

### Efficiency

Avoid unnecessary recomputation

---

### Scalability

Support large-scale research workflows

---

### Transparency

Every computation is traceable

---

### Modularity

Execution logic is independent of graph definition

---

## Summary

The Execution Model transforms a BagelQuant Graph into a deterministic computation process.

It ensures that Panels, Transformers, and Composers are evaluated in the correct order, efficiently, and reproducibly.

Through dependency resolution, topological sorting, caching, and incremental computation, BagelQuant enables scalable and modular quantitative research execution.
