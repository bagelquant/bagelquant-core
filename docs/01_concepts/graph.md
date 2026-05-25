# Graph

## Overview

A Graph is the core computational structure of BagelQuant.

Every quantitative workflow in BagelQuant is represented as a Directed Acyclic Graph (DAG) composed of three node types:

- Panel
- Operator
- Composer

The Graph defines:

- What should be computed
- How computations depend on each other
- The order in which computations are executed

Rather than organizing research as scripts, notebooks, or pipelines, BagelQuant organizes research as a graph of reusable computational components.

---

## Philosophy

The central idea of BagelQuant is:

```text
Everything is a Panel.
Everything is a transformation on Panels.
Everything is a DAG.
```

A factor is a graph.

A prediction model is a graph.

A portfolio construction process is a graph.

An entire investment strategy is a graph.

---

## Why Graphs?

Quantitative research naturally consists of dependencies.

For example:

```text
Book Value
Price
     ↓
   Divide
     ↓
 BM Ratio
     ↓
  ZScore
     ↓
 BM Factor
```

Each computation depends on outputs from upstream computations.

A graph makes these dependencies explicit.

---

## Why a DAG?

BagelQuant uses a Directed Acyclic Graph (DAG).

### Directed

Dependencies have direction.

```text
A → B
```

means:

```text
B depends on A
```

---

### Acyclic

Cycles are not allowed.

Invalid example:

```text
A → B → C → A
```

A node cannot depend on itself.

This guarantees:

- Deterministic execution
- Valid execution order
- Finite computation

---

## Node Types

A BagelQuant graph contains only three node types.

---

### Panel

State node.

Represents data.

Examples:

```text
Price
Returns
Factor
Prediction
Portfolio Weights
```

Panels do not perform computation.

They store results.

---

### Operator

Unary transformation.

```text
Panel → Panel
```

Examples:

```text
Rank
ZScore
Rolling Mean
Normalize
```

---

### Composer

Multi-input transformation.

```text
(P₁, P₂, ..., Pₙ) → P
```

Examples:

```text
Add
Divide
OLS
Neural Network
Optimizer
```

---

## Graph Structure

A graph is formed by connecting Panels, Operators, and Composers.

Example:

```text
Book Value (Panel)
          \
           Divide (Composer)
          /         |
Price (Panel)       |
                    ↓
            BM Ratio (Panel)
                    ↓
            ZScore (Operator)
                    ↓
            BM Factor (Panel)
```

Every edge represents a dependency.

---

## Example: Factor Construction

A Book-to-Market factor can be represented as a graph (see above).

This graph converts raw accounting and market data into an investable factor.

---

## Example: Prediction Model

Multiple factors can be combined into a prediction.

```text
BM Factor (Panel)
                  \
                   \
ROE Factor (Panel)──────── Neural Network (Composer)
                   /              |
                  /               |
Momentum Factor (Panel)           |
                                  ↓
                              Prediction (Panel)
```

The prediction itself is represented as a Panel.

---

## Example: Portfolio Construction

Predictions can be converted into portfolio weights.

```text
Prediction (Panel)
      ↓
 Normalize (Operator)
      ↓
 Portfolio Weights (Panel)
```

Portfolio construction becomes another graph transformation.

---

## Example: End-to-End Strategy

An entire investment process can be represented as a single graph.

```text
Book Value (Panel)
          \
           Divide (Composer)
          /         |
Price (Panel)       |
                    ↓
            BM Ratio (Panel)
                    ↓
            ZScore (Operator)
                    ↓
            BM Factor (Panel)
                    \
                     \
  ROE Factor (Panel)──────── Neural Network (Composer)
                     /              |
                    /               |
  Momentum Factor (Panel)           |
                                    ↓
                                Prediction (Panel)
                                    ↓
                                Normalize (Operator)
                                    ↓
                            Portfolio Weights (Panel)
```

The same abstraction applies at every level.

---

## Reusability

One of the primary advantages of a graph structure is node reuse.

Example:

```text
                BM Factor
               /         \
              /           \
     Prediction Model   Analysis
```

The same factor can be used by multiple downstream components.

Without duplication.

---

## Dependency Tracking

The graph explicitly records dependencies.

Example:

```text
Price
  ↓
Returns
  ↓
Momentum
```

Dependencies:

```text
Momentum depends on Returns

Returns depends on Price
```

This enables automatic dependency resolution.

---

## Incremental Computation

Suppose:

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

only affected downstream nodes need to be recomputed.

```text
Price
  ↓
Returns
  ↓
Momentum
```

Unrelated graph branches remain unchanged.

This significantly reduces computation costs.

---

## Caching

Graph nodes can be cached.

Example:

```text
Price
   ↓
Returns
   ↓
Momentum
```

If Returns has already been computed:

```text
Returns
```

can be reused rather than recomputed.

Benefits:

- Faster execution
- Lower computational cost
- Improved scalability

---

## Graph Execution

Graphs do not define execution.

Graphs define dependencies.

Execution is handled by the Runtime Engine.

The Runtime Engine is responsible for:

- Validation
- Dependency resolution
- Topological sorting
- Scheduling
- Execution
- Caching

The Graph only describes structure.

---

## Graph Granularity

Graphs should be constructed from small reusable components.

Preferred:

```text
BM Ratio
    ↓
 ZScore
    ↓
Neutralize
```

Not:

```text
GenerateBookToMarketFactor
```

Smaller nodes provide:

- Better visibility
- Better debugging
- Better reuse
- Better caching

---

## Graph as a Research Object

In BagelQuant, a graph is not merely an execution plan.

A graph represents a research hypothesis.

For example:

```text
Value Factor
Quality Factor
Momentum Factor
        ↓
 Weighted Sum
        ↓
 Prediction
```

This graph encodes an investment belief:

```text
Value + Quality + Momentum
can predict future returns.
```

The graph itself becomes a first-class research artifact.

---

## Design Goals

The graph system is designed to provide:

### Modularity

Research components can be assembled like building blocks.

---

### Reusability

Nodes can be reused across workflows.

---

### Transparency

Every transformation is visible.

---

### Reproducibility

Graphs fully describe research pipelines.

---

### Scalability

Graphs support incremental and distributed execution.

---

## Summary

A Graph is the computational backbone of BagelQuant.

It connects Panels, Operators, and Composers into a Directed Acyclic Graph that represents quantitative research workflows.

By expressing factors, prediction models, portfolio construction processes, and entire investment strategies as graphs, BagelQuant provides a unified, modular, and reproducible framework for quantitative research.

