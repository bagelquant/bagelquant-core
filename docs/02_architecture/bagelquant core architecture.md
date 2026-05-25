# BagelQuant Core Architecture

## 1. System Overview

BagelQuant Core is a composable quantitative research system built around a Directed Acyclic Graph (DAG) execution engine.

The system unifies:

- Data processing
- Feature engineering
- Alpha modeling
- Portfolio construction
- Backtesting

into a single computational framework.

At the core of the system are three primitives:

- **Panel**: state container (Time × Assets)
- **Operator**: unary transformation (P → P)
- **Composer**: multi-input transformation (Pⁿ → P)

All computation is expressed as a Graph of these primitives.

---

## 2. High-Level Architecture

The system is composed of five main layers:

```text
User Layer
    ↓
Graph API Layer
    ↓
Graph Engine
    ↓
Runtime Execution Engine
    ↓
Data / Cache Layer
```

---

## 3. Core Object Model

### 3.1 Panel

Panels represent all data in the system:

- raw market data
- features
- factors
- predictions
- portfolio weights

A Panel is immutable and indexed by:

```text
Time × Assets
```

---

### 3.2 Node Types

The Graph consists of three node types:

#### Panel Node
Stores data state.

#### Operator Node
Unary transformation:

```text
P → P
```

#### Composer Node
Multi-input transformation:

```text
(P₁, P₂, ..., Pₙ) → P
```

---

### 3.3 Edge Model

Edges represent dependencies:

```text
A → B
```

Meaning:

> B depends on A

Edges are directed and acyclic.

---

## 4. Graph Engine Design

The Graph Engine is responsible for:

- building DAG structure
- validating graph correctness
- resolving dependencies
- generating execution plan

### 4.1 Internal Representation

Graph is stored as:

```python
Node {
    id: str
    type: Panel | Operator | Composer
    inputs: list[node_id]
    outputs: list[node_id]
    metadata: dict
}
```

Edge is implicit via node inputs.

---

### 4.2 Validation Rules

The graph must satisfy:

- No cycles (DAG constraint)
- All inputs must exist
- Type compatibility must be valid
- Composer must have ≥ 1 inputs
- Operator must have exactly 1 input

---

### 4.3 Dependency Resolution

Given a target node, the engine computes:

```text
All upstream dependencies
```

via graph traversal.

---

## 5. Execution Engine

The execution engine transforms a DAG into computed Panels.

### 5.1 Execution Pipeline

```text
Graph → Dependency Graph
      → Topological Sort
      → Execution Plan
      → Runtime Execution
      → Cached Results
```

---

### 5.2 Topological Execution

Nodes are executed in dependency order:

```text
Price
  ↓
Returns
  ↓
Momentum
```

Execution order:

1. Price
2. Returns
3. Momentum

---

### 5.3 Execution Semantics

Each node executes as:

- Operator: `f(P)`
- Composer: `f(P1, P2, ..., Pn)`

Output is always a Panel.

---

### 5.4 Determinism

Execution is:

- deterministic
- stateless
- reproducible

Same input graph → same output.

---

## 6. Runtime System

The runtime is responsible for actual computation.

### 6.1 Responsibilities

- executing operators/composers
- managing intermediate results
- handling parallel execution
- caching results
- incremental recomputation

---

### 6.2 Execution Strategy

The runtime supports:

#### Sequential execution
Default for small graphs.

#### Parallel execution
Independent branches executed concurrently.

```text
Factor A ─┐
          ├→ Parallel Execution
Factor B ─┘
```

#### Incremental execution
Only recompute affected nodes when inputs change.

---

## 7. Caching Layer

Caching is a core performance component.

### 7.1 Cache Key

Each node result is cached using:

```text
(node_id + input_hash + config_hash)
```

---

### 7.2 Cache Behavior

If inputs unchanged:

```text
reuse cached Panel
```

Otherwise:

```text
recompute node
```

---

### 7.3 Benefits

- reduces redundant computation
- enables fast iteration
- supports large DAGs

---

## 8. Data Flow Model

Data flows strictly in one direction:

```text
Raw Data → Transformations → Signals → Portfolio → Metrics
```

At every stage:

```text
Input Panel → Node → Output Panel
```

No in-place mutation is allowed.

---

## 9. Operator & Composer Dispatch

### 9.1 Registry System

Operators and Composers are registered:

```python
operator_registry.register("zscore", ZScoreOperator)
composer_registry.register("divide", DivideComposer)
```

---

### 9.2 Dynamic Dispatch

During execution:

```text
Node type → lookup implementation → execute
```

---

### 9.3 Extensibility

New components can be added without modifying core engine.

---

## 10. Backtesting Integration (Design-Level)

Backtesting is treated as a downstream Composer:

```text
Portfolio Weights
        ↓
    Backtest Engine
        ↓
 Performance Metrics
```

Backtesting consumes Panels and produces Panels.

---

## 11. Visualization Layer (Design-Level)

Visualization is also a graph consumer:

```text
Panel → Visualization Node → Output Plot
```

Supports:

- time series
- distribution
- correlation
- portfolio analytics

---

## 12. Minimal Viable Product (MVP)

Initial implementation should support:

### Core Features

- Panel abstraction
- Operator execution
- Composer execution
- DAG construction
- Topological execution
- Basic caching

### Excluded Initially

- distributed execution
- streaming data
- GPU acceleration
- full backtesting engine
- UI visualization system

---

## 13. Future Extensions

The architecture is designed to support:

### Distributed Execution
- multi-node graph execution

### Streaming Mode
- real-time data updates

### Experiment Tracking
- versioned graphs

### GPU Acceleration
- matrix-based Panel ops

---

## 14. Design Principles

### 14.1 Composability

Everything is a reusable node.

---

### 14.2 Determinism

Same graph + same input → same output.

---

### 14.3 Immutability

Panels are never modified in place.

---

### 14.4 Transparency

Every computation is traceable in the graph.

---

### 14.5 Modularity

Engine is independent of domain logic.

---

## 15. Summary

BagelQuant Core architecture defines a unified system for quantitative research based on:

- Panels (data abstraction)
- Operators (unary transformations)
- Composers (multi-input transformations)
- Graph (dependency structure)
- Runtime (execution engine)

The architecture enables quantitative workflows to be represented, executed, and reused as composable DAGs.
