# BagelQuant Core Proposal

## 1. Problem Statement

Quantitative research systems today are highly fragmented.

A typical workflow involves:

- Data extraction (APIs, databases, vendors)
- Feature engineering (scripts / notebooks)
- Factor research (ad-hoc pipelines)
- Signal generation (inconsistent logic)
- Portfolio construction (separate optimization code)
- Backtesting (another isolated system)

Each stage is often implemented using different abstractions and frameworks.

This leads to:

- Poor reusability of research components
- Difficult reproducibility of results
- Hidden dependencies between modules
- High friction in experimenting with new ideas
- Lack of a unified computational model

In practice, most quant workflows are a collection of disconnected scripts rather than a coherent system.

---

## 2. Motivation

The core problem is not computation speed or data availability.

The core problem is **lack of a unified abstraction layer for quantitative research**.

We need a system that treats:

- data
- features
- factors
- predictions
- portfolios

as first-class, composable objects.

---

## 3. Design Principles

BagelQuant Core is built on the following principles:

### 3.1 Everything is a Panel

All quantitative data is represented as a 2D Panel:

- Time × Assets

No special types for:

- prices
- factors
- predictions
- weights

---

### 3.2 Everything is a Transformation

All computations are either:

- Unary transformations (Operator)
- Multi-input transformations (Composer)

---

### 3.3 Everything is a Graph

All research workflows are represented as a Directed Acyclic Graph (DAG).

Graphs define dependencies, not execution.

---

### 3.4 Modularity over Monoliths

Complex strategies are built from small reusable components.

---

## 4. Core Abstractions

BagelQuant is built on three primitives:

### 4.1 Panel

A 2D time-series matrix:

```text
Time × Assets
```

Represents:

- raw data
- features
- factors
- predictions
- portfolio weights

---

### 4.2 Operator

Unary transformation:

```text
Panel → Panel
```

Examples:

- Rank
- Z-score
- Rolling statistics
- Neutralization

---

### 4.3 Composer

Multi-input transformation:

```text
(P₁, P₂, ..., Pₙ) → Panel
```

Examples:

- arithmetic combinations
- statistical models
- machine learning models
- portfolio optimizers

---

### 4.4 Graph

A DAG connecting all components:

- defines dependencies
- enables reuse
- enables incremental computation

---

## 5. System Vision

BagelQuant is not intended to be another factor library.

It is designed as a:

> **Composable operating system for quantitative research**

In this system:

- Factors are not end products
- Predictions are not endpoints
- Portfolios are not separate systems

They are all nodes in a unified graph.

---

## 6. What This System Enables

### 6.1 Reusable Research

A factor can be reused in:

- multiple models
- multiple portfolios
- multiple experiments

---

### 6.2 End-to-End Graphs

A full investment strategy can be represented as a single DAG:

```text
Data → Features → Factors → Prediction → Portfolio → Backtest
```

---

### 6.3 Incremental Computation

Only affected parts of the graph are recomputed when inputs change.

---

### 6.4 Full Reproducibility

A graph fully defines a research experiment.

---

### 6.5 Modular Experimentation

Researchers can swap components without rewriting pipelines.

---

## 7. Scope

### In Scope

- Panel-based computation system
- DAG execution engine
- Operator system
- Composer system
- Backtesting framework
- Visualization tools

---

### Out of Scope

- Trading execution systems
- Broker integration
- Real-time order routing
- Execution optimization (HFT-level infra)

---

## 8. Long-Term Vision

The long-term goal is to evolve BagelQuant into a unified research operating system where:

- every research idea is a graph
- every transformation is reusable
- every experiment is reproducible
- every strategy is composable

---

## 9. Summary

BagelQuant Core proposes a unified computational model for quantitative research based on:

- Panels (data abstraction)
- Operators (unary transformations)
- Composers (multi-input transformations)
- Graphs (dependency structure)

It replaces fragmented research workflows with a single composable system.
