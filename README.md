# BagelQuant 

> Build quantitative research systems like LEGO bricks.

BagelQuant is a composable framework for quantitative research, portfolio construction, and backtesting.

It provides a unified way to represent data, features, alpha signals, predictions, and portfolio weights as reusable building blocks that can be composed into directed acyclic graphs (DAGs).

Instead of treating factor research, signal generation, portfolio construction, and backtesting as separate workflows, BagelQuant models them as a single computational graph.

## Why BagelQuant?

Quantitative research workflows are often fragmented.

A typical pipeline may involve:

- Market data processing
- Feature engineering
- Alpha research
- Signal generation
- Portfolio construction
- Backtesting
- Performance analysis

These stages are usually implemented as disconnected scripts, notebooks, and libraries.

BagelQuant aims to unify them into a single composable framework.

## Core Philosophy

Three principles drive the design:

### 1. Everything is a Panel

A Panel is a two-dimensional matrix:

- Index → Time
- Columns → Assets
- Values → Numeric data

Panels can represent:

- Prices
- Returns
- Features
- Factors/Alphas
- Predictions
- Portfolio weights

### 2. Everything is a Transformation

Transformations are represented by two primitives:

#### Operator

Unary transformation:

```text
Panel → Panel
```

Examples:

- Rank
- Z-score
- Rolling Mean
- Demean
- Volatility Scaling

#### Composer

Multi-input transformation:

```text
(Panel₁, Panel₂, ..., Panelₙ) → Panel
```

Examples:

- Linear models
- Machine learning models
- Alpha combination
- Portfolio optimization
- Signal aggregation

### 3. Everything is a DAG

BagelQuant represents quantitative workflows as directed acyclic graphs (DAGs).

A DAG is not limited to feature engineering. The same abstraction can be used for factor construction, alpha modeling, portfolio construction, and backtesting.

#### Example 1: Factor Construction

Book-to-Market (BM) factor can be represented as a DAG:

```text
Book Value (Panel) ──┐
                     ├── Divide (Composer)
Price (Panel) ───────┘
                           │
                           ▼
                    BM Ratio (Panel)
                           │
                           ▼
                  Z-Score (Operator)
                           │
                           ▼
               Industry Neutralize (Operator)
                           │
                           ▼
                    BM Factor (Panel)
```


This workflow combines raw accounting and market data into a normalized cross-sectional factor.

### Example 2: Portfolio Construction

Portfolio construction can be represented using exactly the same DAG abstraction:


```text
BM Factor (Panel) ──────────────┐
ROE Factor (Panel) ─────────────┤
Momentum Factor (Panel) ────────┤
Quality Factor (Panel) ─────────┤
                                ▼
                     Neural Network (Composer)
                                │
                                ▼
                      Prediction (Panel)
                                │
                                ▼
                Cap-Weighted Transform (Operator)
                                │
                                ▼
                  Portfolio Weights (Panel)
```

Here, multiple factors are combined into a prediction model and transformed into investable portfolio weights.


### Example 3: End-to-End Research Pipeline

Entire quantitative research pipelines can be represented as a single graph:

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
  ROE Factor (Panel)────────────── OLS (Composer)
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

This unified representation enables:

- Reusable research components
- Dependency tracking
- Incremental recomputation
- Experiment reproducibility
- Modular strategy development

> In BagelQuant, factors, predictions, portfolio weights, and even backtest outputs are all Panels connected by Operators and Composers in a single computational graph. Build your research workflow like LEGO bricks, where each component can be reused, recomputed, and reconfigured with ease.

## Core concepts

You can find core concepts in the documentation:

- [Panel](./docs/01_concepts/panel.md)
- [Operator](./docs/01_concepts/operator.md)
- [Composer](./docs/01_concepts/composer.md)
- [Graph](./docs/01_concepts/graph.md)
- [Execution](./docs/01_concepts/execution.md)



## Project Scope

BagelQuant Core is designed to support:

### Data Processing

- Data ingestion
- Data alignment
- Feature generation

### Alpha Research

- Cross-sectional factors
- Time-series factors
- Multi-factor models

### Portfolio Construction

- Signal combination
- Portfolio optimization
- Risk constraints

### Backtesting

- Portfolio simulation
- Transaction costs
- Performance analytics

### Visualization

- Time series analysis
- Distribution analysis
- Diagnostics


## Project Status

BagelQuant Core is currently under active development.

Current focus:

- Panel abstraction
- DAG engine
- Operator system
- Composer system
- Runtime execution engine

## Vision

BagelQuant Core is not intended to be another factor library.

The goal is to provide a modular computation operating system for quantitative research, where complex research workflows can be assembled from simple, reusable building blocks.

## License

Apache License 2.0
