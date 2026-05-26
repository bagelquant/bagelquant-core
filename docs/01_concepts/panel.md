# Panel

## Overview

Panel is the fundamental data abstraction in BagelQuant.

Every piece of information in the system is represented as a Panel, including:

- Market data
- Features
- Factors
- Predictions
- Portfolio weights
- Backtest outputs

A Panel is a two-dimensional matrix indexed by time and assets.

```text
           AAPL    MSFT    NVDA
2025-01-01  100     200     300
2025-01-02  101     198     305
2025-01-03  103     201     310
```

---

## Definition

A Panel consists of:

```python
Panel(
    index: DatetimeIndex,
    columns: AssetIndex,
    values: ndarray,
    metadata: dict
)
```

### Index

The row index represents time.

```text
2025-01-01
2025-01-02
2025-01-03
```

Typical frequencies include:

- Daily
- Hourly
- Minute
- Tick

---

### Columns

The column index represents assets.

```text
AAPL
MSFT
NVDA
```

Assets may represent:

- Equities
- ETFs
- Futures
- Options
- Cryptocurrencies
- Any other tradable instrument

---

### Values

Values are numerical observations associated with a specific time and asset.

Examples:

```text
Price
Return
Book Value
ROE
Prediction
Weight
```

---

### Metadata

Metadata provides contextual information about the Panel.

Examples:

```python
{
    "name": "bm_factor",
    "frequency": "daily",
    "universe": "sp500",
    "description": "Book-to-market factor"
}
```

Metadata should not affect computation and is intended for identification, validation, and documentation.

---

## Philosophy

BagelQuant intentionally uses a single data abstraction.

There is no distinction between:

- PricePanel
- ReturnPanel
- FeaturePanel
- FactorPanel
- PredictionPanel
- WeightPanel

These are all Panels.

The meaning of a Panel is determined by:

1. Its contents
2. Its metadata
3. Its position within the graph

This design minimizes complexity and maximizes composability.

---

## Examples

### Market Data

```text
Close Price Panel
```

```text
           AAPL    MSFT
2025-01-01  100     200
2025-01-02  101     198
```

---

### Factor

```text
Book-to-Market Factor Panel
```

```text
           AAPL    MSFT
2025-01-01  0.85    0.32
2025-01-02  0.90    0.28
```

---

### Prediction

```text
Expected Return Panel
```

```text
           AAPL    MSFT
2025-01-01  0.012   0.008
2025-01-02  0.015   0.005
```

---

### Portfolio Weights

```text
Portfolio Weight Panel
```

```text
           AAPL    MSFT
2025-01-01  0.06    0.03
2025-01-02  0.05    0.04
```

---

## Panel Lifecycle

A Panel may evolve through multiple stages within a graph.

Example:

```text
Close Price
     ↓
Returns
     ↓
Momentum Factor
     ↓
Prediction
     ↓
Portfolio Weight
```

Each stage produces a new Panel.

Panels are transformed, not modified in place.

---

## Immutability

Panels should be treated as immutable objects.

Instead of:

```python
panel.data = new_values
```

prefer:

```python
new_panel = transform(panel)
```

Immutability simplifies:

- Dependency tracking
- Caching
- Reproducibility
- Incremental computation

---

## Alignment

Panels participating in the same computation must be aligned.

Alignment occurs across:

- Time index
- Asset universe

Example:

```text
Panel A:
2025-01-01  AAPL MSFT

Panel B:
2025-01-01  AAPL NVDA
```

The system must resolve:

- Missing dates
- Missing assets
- Universe mismatches

before computation.

---

## Invariants

Every Panel must satisfy:

### Two-dimensional

```text
Time × Assets
```

---

### Time-indexed

Rows represent observations through time.

---

### Asset-indexed

Columns represent assets.

---

### Numeric

Values must be numeric.

---

### Immutable

Panels are not modified after creation.

---

## Design Goals

The Panel abstraction is designed to provide:

### Simplicity

One data structure for the entire system.

---

### Consistency

Every computation consumes and produces Panels.

---

### Reusability

Panels can be reused across multiple graph branches.

---

### Composability

Panels can flow through arbitrary Transformers and Composers.

---

### Graph Compatibility

Panels are first-class nodes within the DAG.

---

## Summary

Panel is the fundamental state container in BagelQuant.

All information in the system—raw data, features, factors, predictions, portfolio weights, and backtest outputs—is represented as a Panel.

By standardizing on a single immutable data abstraction, BagelQuant achieves a simple, composable, and reusable foundation for quantitative research and portfolio construction.
