# BagelQuant Core Proposal

## Problem

Quantitative workflows are often split across scripts for data processing,
factor research, modeling, portfolio construction, and backtesting. This hides
dependencies and makes experiments difficult to reproduce.

## Proposal

BagelQuant uses an explicit data boundary and a lazy graph model:

```python
price = Panel.from_domain(price_df, domain, name="price")
book = Panel.from_domain(book_df, domain, name="book")
bm_factor = rank(zscore(div(book, price)), name="bm_factor")
```

## Design Principles

### Data is a Panel

Raw inputs and computed outputs are immutable panels indexed by time and
asset. Panels copy incoming frames and return defensive copies through their
public data access.

### Derived research objects are Graphs

Factors, predictions, portfolio weights, and metrics are lazy graphs until
execution materializes their panel outputs.

### Operations are functions

Transformer and composer functions live outside `Graph`. A library can add
many operations without expanding the graph core.

### Graphs define dependencies

Graphs describe logic chains. The runtime determines evaluation order,
alignment, caching, and output materialization.

## Vision

Every research idea is a reusable graph. Every data boundary is explicit.
Researchers can add operations in their own modules with ordinary decorated
functions.
