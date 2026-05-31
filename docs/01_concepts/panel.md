# Panel

## Overview

`Panel` is the explicit data object in BagelQuant. It stores a numeric
two-dimensional frame:

```text
Time x Assets
```

```python
price = Panel(price_df, name="price")
```

## Role

Panels are:

- Raw input data
- DAG leaf nodes
- Materialized graph outputs
- Cache values stored by the execution engine

Factors, predictions, and portfolio weights are normally represented as lazy
graphs until their output panels are needed.

## Invariants

Panels:

- Have a one-dimensional unique index
- Have one-dimensional unique columns
- Contain only numeric values
- Copy input data at construction
- Are treated as immutable

## Alignment

Multi-input composer functions align panel data across time and assets before
computation. The default join is the intersection of indexes and columns.
