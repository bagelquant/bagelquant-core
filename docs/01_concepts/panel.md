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
- Cache values stored by the internal execution runtime

Factors, predictions, and portfolio weights are normally represented as lazy
graphs until their output panels are needed.

## Invariants

Panels:

- Have a one-dimensional unique index
- Have one-dimensional unique columns
- Contain only numeric values
- Copy input data at construction
- Return a defensive copy when data is accessed through `Panel.data`
- Are immutable from the public API

## Alignment

Multi-input composer functions align panel data across time and assets before
computation. The default join is the intersection of indexes and columns.
Already-aligned frames are reused internally, avoiding unnecessary copies.

## Category Panels

`CategoryPanel` is an immutable leaf node for labels such as industry, sector,
or country. It follows the same time-by-asset shape as `Panel` but accepts
string labels:

```python
import pandas as pd

from bagelquant_core import CategoryPanel

industry = CategoryPanel(
    pd.DataFrame(...),
    name="industry",
)
```

Use category panels with the category operations exported from
`bagelquant_core.transformer`.
