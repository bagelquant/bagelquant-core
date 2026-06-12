# Panel

## Overview

`Panel` is the explicit data object in BagelQuant. Every input panel is
created through a `Domain`, which defines trading sessions and asset
membership. A panel stores a numeric two-dimensional frame:

```text
Time x Assets
```

```python
import pandas as pd

from bagelquant_core import Domain, Panel

domain = Domain(
    calendar=pd.bdate_range("2024-01-01", "2024-12-31"),
    universe=["AAPL", "MSFT"],
)
price = Panel.from_domain(price_df, domain, name="price")
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
- Match their Domain's trading-session index and universe columns
- Mask inactive cells for dynamic universes

## Alignment

Multi-input composer functions require equivalent Domains. The runtime
reapplies dynamic-universe membership after each derived computation so
inactive cells cannot affect later operations.

## Dynamic Universes

A dynamic universe is a boolean frame indexed by dates and asset columns.
Missing rows and cells are inactive; membership is not forward-filled:

```python
membership = pd.DataFrame(
    {"AAPL": [True], "MSFT": [False]},
    index=pd.to_datetime(["2024-01-03"]),
)
domain = Domain(
    calendar=pd.bdate_range("2024-01-01", "2024-01-31"),
    universe=membership,
)
```

## Calendar

`Domain` never retrieves calendars. Provide a non-empty, unique, sorted
calendar from your data layer. The first and last sessions define the domain's
start and end dates.

## Category Panels

`CategoryPanel` is an immutable leaf node for labels such as industry, sector,
or country. It follows the same time-by-asset shape as `Panel` but accepts
string labels:

```python
import pandas as pd

from bagelquant_core import CategoryPanel

industry = CategoryPanel.from_domain(
    pd.DataFrame(...),
    domain,
    name="industry",
)
```

Use category panels with the category operations exported from
`bagelquant_core.transformer`.
