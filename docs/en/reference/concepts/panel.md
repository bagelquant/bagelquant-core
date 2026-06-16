# Panel

## Overview

`Panel` is the explicit data object in BagelQuant. Every input panel is
created through a `Domain`, which defines trading times and asset membership.
A panel stores long-form data:

```text
time, asset_id, value
```

```python
import polars as pl

from bagelquant_core import Domain, Panel

domain = Domain(
    calendar=["2024-01-02", "2024-01-03"],
    universe=["AAPL", "MSFT"],
)
price = Panel.from_domain(
    pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "asset_id": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "value": [185.0, 370.0, 187.0, 372.0],
        }
    ),
    domain,
    name="price",
)
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

- Have unique `(time, asset_id)` keys
- Use `time`, `asset_id`, and `value` columns
- Contain only numeric values
- Copy input data at construction
- Return a defensive copy when data is accessed through `Panel.data`
- Are immutable from the public API
- Match their Domain's trading times and asset ids
- Mask inactive cells for dynamic universes

## Alignment

Multi-input composer functions require equivalent Domains. The runtime
reapplies dynamic-universe membership after each derived computation so
inactive cells cannot affect later operations.

## Dynamic Universes

A dynamic universe is a sparse long-form boolean frame keyed by `time` and
`asset_id`. It uses the columns `time`, `asset_id`, and `active`. Missing rows
are inactive; membership is not forward-filled:

```python
membership = pl.DataFrame(
    {
        "time": ["2024-01-03", "2024-01-03"],
        "asset_id": ["AAPL", "MSFT"],
        "active": [True, False],
    }
)
domain = Domain(
    calendar=["2024-01-02", "2024-01-03"],
    universe=membership,
)
```

## Calendar

`Domain` never retrieves calendars. Provide a non-empty, unique, sorted
calendar from your data layer. The first and last sessions define the domain's
start and end dates.

Static calendars and universes may be provided as Python sequences or Polars
Series:

```python
domain = Domain(
    calendar=pl.Series("sessions", ["2024-01-02", "2024-01-03"]),
    universe=pl.Series("symbols", ["AAPL", "MSFT"]),
)
```

## Category Panels

`CategoryPanel` is an immutable leaf node for labels such as industry, sector,
or country. It follows the same long-form shape as `Panel` but accepts string
labels:

```python
import polars as pl

from bagelquant_core import CategoryPanel

industry = CategoryPanel.from_domain(
    pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02"],
            "asset_id": ["AAPL", "MSFT"],
            "value": ["tech", "software"],
        }
    ),
    domain,
    name="industry",
)
```

Use category panels with the category operations exported from
`bagelquant_core.transformer`.
