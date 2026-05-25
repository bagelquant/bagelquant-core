# Code Convention

This document defines the coding standards for this repository, which is designed for a **quantitative research + DAG-based factor system**.

The goal is:
> Make code composable, testable, and easy to reason about in a DAG/expression system.

---

# 1. General Principles

## 1.1 Clarity > Cleverness
- Prefer explicit logic over implicit tricks
- Avoid one-liners that hide intent

## 1.2 Composition-first design
- Everything should be composable (Panel → Operator → Composer → DAG)
- Avoid tightly coupled logic

## 1.3 Functional mindset (where possible)
- Avoid hidden side effects
- Prefer pure transformations for data flow nodes

---

# 2. Project Structure Convention

```plaintext
bagel_factor/
  core/
    panel/
    operator/
    composer/
    dag/
  execution/
  evaluation/
  utils/
  examples/
  tests/
```

Rules:
- `core/` = fundamental abstractions
- `execution/` = runtime logic
- `evaluation/` = IC, returns, metrics
- `utils/` = stateless helpers only

---

# 3. Naming Convention

## 3.1 Variables
- Use descriptive names
- Avoid abbreviations unless standard in quant finance

Good:
```python
price_series
book_to_market_ratio
rolling_ic
```

Bad:
```python
p, btm, r_ic
```

---

## 3.2 Functions
- Must be verb-based

Good:
```python
compute_ic()
build_dag()
evaluate_factor()
normalize_signal()
```

Bad:
```python
ic()
dag()
factor()
```

---

## 3.3 Classes

- Use PascalCase
- Must represent a concept, not a function

Good:
```python
Panel
Operator
Composer
ExpressionNode
DAGEngine
```

---

# 4. Core Abstraction Rules

## 4.1 Panel (data container)

Rules:
- Must be immutable (or treated as immutable)
- Must not contain business logic
- Only holds structured data

```python
class Panel:
    def __init__(self, data: pd.DataFrame):
        self._data = data
```

---

## 4.2 Operator (transformation unit)

Rules:
- Takes one or more Panels
- Outputs one Panel
- Must be stateless

```python
class Operator:
    def compute(self, *inputs: Panel) -> Panel:
        raise NotImplementedError
```

---

## 4.3 Composer (composition layer)

Rules:
- Combines operators into DAG structure
- No numeric computation logic inside
- Only structure definition

---

## 4.4 DAG (execution graph)

Rules:
- Nodes = operators
- Edges = dependencies
- Must be acyclic

Validation rules:
- No circular dependencies allowed
- All nodes must be resolvable

---

## 4.5 Execution Engine

Rules:
- Executes DAG topologically
- Handles caching (if needed)
- No business logic allowed

---

# 5. Type Hinting Convention

Always use type hints:

```python
from typing import List, Dict, Optional
import pandas as pd
```

Example:

```python
def compute_ic(
    factor: pd.Series,
    returns: pd.Series
) -> float:
    ...
```

---

# 6. Docstring Convention

Use structured docstrings:

```python
def compute_ic(factor, returns):
    """
    Compute Information Coefficient (IC).

    Parameters
    ----------
    factor : pd.Series
    returns : pd.Series

    Returns
    -------
    float
        Pearson correlation between factor and returns
    """
```

---

# 7. Error Handling Convention

## 7.1 Fail fast
- Do not silently ignore errors

Bad:
```python
try:
    ...
except:
    pass
```

Good:
```python
raise ValueError("Invalid panel input: missing column price")
```

---

## 7.2 Validation at boundaries
- Validate inputs at system entry points (Panel / DAG input)

---

# 8. Logging Convention

- Use logging module
- Never use print() in core logic

```python
import logging
logger = logging.getLogger(__name__)
```

Rules:
- DEBUG → internal computation
- INFO → pipeline state
- WARNING → unexpected but recoverable
- ERROR → failure cases

---

# 9. Testing Convention

## 9.1 Unit tests required for:
- Operator logic
- DAG execution correctness
- Evaluation metrics (IC, returns)

## 9.2 Test naming:
```python
test_compute_ic_basic()
test_dag_topological_order()
test_operator_missing_input()
```

## 9.3 Golden rule:
> Every operator must have at least one test case.

---

# 10. Performance Convention

- Prefer vectorized pandas/numpy operations
- Avoid Python loops over time-series data
- Cache repeated DAG computations if necessary

Bad:
```python
for i in range(len(df)):
    ...
```

Good:
```python
df['signal'] = df['price'].rolling(20).mean()
```

---

# 11. DAG-Specific Rules

## 11.1 No hidden dependencies

- All dependencies must be explicit in DAG edges

## 11.2 Deterministic execution

- Same input → same output

## 11.3 No side effects in nodes
- Nodes must not modify global state

---

# 12. Import Convention

Order:

```python
# standard library
import os

# third-party
import pandas as pd
import numpy as np

# local
from bagel_factor.core import Panel
```

---

# 13. Code Style

- Follow black formatting (recommended)
- Line length: 88
- Use trailing commas in multi-line structures

---

# 14. Summary

Core idea:

> Code must behave like a DAG: explicit, composable, deterministic, and testable.

Every component should answer:

- What does it take in?
- What does it output?
- Is it composable in a DAG?

