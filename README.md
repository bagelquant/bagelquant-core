# BagelQuant

> Build quantitative research systems from panels and reusable graph logic.

BagelQuant is a composable framework for quantitative research, portfolio
construction, and backtesting.

## Core Model

- `Domain`: trading sessions and static or dynamic asset membership.
- `Panel`: immutable domain-aware numeric data indexed by time and asset.
- `Graph`: a lazy logic chain that transforms or combines panels.
- **Transformer function**: unary logic, `Panel | Graph -> Graph`.
- **Composer function**: multi-input logic, `(Panel | Graph, ...) -> Graph`.

Raw data enters the system as a `Panel`. Derived objects such as factors,
predictions, and portfolio weights are graphs until execution materializes
their output panels.

## Quick Start

```python
import pandas as pd

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import div, weighted_sum
from bagelquant_core.transformer import rank, rolling_mean, winsorize, zscore

# Define the domain of our research.
# The `Panel.from_domain` method aligns raw data to the domain's sessions and assets.
domain = Domain(
    region="US",
    universe=["AAPL", "MSFT"],
    start_date="2024-01-01",
    end_date="2024-12-31",
)

# Create panels for input data. Replace `pd.DataFrame(...)` with actual data loading logic.
price = Panel.from_domain(pd.DataFrame(...), domain, name="price")
book = Panel.from_domain(pd.DataFrame(...), domain, name="book")
quality = Panel.from_domain(pd.DataFrame(...), domain, name="quality")

# Define a graph for the research logic. Graphs are lazy and can be defined in any order.
bm_ratio = div(book, price, name="bm_ratio")
bm_factor = rank(zscore(winsorize(bm_ratio)), name="bm_factor")
quality_factor = rank(zscore(quality), name="quality_factor")

# A simple prediction graph that combines the factors. The `weights` argument is passed to the `weighted_sum` composer.
prediction = weighted_sum(
    bm_factor,
    quality_factor,
    weights=[0.5, 0.5],
    name="prediction",
)

# Define a graph for the trading signal. The `rank` transformer is applied to the prediction graph.
signal = rank(prediction, name="signal")
smoothed_signal = rolling_mean(signal, window=20, name="smoothed_signal")

# Execute the graph to materialize the output panel.
# Execution is lazy and only happens when `compute` is called.
# It will compute all upstream graphs as needed, caching intermediate results for efficiency.
smoothed_signal.compute()
result = smoothed_signal.output  # Access the output panel after computation.
```

`result` is a `Panel`. Its underlying frame is available as `result.data`.
`Domain` caches exchange sessions in the operating system's user cache
directory. Set `BAGELQUANT_CALENDAR_CACHE_DIR` to override the location.

## Custom Operations

Users can define function-style operations in their own modules:

```python
import pandas as pd

from bagelquant_core.composer import composer
from bagelquant_core.transformer import transformer


@transformer
def demean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


@composer
def average(*frames: pd.DataFrame) -> pd.DataFrame:
    return sum(frames) / len(frames)


centered_price = demean(price, name="centered_price")
combined = average(bm_factor, quality_factor, name="combined")
```

## Cached Outputs

Computing a downstream graph materializes outputs for its intermediate graphs:

```python
signal.compute()

prediction_panel = prediction.output
signal_panel = signal.output
```

Run the complete example:

```bash
uv run python example.py
```

## Development

Package source code lives in `src/bagelquant_core`. The test suite is configured
to import from `src`, so local validation can run from the repository root:

```bash
.\.venv\Scripts\python.exe -m pytest
```

`uv run pytest` is also expected to work in environments where `uv` can resolve
the project Python executable correctly.

## Documentation

- [Proposal](./docs/00_proposal/proposal.md)
- [Refactor plan](./docs/00_proposal/panel-graph-refactor-plan.md)
- [Architecture](./docs/02_architecture/bagelquant%20core%20architecture.md)
- [Panel](./docs/01_concepts/panel.md)
- [Graph](./docs/01_concepts/graph.md)
- [Transformer](./docs/01_concepts/transformer.md)
- [Composer](./docs/01_concepts/composer.md)
- [Execution](./docs/01_concepts/execution.md)
- [API reference](./docs/reference/index.md)
- [Transformer reference](./docs/reference/transformers/index.md)
- [Composer reference](./docs/reference/composers/index.md)

## License

Apache License 2.0
