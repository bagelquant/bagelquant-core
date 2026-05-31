# BagelQuant

> Build quantitative research systems from panels and reusable graph logic.

BagelQuant is a composable framework for quantitative research, portfolio
construction, and backtesting.

## Core Model

- `Panel`: immutable numeric data indexed by time and asset.
- `Graph`: a lazy logic chain that transforms or combines panels.
- Transformer function: unary logic, `Panel | Graph -> Graph`.
- Composer function: multi-input logic, `(Panel | Graph, ...) -> Graph`.

Raw data enters the system as a `Panel`. Derived objects such as factors,
predictions, and portfolio weights are graphs until execution materializes
their output panels.

## Quick Start

```python
import pandas as pd

from bagelquant_core import Panel
from bagelquant_core.composer import div, weighted_sum
from bagelquant_core.transformer import rank, winsorize, zscore

price = Panel(pd.DataFrame(...), name="price")
book = Panel(pd.DataFrame(...), name="book")
quality = Panel(pd.DataFrame(...), name="quality")

bm_ratio = div(book, price, name="bm_ratio")
bm_factor = rank(zscore(winsorize(bm_ratio)), name="bm_factor")
quality_factor = rank(zscore(quality), name="quality_factor")

prediction = weighted_sum(
    bm_factor,
    quality_factor,
    weights=[0.5, 0.5],
    name="prediction",
)
signal = rank(prediction, name="signal")

signal.compute()
result = signal.output
```

`result` is a `Panel`. Its underlying frame is available as `result.data`.

## Custom Operations

Users can define function-style operations in their own modules:

```python
import pandas as pd

from bagelquant_core.composer import composer
from bagelquant_core.transformer import transformer


@transformer
def rolling_mean(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    return frame.rolling(window).mean()


@composer
def average(*frames: pd.DataFrame) -> pd.DataFrame:
    return sum(frames) / len(frames)


smoothed_price = rolling_mean(price, window=20, name="smoothed_price")
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

## Documentation

- [Proposal](./docs/00_proposal/proposal.md)
- [Refactor plan](./docs/00_proposal/panel-graph-refactor-plan.md)
- [Architecture](./docs/02_architecture/bagelquant%20core%20architecture.md)
- [Panel](./docs/01_concepts/panel.md)
- [Graph](./docs/01_concepts/graph.md)
- [Transformer](./docs/01_concepts/transformer.md)
- [Composer](./docs/01_concepts/composer.md)
- [Execution](./docs/01_concepts/execution.md)

## License

Apache License 2.0
