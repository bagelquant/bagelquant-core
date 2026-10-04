# BagelQuant Core

`bagelquant-core` is the Polars-native panel and graph layer for BagelQuant.

Panel data is long-form and keyed by `time` and `asset_id`. Numeric panels use a
single `value` column:

```python
import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import div
from bagelquant_core.transformer import rank, zscore

static_domain = Domain(
    calendar=["2024-01-02", "2024-01-03"],
    universe=["AAA", "BBB"],
)
dynamic_domain = Domain(
    calendar=["2024-01-02", "2024-01-03"],
    universe=pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-03"],
            "asset_id": ["AAA", "BBB"],
            "active": [True, True],
        }
    ),
)
domain = static_domain
price = Panel.from_domain(
    pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02"],
            "asset_id": ["AAA", "BBB"],
            "value": [100.0, 50.0],
        }
    ),
    domain,
    name="price",
)
book = Panel.from_domain(
    pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02"],
            "asset_id": ["AAA", "BBB"],
            "value": [40.0, 25.0],
        }
    ),
    domain,
    name="book",
)

factor = rank(zscore(div(book, price)), name="factor")
factor.compute()

print(factor.output.collect(dense=True))
```

Graphs also have a JSON-compatible declarative representation. Only registered
BagelQuant transformers and composers can be restored, and panel inputs are
resolved explicitly by symbolic name:

```python
specification = factor.spec().to_dict()
compiled = Graph.compile(specification)
result = compiled.compute({"price": price, "book": book})
```

Core 0.7 keeps sparse inputs as Polars `LazyFrame` plans. Use
`panel.collect(dense=False)` for sparse output, `panel.collect()` or
`panel.collect(dense=True)` for a dense domain-aligned result, and pass an
`ExecutionRuntime` when a compiled graph is rebound repeatedly.

Time-series operations group by `asset_id` and order by `time`.
Cross-sectional operations group by `time`.
Composer operations join inputs on `(time, asset_id)`.
Prediction composers produce a strongly typed `PredictionPanel`. Transformer
chains and operations whose primary inputs are all predictions preserve that
type. Target-weight operators explicitly return ordinary numeric Panels.
Universes can be static lists/Series or sparse dynamic membership frames with
`time`, `asset_id`, and boolean `active`; missing dynamic rows are inactive and
are not forward-filled.

`smooth_prediction` is an explicit eager numerical boundary for resumable
Prediction Processing. It preserves `PredictionPanel`, accepts an ordered
evaluation calendar and `PredictionSmoothingConfig` (`none`, `sma`, `ewma`),
and returns a typed prediction plus an immutable serializable checkpoint.
Missing/nonfinite observations and Universe exits reset history; SMA requires
the full contiguous window. Callers, not Core, own storage and orchestration.

Core 0.10 also exposes immutable content-addressed logical DAGs. Authoring
templates remain convenient local graphs; `canonicalize_graph(template,
input_keys=...)` or `graph.logical_spec()` interns their numerical dependencies.
Names, research metadata, input bytes, Domain and snapshots never enter a
logical node ID. Positional input order and named auxiliary dependency roles
do. `LogicalGraphSpec.union()` shares identical fragments across independent
roots, and `Graph.from_logical_spec()` binds the resulting DAG to explicit
source Panels. Generic execution placeholders such as `$data_start` resolve
through `parameter_bindings` without changing the immutable logical definition.

`ExecutionRuntime(materialization_store=adapter)` uses the same sparse planner
with a caller-owned durable store. `MaterializationStore.query()` reports HIT,
PARTIAL or MISS; only a verified exact HIT is admitted. Numerical keys include
operator/runtime versions, relevant immutable input receipts, availability
traces, Domain, resolved parameters and causal checkpoint context. Stored
`NodeMaterialization` records contain typed Panels, checkpoints, fit audits and
generic named artifacts, including `rebalance` decisions. Pending intermediate
values are collected in resource-budgeted sparse batches, then replaced by the
store's scan-backed immutable Panels. Core never chooses a database, snapshot,
research stage, cache retention policy or publication authority.

Give reconstructed inputs a permanent `source_key` and a checksum-backed
`identity`. Checkpoint keys are logical node IDs, so independently constructed
graphs can resume a verified causal prefix. `canonicalize_values` makes a
significant-digit boundary explicit; `project_domain` restricts a consumer's
Domain after computing its upstream source universe.

`ResourceLimits.parallel_nodes` admits independent eager kernels from the DAG's
ready frontier; lazy work stays fused in Polars' shared process thread pool.
Each kernel receives a share of the total thread, LightGBM histogram and
temporary-array budgets. Resident-memory feedback reduces subsequent admission,
shrinks batches and releases optional caches; it is a soft target, so callers
still submit bounded date blocks. `runtime.resource_usage` reports observed
resident memory and actual eager concurrency.

`runtime.plan_materialization_keys(graph)` and `runtime.node_domains` expose the
full numerical identities and Domains without evaluating input payloads.
`causal_history_requirements(spec)` proves finite preceding observation counts
for supported built-ins; unknown or stateful closures require full history.
Counts follow each asset's admitted coordinates. Dynamic membership gaps need a
membership-based halo proof before calendar chunking. Applications own that
proof and immutable block publication. `Graph.from_logical_spec(node_bindings=...)`
binds caller-proven intermediate Panels while retaining the complete logical
definition; block receipts do not become full-domain receipts implicitly.

## Development

AI contributors start with [AGENTS.md](AGENTS.md) and the
[local workflow and topic routes](.ai/README.md). Integrated checkouts use the
verified workspace's bilingual AI workflow guide and ignored task records;
standalone checkouts use the local rules and conversation handoff.

```bash
uv run ruff check .
uv run python -m pytest
```
