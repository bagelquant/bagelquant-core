# BagelQuant Core Architecture

## Unified Node and Operator

Core has two calculation abstractions: nodes and operators. Panel, CategoryPanel,
PredictionPanel, Domain, graph specs, types and runners are infrastructure.
OPERATOR_REGISTRY provides one registry; Transformer, Composer and Prediction
Composer share OperationNode, dependency handling and execution.

```text
Domain / Panel inputs → Operator / OperationNode DAG → runner → Panel / weight values
```

Node.dag() exports a complete JSON-safe graph before execution; Node.mermaid()
exports dependencies. Primary inputs, named auxiliary Panels and state edges
participate in serialization, validation and topology. Operators have explicit
versions, parameters, types, causality, density and documentation contracts.
Transformer / Composer retain their names and mathematics as catalog categories.

## Panel and Domain

Immutable Panels use time/asset_id keys. Domain defines trading sessions and
dynamic membership. Inputs remain sparse until an explicit dense boundary.
Outputs are defensive copies; temporal/cross-sectional operations retain
missingness and causality.

## Graph and execution

Graph collects dependencies, validates cycles, exports specs and delegates
execution. Graph.compile(spec) validates once and binds successive batches.
Shared nodes execute once. Pure Polars operations fuse; NumPy, regression and
optimization create explicit eager barriers. Cache keys include input, Domain
and node configuration.

## Weight and training operators

top_n, equal_weight, regularized_weights, exposure_constrained_weights and
rebalance belong to Core. Optimizers reference historical calculated targets,
never accounts. Rebalance stores full targets and hold/unavailable/rebalance
states anchored at the first Data Start trading session; zero means exit.
Rolling ElasticNet/LightGBM, mature-label windows, deterministic sampling and
fit audits are generic implementations. Callers provide labels, Universe,
market rules, training lifecycle and persistence explicitly.

## Resources and package boundaries

ResourceLimits controls native threads, concurrency, batches, caches and
memory pressure. Hardware settings are identity-neutral; sampling, model and
solver tolerance are numerical operator parameters. Core imports no Data,
BT or Workbench and owns no Provider, account, market rule or evaluation
persistence. BT owns account simulation, return diagnostics, metrics and charts.
