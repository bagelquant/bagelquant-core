# Panels, operators and catalog

## Data and typed boundaries

- Panel data is immutable, long-form, keyed by `(time, asset_id)`; numeric
  Panels use `value`. Preserve sparse/dense and missing-membership semantics.
- Time-series operations group by `asset_id` and order by `time`; cross-sectional
  operations group by `time`; peer inputs align on exact `(time, asset_id)` keys.
- Apply Universe membership before cross-sectional transforms. Missing dynamic
  membership rows are inactive, never forward-filled. Preserve lazy sparse plans;
  do not collect/densify eagerly without a correctness or contract reason.
- Ordinary Transformers/Composers produce AlphaValue `Panel`.
  `PredictionComposer` produces typed `PredictionPanel` for BT's prediction
  boundary; BT's saved-target boundary is separate. Do not bypass the applicable
  typed boundary.
- `PredictionComposer` produces `PredictionPanel`. It may feed only a
  Transformer's semantic input; that Transformer preserves the type. Prediction
  never feeds a Composer or an auxiliary Panel parameter.

Known implementation discrepancy, deferred to Core refactor stage 3: the current
`graph.py` type propagation and
`test_predictions_can_share_multi_input_operators` permit all-prediction Composer
peers. The preceding restriction retains the original agent governance contract
until that stage resolves the type-flow contract. Stage-1 workflow maintenance
does not remove the existing API or silently change numerical/type-flow behavior.

Data returns neutral data/evidence without Core types. Generic conversion and
Domain/Panel validation belong to Core public APIs; Workbench supplies China
semantics and research bindings. Generic numerical artifact persistence is a
stage-3 target under [Graph and runtime](graph-runtime.md), not an existing store.

## Operator shape and names

- Transformers are stateless and have exactly one semantic `input`; auxiliary
  Panels are keyword-only named `panel_parameters`. Composers describe graph
  structure, have at least two peer `inputs`, and no Panel parameters.
- Every primary/auxiliary dependency participates in topology, Domain alignment,
  cache identity and availability tracing. A Transformer uses the maximum
  availability across its semantic input and all Panel parameters.
- Grouping, neutralization, masking, projection, volatility scaling, logical
  negation and rolling regression are Transformers. Use keyword-only forms such
  as `group_demean(source, group=industry)` and
  `orthogonalize(source, factors=(size, beta), fit_intercept=False)`.
- Keep canonical names only. Do not restore `category_*`: demean/mean/zscore
  map to `group_*`, while old `category_rank` maps mathematically to
  `group_percentile`, not `group_rank`. Do not restore arithmetic aliases
  `subtract`, `multiply`, `divide`, `min` or `max`.
- Nodes have no hidden dependencies or side effects. Graphs are acyclic,
  allowlisted, serializable and reproducible. Keep factor computation separate
  from evaluation prices, returns and governance. Numerical operators remain
  separate from explicit artifact persistence/publication at the Core runtime
  boundary; generic storage implementation is pending stage 3.

## Catalog maintenance

After public Transformer/Composer export or documentation-metadata changes,
run `uv run python scripts/generate_operator_reference.py`. Stable functional
categories and executable `OperationExample` fixtures are public contracts:
each operation page shows actual input, Panel-parameter and output tables from
the fixture. Coordinate Workbench governed catalog changes independently.
