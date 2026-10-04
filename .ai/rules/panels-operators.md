# Panels, operators and catalog

## Data and typed boundaries

- Panel data is immutable, long-form, keyed by `(time, asset_id)`; numeric
  Panels use `value`. Preserve sparse/dense and missing-membership semantics.
- Time-series operations group by `asset_id` and order by `time`; cross-sectional
  operations group by `time`; peer inputs align on exact `(time, asset_id)` keys.
- Apply Universe membership before cross-sectional transforms. Missing dynamic
  membership rows are inactive, never forward-filled. Preserve lazy sparse plans;
  do not collect/densify eagerly without a correctness or contract reason.
- Ordinary Transformers/Composers produce AlphaValue `Panel`; only allowlisted
  `SignalComposer` produces the terminal `SignalPanel` accepted by the public
  BT signal backtest boundary. Never bypass it using frames, Panels or weights.
- `PredictionComposer` produces `PredictionPanel`. It may feed only a
  Transformer's semantic input; that Transformer preserves the type. Prediction
  never feeds a Composer or an auxiliary Panel parameter.

Known implementation discrepancy: the current `graph.py` type propagation and
`test_predictions_can_share_multi_input_operators` permit all-prediction Composer
peers. The preceding restriction retains the original agent governance contract
pending a separately authorized contract decision. Workflow maintenance does
not remove that existing API or silently rewrite either behavior or instruction.

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
  from evaluation prices, returns, governance, storage and publication.

## Catalog maintenance

After public Transformer/Composer export or documentation-metadata changes,
run `uv run python scripts/generate_operator_reference.py`. Stable functional
categories and executable `OperationExample` fixtures are public contracts:
each operation page shows actual input, Panel-parameter and output tables from
the fixture. Coordinate Workbench governed catalog changes independently.
