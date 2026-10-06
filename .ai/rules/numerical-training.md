# Numerical prediction, training and target contracts

## Causality and training

- Explicitly distinguish observation, publication/information cutoff,
  signal-effective and execution dates. Never replace PIT joins with latest
  values or admit information unavailable at decision time.
- Keep `machine_learning.py` deterministic and application-neutral: matrix
  validation, folds, reusable fit/predict are allowed; production selection,
  BT imports, price retrieval, scheduling/evaluation policy and lifecycle are not.
- Supervised primitives receive generic targets, label ends and availability
  inputs. Training operator v2 selects complete mature rows with date-balanced
  seeded stable keys before gathering wide matrices. Both estimators default
  to seed 1729. Actual training-key hashes and bounded fit audits are explicit
  numerical evidence, preserved by checkpoints.
- Respect exact requested dates and both label end and availability gates;
  sparse embargo/decay weights must match direct eligible-row references.
  Rescale exponential moments from each admitted batch, never a future frame end.
  Current consumers supply multi-horizon lifecycle/kernel identities and Raw
  timing evidence. In the target, Workbench owns research lifecycle/bindings,
  Data owns source timing evidence and Core owns numerical artifact identities;
  Workbench references backend receipts. Generic evidence/identity mechanisms
  belong to Data/Core; Workbench supplies source declarations and scheduling.

## IC and OLS

- Core owns the single quantile-rank-IC formula and public
  `QuantileICWeightedPredictionOperator`. Form quantiles from every finite Alpha
  before target missingness. `window` and `quantiles` enter graph/cache identity.
  Rolling weights require a complete contiguous window, clip `mean_ic` at zero,
  and renormalize per asset over finite Alphas with positive weights.
- `ICWeightedDecayPredictionOperator` applies `2 ** (-age / half_life)` within
  the complete contiguous Spearman IC window before the same clipping and
  renormalization. `window` and `half_life` enter identity; preserve ordinary
  `ICWeightedPredictionOperator` arithmetic-mean behavior.
- `fama_macbeth_ols_prediction` is the single numerical OLS entry point used
  by `OLSPredictionOperator` and Workbench diagnostics. Preserve intercept,
  complete-case/full-rank, contiguous-window, availability and missing-value
  behavior. Results include prediction, factor returns/cross-sectional standard
  errors, rolling premia and period diagnostics; do not duplicate formulas.

## Computed targets

- Core optimizers reference historical computed targets, never account positions.
  BT owns generic account execution, fills, cash, costs and corporate-action
  mechanics with explicit inputs. Workbench owns China semantics, including
  market-specific lots, T+1 and limits, supplied as policy inputs to BT.
- `rebalance(weights, every=5, anchor="data_start")` anchors at the first trading
  session on/after Data Start. Save complete target snapshots including zero
  exits, explicit hold decisions, and unavailable reasons for inadequate warm-up
  or selection. Never interpret NaN as an order; rank ties use stable `asset_id`.
- Regularized optimizer v3 refines its scalar dual and preserves exact zero
  support in numerical projection so roundoff cannot open whole-lot positions.
  Consumers include that version in optimizer decision/checkpoint identities
  without invalidating equal-weight or Prediction artifacts.
