# Causal version inputs

Core accepts neutral scalar versions through `CausalInputWindow(frame,
source_identity)`. Required fields are `time: Date`, `asset_id: String`,
`value: Float64` (or String for category), `known_at: Date`, and
`revision: UInt64`. Declared trace columns remain typed channels. A source
normalizes its exact version-selection ordering; Core does not infer provider
availability or manufacture first-known dates. Null values remain versions.
Public `CAUSAL_EXECUTION_POLICY` versions the causal selection/execution contract.
Hosts bind it together with operator implementations in metadata completion and
cache identities, including shortcuts that run before constructing a Domain.

Pass `input_definition_identities={context: {input_key: stable_definition_hash}}`
to distinguish source numerical definitions from changing source content. When
provided, it must cover every admitted context and every selected input key
with a nonempty string. These identities describe the producer's numerical
definition or implementation, excluding versions, enclosing receipts, cutoffs,
display text and resource settings. Each causal node binds only its actual
input ancestors in its full materialization, completion and block policy.
A relevant definition change permits initialization of a new authority while
retaining original receipts. An unchanged definition with entirely unknown
historical content proofs still blocks before numerical work; do not rotate
definition identities to bypass that gate. Policy version 2 binds this boundary.

Use the existing `Graph.plan_update` with `causal_inputs` and
`input_version_loader(context, key, first, last, frozen_cutoff)` plus the scoped
metadata proof callback `(context, key, first, last, consumption_cutoff)`.
An explicit consumption cutoff requires this scoped callback; an unscoped
identity cannot certify which input vintage was consumed. Scoped-only callers
can provide this callback without a separate unscoped proof callback.
The loader may retain versions known later than a particular block under the
original frozen ceiling. Core selects only the versions visible at each
effective output segment. Ordinary appends do not require per-decision queries.

An optional `causal_output_domains` mapping supplies each context's published
causal coordinate union. Source placeholder Domains cover observation inputs,
including older keys only discovered later. Those keys may be needed in a
later rolling halo without becoming historical output members. Source coverage
can use a compact rectangular calendar/asset declaration; output membership
may be dynamic. Old output interval identity excludes future-only membership.

The same CoreStore owns canonical 32-session blocks, original result references,
node completion rows and publication fences. Shared dependency evaluation loads
each needed source range once, retains bounded version buffers, and densifies
temporary rolling windows on admitted calendar sessions. MAD population z-score,
full-rank cross-sectional OLS residuals and positive-weighted means are generic
Core operators. Mathematical contracts must declare finite backward history;
unknown completed history cannot silently fall back to a cold computation.

Numerical `log1p` version 2 uses the stable Polars `log1p` implementation,
preserving tiny returns that would be lost by evaluating `log(1+x)` directly.
Implementation versions participate in artifact identity.
