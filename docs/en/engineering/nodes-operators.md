# Domain, Node and Operator contracts

- Domain owns neutral calendars and dynamic membership; China selection belongs downstream.
- Node stores sparse `(time, asset_id, value)` Polars data and one explicit ValueType.
  Derived Nodes may be unmaterialized. Use stable source_key independently of display names,
  and immutable evidence for persisted lazy sources.
- One Operator registry and `operator` decorator; modules are mathematical categories.
  Operator calls accept/return Nodes; scalar parameters and named auxiliary roles are distinct.
- Numeric and prediction peers never mix. Declared prediction and weights producers preserve
  those boundaries. Validate category, risk, label, availability, calendar and reference roles.
- Python/DSL/bound saved specs/cache reads use the common type/dependency contract.
  Unbound authoring placeholders defer source types until binding.
- Preserve coordinates, sparse membership, missingness and availability through optimization.
  Only pure identity and validated binary exchange normalize; execute the normalized definition.
- The safe AST DSL, versioned operator contract and generated catalog belong to Core.
  Regenerate catalogs after exports/contract changes; never edit generated pages.
- Tests cover zero/NaN/empty/sparse/type/trace equivalence and typed downstream boundaries.

Rolling windows require positive integers; `min_periods` is either `None` or an
integer from zero through the window size. Eager and lazy execution reject
booleans, fractional numbers and strings instead of coercing their values.

CURRENT_AND_SHIFT_MAX requires current trace support. Missing current support
produces a null trace, even if the shifted coordinate is known. An explicitly
known null value keeps its current trace; paired operations retain session-based
numerics and do not bridge missing observations. Runtime kernel v2 separates
new materializations from older trace evidence; historical receipts remain.

Cross-sectional transforms carry the maximum of every declared dependency-date
channel over the consumed daily population, including selection and equal-weight
normalization. Grouped transforms scope value dates to `(time, group)` and include
the daily classification evidence that establishes both inclusion and exclusion.
Composite Kelly transforms propagate cross-sectional dates through their rolling
window. Stateful target optimizers retain cumulative population dependency dates;
their checkpoint carries a compact summary across continuation rather than
loading historical values. These trace scopes bind materialization implementation
identity. Original receipts remain readable and never become corrected-policy hits.
Shared Date/Datetime channels use a schema-derived common type that preserves the
finest declared time unit; differing time zones require explicit caller alignment.
Rolling traces retain that type, and cumulative checkpoints store exact physical
temporal integers so nanoseconds survive persistence and continuation.
Explicit final-session cutoff repair admits these zero-history daily/grouped
scopes over the complete selected population; rolling and cumulative scopes
retain their existing separate evidence requirements.

`mask(source, mask_frame=selector)` replaces excluded values with typed nulls.
Its `replace_value=None` default is portable through Python, DSL and saved logical
graphs; finite numeric replacements remain supported. Categorical masking retains
the input value type and a null default rather than manufacturing a string NaN.

Eager and consumed lazy sources reject malformed dates and null asset keys.
Lazy membership selection and dense alignment retain malformed keys until typed
validation. Runtime validates already scoped source batches before joins or
reductions can remove invalid keys. It neither collects complete historical inputs
at construction nor adds a value scan to metadata-only no-op planning. Valid
canonical Date predicates continue to reach bounded persisted scans.
Already collected unvalidated eager inputs also undergo uniqueness validation
before a custom reduction can erase duplicates or normalization collisions.
