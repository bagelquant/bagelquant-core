# Public API

Use `Domain`, `Node`, `ValueType`, `Operator`, `Graph`, `CoreStore`,
`ExecutionRuntime`, `MaterializationKey` and `NodeMaterialization` from
`bagelquant_core`. Registered numerical operations live in
`bagelquant_core.operator`.

The [architecture guide](../architecture.md) documents persistent execution and
types; the [catalog](operators/index.md) documents each operator.

- `ArtifactVerification` provides finite, bounded checksum proofs keyed by receipt and file identity. `CoreStore.read_context()` owns its lifetime.
- `CoreStore.describe_update(identity)` and `describe_evidence(identity)` inspect committed metadata without loading numerical artifacts.
- `CoreStore.read_values(identity, start=..., end=..., include_traces=...)` reads selected calendar/membership observations and monthly values, returning typed sparse values without byte rehashing. Empty windows preserve types/traces; unbounded reads consume the complete Domain. `verify_update` and `check_integrity` remain full audits.
- `CoreStore.publish(..., value_references=[{"identity": ..., "start": ..., "end": ...}, ...])` composes original typed intervals and new repair materializations. References must cover the complete ordered calendar exactly once with identical original membership/type/traces. Each retained Parquet partition carries its selected bounds, which are pushed down on reads. Publication fences original authority and file existence without historical decoding or rewriting.
- `CoreStore.describe_artifact(identity, name)` describes the same recursively selected original channel as `evidence`; `artifact_interval_identity(identity, name, start=..., end=...)` proves all dated columns, including decision status/reason and empty calendar dates, from original-bound metadata. Missing optional proof returns `None` without reading values.
- `CoreStore.read_artifact(identity, name, start=..., end=...)` prunes original concat leaves before Parquet reads and pushes date predicates down. `artifact_index_plan(identity, name)` / `build_artifact_index(plan)` explicitly audit and index older original artifacts while preserving receipts/files. Missing legacy indexes require this explicit maintenance, never an implicit historical read. Subsequent concatenations reuse the retained original index.
