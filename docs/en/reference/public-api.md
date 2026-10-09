# Public API

Use `Domain`, `Node`, `ValueType`, `Operator`, `Graph`, `CoreStore`,
`ExecutionRuntime`, `MaterializationKey` and `NodeMaterialization` from
`bagelquant_core`. Registered numerical operations live in
`bagelquant_core.operator`.

The [architecture guide](../architecture.md) documents persistent execution and
types; the [catalog](operators/index.md) documents each operator.

- `ArtifactVerification` provides finite, bounded checksum proofs keyed by receipt and file identity. `CoreStore.read_context()` owns its lifetime.
- `CoreStore.describe_update(identity)` and `describe_evidence(identity)` inspect committed metadata without loading numerical artifacts.
- `CoreStore.read_values(identity, start=..., end=..., include_traces=...)` verifies only required Domain and selected monthly files, returning typed sparse values. `verify_update` and `check_integrity` remain full audits.
