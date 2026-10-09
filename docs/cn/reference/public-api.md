# Public API

Use `Domain`, `Node`, `ValueType`, `Operator`, `Graph`, `CoreStore`,
`ExecutionRuntime`, `MaterializationKey` and `NodeMaterialization` from
`bagelquant_core`. Registered numerical operations live in
`bagelquant_core.operator`.

The [architecture guide](../../en/architecture.md) documents the persistent API and
type contracts; the [catalog](../../en/reference/operators/index.md) documents each operator.

- `ArtifactVerification` 提供绑定 receipt 与文件身份的有界校验证明，由 `CoreStore.read_context()` 管理生命周期。
- `CoreStore.describe_update(identity)`、`describe_evidence(identity)` 只检查已提交元数据。
- `CoreStore.read_values(identity, start=..., end=..., include_traces=...)` 验证所需 Domain 与月分区并返回类型化稀疏值；`verify_update`、`check_integrity` 保持全量审计。
