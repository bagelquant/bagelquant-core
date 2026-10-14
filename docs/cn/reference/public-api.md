# Public API

Use `Domain`, `Node`, `ValueType`, `Operator`, `Graph`, `CoreStore`,
`ExecutionRuntime`, `MaterializationKey` and `NodeMaterialization` from
`bagelquant_core`. Registered numerical operations live in
`bagelquant_core.operator`.

The [architecture guide](../../en/architecture.md) documents the persistent API and
type contracts; the [catalog](../../en/reference/operators/index.md) documents each operator.

- `ArtifactVerification` 提供绑定 receipt 与文件身份的有界校验证明，由 `CoreStore.read_context()` 管理生命周期。
- `CoreStore.describe_update(identity)`、`describe_evidence(identity)` 只检查已提交元数据。
- `CoreStore.read_values(identity, start=..., end=..., include_traces=...)` 在构造 Domain 前筛选请求日期的日历和成员坐标，读取相应月分区并返回类型化稀疏值，不重新计算字节哈希。空窗口保留类型与 trace；无范围读取仍检查完整 Domain。`verify_update`、`check_integrity` 保持全量审计。
- `CoreStore.publish(..., value_references=[{"identity": ..., "start": ..., "end": ...}, ...])` 用原类型化区间与修复 materialization 拼接完整结果，必须严格有序、无重叠地覆盖全部 calendar，并验证原 membership/type/traces。每个保留 Parquet 分区带选择边界，读取时推入谓词；发布验证原 authority 与文件存在，不解码或重写历史。
- `CoreStore.describe_artifact(identity, name)` 描述与 evidence 相同的递归原 channel；`artifact_interval_identity(identity, name, start=..., end=...)` 基于原内容绑定索引证明全部日期列值，含 decision status/reason 与空日期 calendar。缺少可选证据返回 None，不读取数值。
- `CoreStore.read_artifact(identity, name, start=..., end=...)` 在打开 Parquet 前剪枝不相交 concat leaf，并下推日期谓词。`artifact_index_plan(identity, name)` / `build_artifact_index(plan)` 显式审计并索引旧原 artifact，保留 receipt/文件。缺失旧索引需此显式维护，不隐式读取历史；后续组合复用原索引。
