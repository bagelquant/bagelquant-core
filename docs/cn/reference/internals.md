# Implementation guide

See the implemented [architecture](../architecture.md).

`node.py` and `domain.py` hold typed values and coordinates; `operator/` is the
single operator registry and numerical module tree. `logical.py`, `graph.py` and
`dsl.py` share validation and normalized definitions. `updates.py` plans ready
nodes; `incremental.py` implements canonical finite blocks and verified
checkpoint replay. `store.py` owns publication, integrity, lifecycle and receipt
recovery. Private modules are not application integration APIs.

## 只读存储检查

`CoreStore(meta_path, artifact_path).inspect()` 返回 `uninitialized`、`ready`
或 `incompatible`，并提供可用的 schema 版本及原因。检查使用只读元数据快照，
验证版本、必需表及必需列；版本正确但列不完整的 schema 仍不可用。
公共 `bagelquant_core.inspection.open_metadata_snapshot` 包含 WAL 已提交内容，
复制前后原文件签名必须一致，持续变化的存储重试后报告不可读。SQLite 的 WAL
索引只在私有系统临时目录创建，检查后删除；不创建或修改原存储目录、数据库、
旁路文件及恢复状态。活动或 hot rollback journal 会报告不可读，不读取未提交
页面，也不触发恢复；零头失效的 PERSIST journal 可以检查。空存储只能通过
显式 `initialize()` 创建。
