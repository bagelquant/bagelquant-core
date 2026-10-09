# Core：类型化计算与持久图

Core 0.11 是可独立运行的 Python 3.13 package，不依赖 Data、BT 或 Workbench。
调用方传入输入证据、Domain、SQLite 路径、产物目录、观察区间、信息 cutoff 与资源参数；
Workbench 负责机器探测、全局调度、中国市场语义和 GUI。

## 对象与接口

`Domain` 定义日历、资产和动态 membership。`Node` 统一承载稀疏长表
`(time, asset_id, value)`，支持 Polars DataFrame/LazyFrame；`value_type` 分为
numeric、category、prediction、weights。派生节点未计算时不可读取值。
`Operator` 接受 Node 与命名辅助 Node，标量配置单独传入，返回 Node。
所有算子集中在 `bagelquant_core.operator`，按数学功能组织。
预测和普通数值不能作为同等输入混合；辅助角色独立校验并进入图。

`Graph.from_dsl` 安全解析赋值、调用和表达式；`add_dsl` 原子合并且不计算。
`merge`、`resolve_local`、`upstream`、`downstream` 包含完整辅助依赖。
请显式设置跨会话稳定的 source_key；显示名称不参与逻辑身份。持久执行的 LazyFrame
必须携带不可变输入 identity，值或可用性证据变化必须更新 identity。
逻辑身份排除 Domain、输入版本、名称与硬件；物化身份包含这些数值上下文。
仅消除纯恒等操作及规范化已验证的二元交换规则，规范定义也是实际执行定义。

## 存储、增量与发布

`CoreStore(meta_path, artifact_path)` 提供 SQLite＋Parquet，持有图、节点状态、
数值、Domain、类型、availability traces、checkpoint、训练审计和附属产物。
缓存区分精确命中、部分覆盖和缺失。精确命中不读取数值源、不执行算子。
有限历史使用调用方锚点固定的 32-session 数学块，验证值、坐标、membership 与 traces
后复用。状态节点只在全部父节点历史前缀证据一致时恢复 checkpoint；否则完整重放。
物理分批与 workers 不改变数值契约或身份；读取窗口不改变冻结信息 cutoff。

全局更新冻结图与状态版本，要求所有登记 Domain 的全部活跃派生节点成功。
调用方从 `plan.ready(context)` 选取任务并执行 `plan.execute`，完成后 `plan.publish`。
分支更新提供 roots，不推进整体版本。失败、取消和冲突保留可重试的不可变缓存。
`sleep` 级联全部下游；`wake` 只恢复显式选择且上游活跃的节点，批量操作原子验证。
休眠限制覆盖构图、准入、执行和发布，缓存不能绕过；历史证据仍可读取。

应用通过 `describe`、`read_values`、`evidence`、`inventory`、`check_integrity` 和
`cleanup_plan/apply_cleanup` 管理 Core 凭据，不直接读写后端数据库或文件。
清理仅回收已验证的无引用完整 generation；已提交历史证据保留。
Core 先提交，应用随后使用 `reference_publication` 状态保护，在自己的事务中绑定凭据；
中断后按稳定 request_id 找回并幂等绑定，不模拟跨库事务。

新版存储拒绝旧格式，没有兼容读写或迁移。真实数据库和服务切换属于第 6 步。
英文[完整 API 用例](../en/architecture.md)与[算子目录](../en/reference/operators/index.md)
提供可执行示例。验证使用临时数据根和假输入；可选 LightGBM 运行还需对应原生库。

`diff`、`pct_change` 等成对时序运算只有当前坐标追踪有效时才合并当前与移位证据。
缺少当前输入不能继承前一坐标的可用日期；已知空值仍保留当前追踪，包括首个
未定义变化。`logical_runtime.v2` 执行身份阻止复用旧追踪产物，不改写逻辑身份或
历史回执。
