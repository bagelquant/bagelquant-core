# Core：类型化计算与持久图

Core 0.11 是可独立运行的 Python 3.13 package，不依赖 Data、BT 或 Workbench。
调用方传入输入证据、Domain、SQLite 路径、产物目录、观察区间、信息 cutoff 与资源参数；
Workbench 负责机器探测、全局调度、中国市场语义和 GUI。

Core 元数据库 schema 2 对每个 graph/context/逻辑节点保留一行小型成功状态。
`CoreStore.node_completions(graph_id, context_id)` 批量读取定义和上游摘要、
完成观察日期、成功检查日期及不可变结果引用。完全相同的计划直接完成，不读取数值、
调用输入证明或执行 kernel；缺失子节点仅为其需要的已完成父节点构造 lazy typed plan。
每个节点的结果持久发布后立即推进状态，中断续跑只执行剩余 frontier；完整发布仍受
覆盖范围和图修订约束，历史分支不会回退较新的状态。schema 2 仅用于新库，无旧库迁移。

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
后复用。有限路径已完成且缺少区间证明时，需要显式维护，不静默重放。
状态节点只在全部父节点完整历史前缀证据一致时恢复 checkpoint。缺少证明或有效
前置状态时明确要求维护，不能静默重放已完成历史。验证数值签名与声明的状态
结构，损坏的可选状态尝试更早的已证明 checkpoint。
物理分批与 workers 不改变数值契约或身份；读取窗口不改变冻结信息 cutoff。

`Graph.plan_update(input_window_loader=...)` 只加载缺块的直接父输入窗口，并验证
完整冻结身份、类型、traces 及实际 calendar/membership 坐标。canonical 部分块
追加只发布新增输出；pointwise 只计算新增日。内置 `pct_change` 和 `smooth` 仅读取
新增输出实际成员所需的已准入历史坐标，包括空值与资产重新进入时需要的旧观察。
canonical 证明范围和物化 key 保持不变；其余 rolling 保持原数学块 warm 状态重建，
以保证浮点算术一致。`computed_output_sessions`、`kernel_sessions`、`input_sessions`
分别记录输出、内核和输入工作，不能将单日输出称为单日内核计算。

`CoreStore.publish(value_references=...)` 复用登记数值分区，不解码、重 hash 或
重写旧值。索引证明完整有序 calendar、membership、类型/traces 与逐日恰好一次
覆盖；Domain 原文件复用在 writer transaction 中再次检查。缺索引需要显式维护。
`completed_materializations` 从成功的 scoped update receipt 判断完成，不能将
中断初始化的局部 cache 当作已完成结果。

`OperationContract.checkpoint_history` 声明续算读取 halo：EWM 只读新增日期，
权重状态包含前一 session，训练读取窗口加 label maturity；完整执行日历保留
fit anchor 与 embargo 语义。累计 trace 保存逐资产 carry；旧状态消费时可从原
输出 trace 聚合一次。续算只发布有类型的新尾部并引用原分区。同截止的新外层
凭据不读取输入或运行内核。只有实际引用的 `$` 参数影响 block/state 策略，无关
参数不失效；未知的旧策略/参数范围要求显式维护。引用发布不审计原值字节，
消费损坏的原值或显式审计仍然失败。

训练通过 checkpoint_parent_scopes 声明最后一次计划 fit 按 maturity 实际消费的
label 观察前缀及 fit 信息截止日。调用方 input_scope_proof 可证明未来经济终点
在该截止日不合格，日历扩展无需使旧 fit 失效；不改变采样窗口或 maturity。
policy 保留原始 scope/cutoff，旧证明不能被隐式收窄。之外的新成熟标签不使旧
拟合失效。位置 feature 保留完整预测
前缀，包括同一源同时充当 label 的情况；输出 calendar/membership 前缀单独
验证。带融合 trace 证明的辅助源仍保留全前缀，更细修订需独立 trace 证明及
fit 边界 checkpoint，不能宣称已完全解决所有局部修订的旧 fit 重用。

有限历史更新在内核前预检已完成 block。若 opaque proof 变化使全部已完成前缀
失效，则明确阻断，要求影响解析或显式证明维护；证明格式变化不能作为数值
定义变化，也不能触发历史重算。

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

`CoreStore.inspect()` 只报告 schema 就绪状态，不初始化、恢复或修改原始文件/sidecar；
默认离线元数据快照拒绝活动或 hot 回滚日志。在线应用可显式使用
`inspect(runtime=True)`，通过普通 SQLite WAL/SHM 协调在并发写入时读取一个已提交
事务，不复制元数据。`inspection.open_metadata_snapshot(path, runtime=True)` 向
下游 backend 提供相同通用读取原语。运行态允许协调 sidecar、禁止写入，遇到需要
恢复的遗留 hot journal 时拒绝读取。此接口不替代产物完整性校验。

已为 Date 的时间列保持恒等表达式，让有限窗口过滤下推到持久 Parquet 扫描，
避免每个 32-session 块反复解压全历史。真实 String/Datetime 转换保持不变，
membership、数值、traces 和物化身份不变；此优化不替代完整性证明。

新版存储拒绝旧格式，没有兼容读写或迁移。真实数据库和服务切换属于第 6 步。
英文[完整 API 用例](../en/architecture.md)与[算子目录](../en/reference/operators/index.md)
提供可执行示例。验证使用临时数据根和假输入；可选 LightGBM 运行还需对应原生库。

`diff`、`pct_change` 等成对时序运算只有当前坐标追踪有效时才合并当前与移位证据。
缺少当前输入不能继承前一坐标的可用日期；已知空值仍保留当前追踪，包括首个
未定义变化。`logical_runtime.v2` 执行身份阻止复用旧追踪产物，不改写逻辑身份或
历史回执。

## 发布完整性与读取

CoreStore 拥有不可变 manifest 和已提交的 graph/update receipt。`describe`、`describe_update`、`describe_evidence` 检查元数据身份，不扫描 artifact 字节；发布检查不可变身份和事务边界；显式完整性审计检查全部保留通道及派生索引。限定日期的 `read_values` 在构造 Domain 前筛选持久化日历与动态成员坐标，只读取请求日期涉及的月分区，不重新计算字节哈希，保留稀疏值、类型与 trace。仍检查标准 Domain 文件类型和无效空日期键；空窗口返回类型化空表，无范围读取仍验证完整 Domain。`evidence` 仅加载辅助证据。

`read_context` 在一次操作中共享有界、线程安全的 checksum 证明。证明绑定 receipt、预期 checksum 和文件身份；文件修改、替换或 context 结束后不能复用。它不是永久的全局有效标签。检查和 graph 元数据读取不创建缺失的 store/graph；显式初始化启用 SQLite WAL，读取使用已提交、只读事务，发布继续检查最新 revision/CAS。

## 显式维护输入证明格式

`CoreStore.plan_input_proof_rebind(spec, identity, **source_bindings)` 核对原始
canonical finite block 与冻结 Source owner 的实际消费投影，登记不可变维护计划。
虽然名称含 plan，它会写入计划元数据，只能作为显式维护，不能用于日常前置检查
或 Graph 更新。调用者提供原始参数、直接父节点 Domain、完整绑定的区间 token，
以及两侧 `ConsumedInputEvidence`：原始收据及 digest、消费内容、读区间、Domain、
value type 和全部 trace。只有 scalar 的证据不能证明有 trace 输入等价；旧输出
相等也不能代替旧输入相等。原绑定未知时必须由 Source owner 补充真实证明。

`Node.source_interval_identity(source_token, start, end)` 使用 Graph 相同的
Core Domain/type/trace 绑定方式；边界必须是 `datetime.date`。Source owner
证明实际消费的通道，Core 只读取真实坐标与元数据，不读数值或发布索引。
Graph 的操作内 memo 区分实际不可变 Domain 对象，不能只凭调用者给出的相同
opaque Domain identity 共用证明。

多个已登记收据共用 Domain 身份时，引用发布在准备与最终写事务中采用同一
选择规则，优先使用绑定原内容的有效索引，仍核对实际有序坐标。缺少证明须
显式维护，选中证明损坏则失败；不会为发布自动读取旧 Domain 文件建索引。

`apply_input_proof_rebind(plan)` 在缓存命中前和 publication 事务内验证已登记
的完整计划，只引用原数值分区，不运行历史内核或重 hash 原文件。保留原收据、
数值 context、参数及执行策略，只替换已证实的输入身份，并可补充已核对的相关
参数声明。读取与显式审计递归核对整条原收据和维护计划链；缺失、损坏时阻止
复用，不自动修复。登记前取消不产生正面证明；登记后取消可保留计划元数据，
但不发布数值结果或推进图指针。


Rebalance 的检查点续算使用具名 calendar Node 的完整 Domain 日期轴，包含成员为空的交易日。
恢复状态前必须证明原锚点和全部已完成输入前缀一致，续算只读取、计算新交易日。
CoreStore.read_checkpoint 只读取状态通道；CoreStore.compose_artifact 引用原有日期唯一的
决策产物并追加同类型、不重叠的尾部，保留历史文件。缺少检查点、日历或日期产物元数据
的旧完成记录，需要先执行显式的所属后端证据维护。

### 外部输入证明窗口

五个监督 `prediction:*` 工厂保存原始每期 IC，或共用 OLS 函数所得系数、协方差及诊断，
以及样本身份、availability、未成熟状态和包含空成员日期的完整 Domain 期间轴。
`OperationContract.checkpoint_periods` 依据逐期源证明选择新增、修订和刚成熟的样本；
同一 Runtime 先更新统计，再仅计算依赖这些变化的预测日期，支持任意长标签延迟。
历史拟合和未变值文件继续复用，已完成统计缓存可在预测或发布中断后继续使用。
缺少原统计证据的旧结果必须从已保存 IC/回归记录显式维护，不能用预测输出推造，
也不能静默重做历史拟合。缺证明或影响全部历史的变化仍产生明确阻断。

`UpdatePlan.input_proof_windows(context_id)` 返回 typed Polars 元数据表：
`context_id`、`input_key`、`consumer`、Date `start`/`end`、可空 Date
`consumption_cutoff`、`strategy` 和 `status`。空 consumption cutoff 表示不覆盖调用方
实际 frozen global information cutoff。它复用本计划实际绑定的 Domain
和已支持的 finite policy，用执行路径相同的 32-session canonical block、实际
membership observation halo 和原 partial block 端点；named auxiliary 外部父节点
同样纳入。Checkpoint 复用执行路径的配置、role、trace 限制与 relevant parameter
候选选择，并保留原 `parent_ends`/`parent_cutoffs`；legacy 缺字段表示原完整 prefix，
不能以今天更窄的标签消费合同替换。

`required` 是当前证明请求；`candidate` 是条件性的原 prefix 前置请求，最终仍须
执行时证明输入和 checkpoint 有效；`empty` prefix 不需标量 token；`unknown` 对应
通用或不支持的增量策略，日期为空，不能承诺有限重放。Exact materialization hit
没有输入证明请求。此表只描述 direct external parents；完成的 operator parents
由 Core 自己的收据证明。独立期间检查点返回 `strategy="checkpoint_periods"`，
为每个源期间（含空日）列出 required 的逐日 Date 区间，不覆盖实际信息 cutoff。

该 API 只读元数据和实际坐标，不调用 proof callback、不加载值、不执行 kernel、
不审计原 bytes、不 merge graph、不发布记录或补索引。调用方可在明确维护流程中
按这些精确 scope 准备 source-owner 证明，保留真实 information cutoff 与坐标。
Candidate 是保守集合，不代表每个候选都会读取或复用；并发新增 inventory 带来的
缺失 scope 仍显式阻断。月份 hash 不能替代不同 block/halo/Domain scope，也不能拼成
新的 canonical hash。窗口计划不是实际验收或真实维护授权。
