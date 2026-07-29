# 从 0.1.x 迁移到 0.2.0

- `Panel.from_domain` 支持 `DataFrame | LazyFrame`，构造时不再稠密化。
- `Panel.collect(dense=False)` 返回稀疏结果；`.data` 保留为稠密兼容接口。
- 重复输入批次使用 `Graph.compile(spec)` 和 `CompiledGraph.compute(...)`。
- 自定义裸 decorator 默认进入稠密 eager barrier；处理 trace columns 时必须声明 trace rule。
- 显式 identity 表示调用方保证输入不可变且内容等价；无法保证时不要传入，Core 会使用安全的实例 token。
- 0.2 不新增磁盘缓存；持久化、失效和并发写仍由上层数据系统负责。
