# 架构与设计

BagelQuant Core 将具体面板数据和惰性研究逻辑分开。

```text
Panel 输入
    |
    v
Transformer 与 Composer
    |
    v
Graph 逻辑链
    |
    v
内部执行运行时
    |
    v
缓存后的 Panel 输出
```

## 设计哲学

- 数据对象要明确：所有输入都先成为带 `Domain` 的 `Panel`。
- 研究逻辑要可组合：因子、预测和权重都用图描述，而不是立即计算。
- 执行要可复现：依赖、参数、名称和输出缓存都由图节点统一管理。
- 包边界要清晰：数据获取、回测和 UI 不属于 core。

## 结构

- `Domain` 负责交易日和资产成员关系。
- `Panel` 和 `CategoryPanel` 负责不可变的 `(time, asset_id)` long-form 数据。
- `Graph` 负责惰性逻辑链和用户侧执行入口。
- Transformer 负责一元变换。
- Composer 负责多输入组合。
- 执行运行时负责依赖求值、缓存和动态资产池掩码。

## 执行路径

调用操作函数时不会立即计算，而是创建内部节点。调用 `Graph.compute()` 后，运行时递归计算依赖，检查多输入 domain 是否兼容，执行 Polars frame 级别的函数，并把结果包装回 `Panel`。

当前调度是顺序执行。持久化缓存、增量失效和并行调度属于后续扩展方向。

