# Graph

`Graph` 表示一条惰性的研究逻辑链。

```python
price = Panel.from_domain(price_df, domain, name="price")
signal = rank(zscore(price), name="signal")
```

这里原始输入是 `Panel`，派生出来的 `signal` 是 `Graph`。

## 职责

Graph 负责收集依赖、校验 DAG、生成拓扑顺序、保存图规格、委托运行时执行，并提供物化后的输出访问。

校验会拒绝环、重复节点名、非法父节点类型，以及父节点数量不匹配的操作节点。

## 全局逻辑定义

`Graph.spec()` 是带局部符号名称的编写模板；`Graph.logical_spec()` 或
`canonicalize_graph(template, input_keys=...)` 将其转换为唯一持久化格式
`logical_dag.v1`。稳定输入来源键、数值类型、注册算子、已补齐默认值的标量参数、
有序主输入和具名辅助输入共同决定节点 ID。名称、研究对象 metadata、数据内容、
Domain、Snapshot 与算子实现版本不进入逻辑 ID。

`LogicalGraphSpec.union()` 合并不同根的共同节点。`Graph.from_logical_spec()`
按来源键显式绑定 Panel；多个输出名称可以引用同一个根。序列化读取会校验节点内容
哈希和依赖拓扑。仅用于展示的 Raw 来源边由应用保留，不进入 Core 数值输入边界。

参数不可变；修改参数会形成新节点。`parameter_bindings` 可在执行时绑定
`$data_start` 等显式占位符，原始逻辑定义及节点 ID 保持不变。

## 输出

执行前访问 `Graph.output` 会报错。调用 `compute()` 后，`output` 返回对应的 `Panel`。

如果执行的是下游图，中间图节点的输出也会被填充，方便调试和复用。

## 多输出图

`Graph(outputs=[...])` 可以一次执行多个输出节点。多输出图的 `output` 是从输出名到 panel 的映射。
