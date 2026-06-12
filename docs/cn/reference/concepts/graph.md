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

## 输出

执行前访问 `Graph.output` 会报错。调用 `compute()` 后，`output` 返回对应的 `Panel`。

如果执行的是下游图，中间图节点的输出也会被填充，方便调试和复用。

## 多输出图

`Graph(outputs=[...])` 可以一次执行多个输出节点。多输出图的 `output` 是从输出名到 panel 的映射。
