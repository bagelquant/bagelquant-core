# Panel

`Panel` 是 BagelQuant 中显式的数据对象。每个输入 panel 都通过 `Domain` 创建，并继承该研究域的交易日历和资产空间。

## 角色

Panel 可以表示：

- 原始输入数据
- DAG 的叶子节点
- 图执行后的物化输出
- 内部执行运行时缓存的中间结果

因子、预测和组合权重通常先表示为惰性 `Graph`，只有在需要输出时才被计算成 `Panel`。

## 不变量

Panel 要求唯一的一维索引和列名，只包含数值，构造时复制输入数据，通过 `Panel.data` 访问时返回防御性副本，并从公开 API 角度保持不可变。

Panel 必须匹配 `Domain` 的交易日历和资产列。对于动态资产池，非活跃单元会被屏蔽。

## 动态资产池

动态资产池是一个稀疏的 long-form 布尔表，以 `time` 和 `asset_id` 为键，并使用 `active` 表示该资产在该时间是否属于研究域。缺失的 `(time, asset_id)` 行视为非活跃，成员关系不会自动向前填充。

```python
membership = pl.DataFrame(
    {
        "time": ["2024-01-02", "2024-01-03"],
        "asset_id": ["AAPL", "MSFT"],
        "active": [True, True],
    }
)
domain = Domain(
    calendar=["2024-01-02", "2024-01-03"],
    universe=membership,
)
```

## CategoryPanel

`CategoryPanel` 用于行业、板块、国家等标签数据。它与 `Panel` 具有相同的时间乘资产形状，但接受字符串标签，可配合 `bagelquant_core.transformer` 中的 category 操作使用。
