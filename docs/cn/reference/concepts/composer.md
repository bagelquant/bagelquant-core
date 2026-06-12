# Composer

Composer 是多输入函数式操作：

```text
(Panel | Graph, ...) -> Graph
```

它用于把多个输入组合成一个惰性图节点。

## 常见用途

Composer 覆盖的操作包括：

- 算术组合，例如 `add`、`subtract`、`multiply`、`divide`
- 多输入聚合，例如 `mean`、`sum_frames`、`weighted_sum`
- 逻辑或条件组合，例如 `mask`、`coalesce`
- 分组截面处理，例如 `group_mean`、`group_zscore`
- 滚动关系，例如 `rolling_corr`、`rolling_cov`、`rolling_ols`

完整逐项参考见英文生成目录：[Composer reference](../../../en/reference/composers/index.md)。

## 对齐约束

多输入 composer 要求输入具有等价的 `Domain`。执行时会在派生计算后重新应用动态成分约束，确保非活跃资产不会影响后续操作。
