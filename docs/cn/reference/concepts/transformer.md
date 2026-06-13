# Transformer

Transformer 是一元函数式操作：

```text
Panel | Graph -> Graph
```

它接收一个 `Panel` 或 `Graph`，返回新的惰性 `Graph`。

## 内置类别

内置 transformer 覆盖常见因子处理步骤：

- 基础元素级操作，例如 `identity`、`abs_value`、`negate`
- 缺失值处理，例如 `fillna`、`ffill`、`bfill`
- 滚动窗口，例如 `rolling_mean`、`rolling_std`、`rolling_zscore`
- 标准化和排序，例如 `rank`、`rankpct`、`zscore`、`winsorize`
- 极值和变换，例如 `truncate`、`trim`、`boxcox`、`fisher`
- 类别面板操作，例如 `category_demean`、`category_rank`

完整逐项参考见英文生成目录：[Transformer reference](../../../en/reference/transformers/index.md)。

## 自定义 Transformer

项目可以用 `@transformer` 装饰器把自定义 Polars 逻辑包装成 BagelQuant 操作：

```python
import polars as pl

from bagelquant_core.transformer import transformer


@transformer
def demean(frame):
    means = frame.group_by("time").agg(pl.col("value").mean().alias("mean"))
    return frame.join(means, on="time").with_columns(
        (pl.col("value") - pl.col("mean")).alias("value")
    ).select("time", "asset_id", "value")
```
