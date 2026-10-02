# 架构与设计

## 统一 Node 与 Operator

Core 的计算抽象只有节点和算子。Panel、CategoryPanel、PredictionPanel、Domain、
图规范、类型信息和运行器是基础设施。算子注册到统一 OPERATOR_REGISTRY；Transformer、
Composer 和 Prediction Composer 共用 OperationNode、依赖处理和执行体系。

```text
Domain / Panel 输入 → Operator / OperationNode DAG → 运行器 → Panel / 权重值
```

Node.dag() 在计算前返回完整 JSON-safe 图，Node.mermaid() 导出依赖图。主输入、命名辅助
Panel 和状态依赖都参与序列化、校验和拓扑。算子拥有显式版本、参数、类型、因果性、密度
与文档契约。Transformer / Composer 保留原有名称和数学行为，是目录分类。

## Panel 与 Domain

不可变 Panel 按 time、asset_id 索引。Domain 提供交易日与动态成员关系；源保持稀疏，
只有显式 dense 边界才对齐。输出返回副本，跨截面与时间序列计算保留缺失与因果性。

## 图与执行

Graph 收集依赖、检查循环、导出规范并委托运行器。Graph.compile(spec) 校验一次后可
绑定多个批次。共享子图只执行一次，纯 Polars 算子融合为惰性计划，NumPy、回归和优化
是显式 eager barrier。缓存身份包括输入、Domain 与节点参数。

## 权重与训练算子

top_n、equal_weight、regularized_weights、exposure_constrained_weights 和 rebalance
属于 Core。优化器只参考历史计算目标，不读取账户。rebalance 保存完整目标及 hold /
unavailable / rebalance 状态，锚定 Data Start 首个交易日；零表示退出。
滚动 ElasticNet、LightGBM、成熟标签窗口、固定采样与训练审计属于通用训练实现。
标签数据、Universe、市场规则、训练生命周期与持久化由下游显式提供。

## 资源与包边界

ResourceLimits 控制线程、并发、批量、缓存与内存压力。硬件参数不改变研究身份；
采样、模型和求解精度是算子参数。Core 不依赖 Data、BT 或 Workbench，也不拥有
Provider、账户、市场规则或 evaluation 持久化。账户模拟、收益指标与图表由 BT 负责。
