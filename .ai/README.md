# Core AI workflow and rule routes

Always read [development rules](rules/development.md). Then load the rows that
match the task; rules are authoritative instructions, linked docs explain use.

| Task topic | Owner rules | Relevant docs |
| --- | --- | --- |
| Node/Domain, neutral input conversion, membership, Operator, typed Prediction, catalog | [Nodes and operators](rules/nodes-operators.md) | [Concepts](../docs/en/concepts.md), [Operator](../docs/en/reference/concepts/operator.md) |
| Logical graph, numerical artifact/cache/checkpoint persistence, canonical identity, materialization, causal history, execution/resources | [Graph and runtime](rules/graph-runtime.md) | [Architecture](../docs/en/architecture.md), [Execution](../docs/en/reference/concepts/execution.md), [Performance](../docs/en/performance.md) |
| Generic ML, mature labels, deterministic training, IC weighting, OLS, optimizers | [Numerical and training contracts](rules/numerical-training.md) | [Public API](../docs/en/reference/public-api.md), [Internals](../docs/en/reference/internals.md) |
| Public operation/export/doc metadata change | [Nodes and operators](rules/nodes-operators.md), [development](rules/development.md) | [Operation reference](../docs/en/reference/index.md) |

Core is independently versioned and imports no Data, BT or Workbench. Stage 3
implements typed Nodes/Operators, safe DSL, global graph/lifecycle and CoreStore
SQLite/Parquet caches, checkpoints, integrity and recovery. Callers provide paths
and execution contexts; Workbench owns machine policy and global admission.

Data provides neutral frames/schema/availability/immutable identity evidence;
Core owns generic Domain/Node conversion without querying Data. Data owns input
freeze authority; Core owns numerical artifacts; BT owns account/evaluation
artifacts. Workbench owns China semantics, app metadata, governance and task
orchestration while referencing backend artifacts through public APIs.
BT and Workbench are downstream consumers; read their owner rules only when
changing a shared contract. Core docs are
collected by the website from GitHub default branches, independently of workspace
gitlinks; change documentation here, not in generated website content.

Integration discovery and standalone fallback are in [AGENTS.md](../AGENTS.md).
After verifying an integration root, read its
`.ai/rules/contracts/package-boundaries.md` for cross-package work and use its
`.ai/workflow.md` and root task CLI;
the bilingual usage guide is `docs/ai-workflow.md` / `docs/zh-CN/ai-workflow.md`
in that verified root. Rules/templates are versioned; actual task records are
local, ignored, and owned only by the workspace root. A clone does not restore them.
