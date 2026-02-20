# 文档补全清单（Completion List）

- Date: 2026-02-18
- Scope: `docs/` 全量文档（含跳转入口）

本清单记录本轮对“未补全文档”的补全与重构结果，便于验收与后续持续改进。

## 1. 已补全/已完善（由骨架页补齐为可交付内容）

| 文档 | 类型 | 补齐要点 |
|---|---|---|
| [quickstart.md](/docs/guides/quickstart.md) | Guide | 补齐目的/范围/前置条件/Local+Slurm最小链路/产物校验/完成标记/排障入口/变更记录。 |
| [cli.md](/docs/guides/cli.md) | Guide | 补齐子命令职责与输入输出、退出码、完成标记与排障入口、实现入口说明。 |
| [configuration.md](/docs/guides/configuration.md) | Guide | 补齐路径解析规则、Submission(Local/Slurm)结构、各Workflow最小配置示例与常见错误处理。 |
| [slurm.md](/docs/guides/slurm.md) | Guide | 补齐Slurm配置、日志命名与完成标记监控、运行示例、常见故障排查。 |
| [troubleshooting.md](/docs/guides/troubleshooting.md) | Guide | 补齐结构化排查顺序、环境/数据/作业/数值四类问题处理建议与变更记录。 |

## 2. 新增专题文档（从零建立可交付内容）

| 文档 | 类型 | 内容要点 |
|---|---|---|
| [integration-slurm.md](/docs/guides/testing/integration-slurm.md) | Guide | Slurm集成测试交付主页：交付物索引、链式编排原则、裁剪策略、运行方式、异常处理与变更记录。 |
| [integration-slurm-plan.md](/docs/guides/testing/integration-slurm-plan.md) | Plan/Guide | 集成测试里程碑、DoD、交付物与异常处理策略。 |
| [integration-config-templates.md](/docs/guides/testing/integration-config-templates.md) | Guide | 模板索引与约定、输入输出语义、异常处理与变更记录。 |
| [multi-datapool-artifacts.md](/docs/guides/testing/multi-datapool-artifacts.md) | Report/Guide | 生产目录I/O拆解、Workflow映射、关键观察、异常处理与变更记录。 |
| [config-schema.md](/docs/reference/config-schema.md) | Reference | 配置字段字典迁移落点（作为单一权威来源）。 |
| [validation.md](/docs/reference/validation.md) | Reference | 校验规则迁移落点（作为单一权威来源）。 |
| [maintenance.md](/docs/policy/maintenance.md) | Policy | 文档版本维护、Owner、触发规则、Review与过期治理。 |
| [quality.md](/docs/policy/quality.md) | Policy | 文档质量评分维度、元信息规范与类型验收清单。 |

## 3. 旧入口清理

本轮已删除所有 deprecated 的旧路径入口文档，避免多入口导致内容分叉。
