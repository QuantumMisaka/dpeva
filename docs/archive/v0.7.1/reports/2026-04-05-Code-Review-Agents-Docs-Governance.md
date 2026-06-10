---
title: Code Review - AGENTS and Docs Governance
status: archived
audience: developers
last-updated: 2026-06-10
owner: Docs Owner
---

# DP-EVA 文档治理与 AGENTS 审阅报告

> 历史治理记录：本文档记录 2026-04-05 阶段性治理闭环，不作为现行规范来源。当前规则入口以 [developer-guide.md](../../../guides/developer-guide.md)、[docs-governance-quickstart.md](../../../guides/docs-governance-quickstart.md) 与 [maintenance.md](../../../policy/maintenance.md) 为准。

## 1. 范围与方法

本轮审阅覆盖三类证据源：

- `docs/`：项目定位、开发规范、文档治理、架构说明、测试专题、归档入口
- `examples/recipes/`：各工作流对外配置契约与示例文件命名
- `src/`：CLI 入口、配置模型、路径解析、workflow/manager 主链路

交叉核查重点：

1. 项目目标意义与治理规范是否在文档中表述清晰；
2. 文档声明的工程化规范是否被核心代码与 recipes 真实落实；
3. 核心代码中是否存在稳定但未被文档充分表达的工程实践；
4. `AGENTS.md` 是否适合作为项目核心目标与规范“目录”，以及应如何瘦身重构。

## 2. 总结结论

总体结论：**项目的工程化基础是扎实的，文档治理体系也已初步成形。本轮经过第二阶段收口后，最初识别出的 active 文档入口级偏差已基本闭环；当前剩余问题主要收敛为后续优化事项，而不再阻断 `AGENTS.md` 的目录化实施。**

主题结论如下：

| 主题 | 结论 | 说明 |
|---|---|---|
| 项目目标与架构定位 | 已落实 | `docs/README.md`、`developer-guide.md`、`design-report.md` 基本能够说明 DP-EVA 的主动学习目标、模块边界与治理意图 |
| 配置与默认值统一管理 | 已落实 | `src/dpeva/config.py` 已形成 Pydantic 中心化配置模型，默认值主要由常量和字段定义统一收口 |
| CLI 到 workflow 主链路 | 已落实 | `src/dpeva/cli.py` 明确承接 `train/infer/feature/collect/analysis/label/clean` 七个正式入口 |
| 工作流解耦与共享底层模块 | 已落实 | `workflow -> manager/module` 分层已形成事实标准，特别是 collect、labeling、inference |
| 文档单一权威来源 | 已落实 | 入口级错误已修复；并已确认 `docs/source/reference` 是指向 `../reference` 的符号链接，因此 `docs/reference/*` 本身就是 Reference 正文的单一权威来源 |
| 文档归档与索引规则 | 已落实 | 入口页版本清单、过时导航与主要路径错误已修复；剩余 archive 历史内容保持只读，仅作为后续优化对象 |
| AGENTS 目录化职责 | 未落实 | 当前 `AGENTS.md` 仍偏操作说明，尚未承担项目目标、核心契约与文档入口目录的角色 |

## 3. 主要发现

### F1. Active 文档存在直接误导使用者的契约偏差

严重级别：High

问题：

- CLI 文档漏写 `clean` 正式子命令；
- collection recipe 文件名写成不存在的 `config_multi_normal.json` / `config_multi_joint.json`；
- data cleaning recipes 文档列出了当前仓库并不存在的 `config_clean_force_only.json`、`config_clean_passthrough.json`；
- `developer-guide.md` 中 `dpeva infer config_test.json`、`dpeva collect config_collect.json` 等示例与当前 recipes 不匹配；
- 训练配置示例仍使用过时字段 `project`、`submission.omp_threads`，与当前 `TrainingConfig` 不符。

影响：

- 首次使用者会按错误文件名执行命令；
- 开发者会误以为训练配置仍允许旧字段写法；
- 文档对外契约与 `src/dpeva/config.py` 已发生偏离。

本轮处理：

- 已修复 `docs/guides/cli.md`
- 已修复 `docs/guides/developer-guide.md`
- 已修复 `examples/recipes/README.md`
- 已修复 `docs/guides/testing/multi-datapool-artifacts.md`

### F2. 文档治理规则内部存在自相矛盾

严重级别：High

问题：

- `docs/README.md` 曾把 `docs/plans/` 描述为“执行期草案、临时计划”，但 `docs/plans/README.md` 明确规定该目录只收纳团队共享、可复盘的收敛计划；
- `docs/source/plans/README.md` 仍指向已归档的 parity 计划，已不再反映当前 active 状态；
- `docs/archive/README.md` 列出了不存在的路径和版本。

影响：

- 开发者无法准确判断“草案”与“项目资产”的边界；
- 归档索引失真，会误导后续治理与版本盘点；
- Sphinx 导航占位页与真实计划状态脱节。

本轮处理：

- 已修复 `docs/README.md` 中 `docs/plans/` 的职责描述
- 已重写 `docs/source/plans/README.md` 为通用导航占位页
- 已修复 `docs/archive/README.md` 的版本与目录说明

### F3. 入口页与治理页存在错链和失真路径

严重级别：Medium

问题：

- `docs/README.md` 把 logo 资产目录写成 `docs/logo_design/`，实际路径是 `docs/assets/logo_design/`；
- `docs/policy/README.md` 使用 `/policy/contributing.md`、`/CONTRIBUTING.md` 这类不符合仓库约定的路径；
- 多个 active 入口页仍以 `blob/main` 指向仓库内文档，而贡献规范要求仓库内文档优先使用相对路径。

影响：

- 导航入口自带坏链接或环境绑定；
- 文档治理规则无法做到自证一致；
- 后续搬迁或重构时维护成本升高。

本轮处理：

- 已修复 `docs/README.md`
- 已修复 `docs/policy/README.md`
- 已修复 `docs/policy/contributing.md`
- 已修复 `docs/reference/README.md`
- 已修复 `docs/architecture/README.md`
- 已修复 `docs/guides/troubleshooting.md`

### F4. 开发文档对代码结构的描述存在“概念层”和“物理层”混写

严重级别：Medium

问题：

- `developer-guide.md` 把 `examples/recipes/` 误写成 Python 调用示例目录；
- 文档目录树写了概念性的 `services/` 层，但当前仓库并不存在该物理目录；
- `tests/` 目录在文档中写成了 `test/`。

影响：

- 新贡献者在读目录树时会对真实代码布局形成错误认知；
- “工作流解耦已完成”这一事实会被错误地投射到不存在的目录上。

本轮处理：

- 已将 `examples/recipes/` 更正为 JSON 配置模板目录
- 已补充 `examples/scripts/` 的角色
- 已把 `services/` 改写为真实模块布局
- 已将测试目录修正为 `tests/`

### F5. 文档仍未完整表达一批稳定工程实践

严重级别：Medium

问题：

以下实践已在代码中稳定存在，但文档只零散出现，缺少统一、明确、低重复的表达：

- `src/dpeva/config.py` 作为配置、类型、默认值和约束的中心化入口；
- `src/dpeva/utils/config.py` 的相对路径解析机制；
- `src/dpeva/cli.py` 的统一前置校验；
- `src/dpeva/workflows/__init__.py` 的包级导出层与 `cli.py -> workflow` 执行入口边界；
- labeling 的阶段化入口；
- workflow 完成标记的实际语义边界。

影响：

- 工程规范更多依赖“读代码后自行领会”；
- `AGENTS.md` 无法从现有文档中直接抽取高密度、低重复的项目契约目录。

本轮处理：

- 已在 `cli.md` 中补充 `infer` 完成标记的使用边界
- 其余实践已纳入本报告和 AGENTS 改造方案，作为下一步规范收敛依据

### F6. `AGENTS.md` 仍未成为项目目标与规范目录

严重级别：Medium

问题：

当前 `AGENTS.md` 仍以安装、测试、构建、计划/报告归档规则为主体，更像是“轻量开发须知”。它没有高密度呈现：

- 项目目标与价值；
- 核心工程契约；
- 哪些规范在 `docs/` 中是权威来源；
- 哪些内容不应在 AGENTS 重复维护。

影响：

- `AGENTS.md` 无法承担“第一眼快速定位项目核心约束”的职责；
- 容易与 `developer-guide.md`、`policy/*`、`governance/*` 形成重复。

处理策略：

- 本轮不直接改写 `AGENTS.md`
- 单独输出一份可执行的 AGENTS 改造方案到 `docs/plans/`

## 4. 已执行的文档治理动作

本轮已完成以下文档系统治理：

- 修正 active 文档中的 CLI/recipe 文件名错误
- 修正训练配置示例与真实 `TrainingConfig` 的不一致
- 修正 `docs/plans/` 与 `.trae/documents/` 的职责边界冲突
- 修正归档入口页的版本清单与目录说明
- 修正 policy/reference/architecture 等入口页中的相对路径与内部链接
- 修正测试专题文档中的失效文件名
- 修正 `docs/source/plans/README.md` 的过时导航内容
- 确认 `docs/source/reference` 是指向 `../reference` 的符号链接，纠正了此前对“Reference 双维护”的误判，并将治理结论收敛为单源正文
- 确认 `docs/build/` 已被 `.gitignore` 覆盖，且当前未被 Git 跟踪，无需额外仓库清理
- 将 `docs/guides/developer-guide.md`、`docs/guides/configuration.md` 与 `docs/governance/traceability/feature-doc-matrix.md` 的配置入口表述收敛为稳定读者入口

## 4.1 第二阶段治理闭环状态

本轮补做的第二阶段治理结论如下：

| 项目 | 状态 | 结论 |
|---|---|---|
| `docs/reference` / `docs/source/reference` 双维护 | 已关闭 | 已确认 `docs/source/reference -> ../reference` 为符号链接，当前不存在手工双维护；Reference 正文本体以 `docs/reference/*` 为唯一来源 |
| `docs/build` 清理策略 | 已确认 | `.gitignore` 已覆盖 `build/`，且 `git ls-files docs/build` 无输出，说明当前仅为本地构建产物，不是版本库污染 |
| 配置入口的稳定读者表达 | 已收敛 | active 文档中面向读者的主入口统一收敛为 `API Reference` + `docs/reference/validation.md`，`docs/source/api/config.rst` 仅保留在维护语境中作为构建入口提示 |

结论：

- `spec`、`tasks`、`checklist` 中关于“先治理文档系统、再形成 AGENTS 改造基础”的要求现已真实闭环；
- 当前剩余事项不再属于“未完成治理”，而属于“后续优化建议”。

## 5. 代码中已存在但应被视为项目工程规范的实践

建议将以下内容视为 DP-EVA 的稳定工程契约：

### 5.1 配置与默认值统一管理

- 配置模型以 `src/dpeva/config.py` 为中心；
- 类型、默认值、约束、兼容层尽量不分散到 workflow 内部；
- 新配置优先进入 Pydantic 模型，而非以裸字典临时拼接。

### 5.2 路径统一解析

- 配置文件中的相对路径以配置文件所在目录为基准；
- 该语义由 `src/dpeva/utils/config.py` 统一承接；
- 文档和 recipes 应默认鼓励使用相对路径，避免环境绑定。

### 5.3 工作流独立运行 + 共享底层复用

- 各功能工作流独立存在并通过 CLI 单独启动；
- workflow 负责编排，manager/module 负责共享底层能力；
- 新功能应优先复用 `io/`、`sampling/`、`uncertain/`、`submission/`、`labeling/` 等模块，而不是在 workflow 中重写逻辑。

### 5.4 执行入口与导出入口分离

- 真实执行主链路以 `cli.py -> workflows/*.py` 为准；
- `src/dpeva/workflows/__init__.py` 只承担导出面，不承担业务执行入口；
- 审查“是否接线”时，应以调用链为准，而非以 `__init__.py` 是否导出为准。

### 5.5 阶段化工作流与完成标记

- labeling 已形成 `prepare/execute/extract/postprocess` 阶段化执行契约；
- `DPEVA_TAG: WORKFLOW_FINISHED` 是重要编排锚点，但不同 workflow 的主流程与 worker 日志语义仍需在文档中精确区分。

## 6. 冗余、过时与可删减项判断

### 建议已修复或已重写

- `docs/source/plans/README.md`：已从过时计划页改写为通用导航页

### 建议后续继续治理

- archive 中带绝对路径或旧命令占位符的历史文档应保持只读，但可在归档索引中加“历史上下文，仅供参考”的说明，避免被误当成现行规范
- `guides/`、`policy/`、`architecture/` 等目录与 `docs/source/` 的镜像关系仍可进一步收敛，但这不影响当前 AGENTS 方案实施

## 7. 后续建议

1. 以本轮已修复的入口页为基线，后续继续将 active 文档中的仓库内文档链接收敛到相对路径。
2. 将“配置中心化、路径解析、workflow 独立 + manager 复用、执行入口边界”这些稳定实践收敛为可引用的工程契约说明。
3. 按照本轮新增计划文件，重构 `AGENTS.md` 为“项目目标 + 核心契约 + 文档导航”的目录型文档。
4. 若下一个版本继续进行文档治理，优先评估 `guides/`、`policy/`、`architecture/` 与 `docs/source/` 的镜像收敛策略。

## 8. 相关交付物

- AGENTS 改造方案：`docs/plans/2026-04-05-AGENTS-Docs-Governance-Plan.md`
