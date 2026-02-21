# DP-EVA 文档导航（Docs Index）

本文档是 `docs/` 的统一入口：提供读者分流、权威信息源定位、以及文档维护规则。

## 1. 我应该读哪一类文档？

### 1.1 使用者（想把流程跑起来）

- 项目概览与快速开始：[README.md](/README.md)
- 安装与环境准备：[installation.md](/docs/guides/installation.md)
- Quickstart（最短路径跑通）：[quickstart.md](/docs/guides/quickstart.md)
- CLI 使用指南：[cli.md](/docs/guides/cli.md)

### 1.2 开发者（要改代码/加功能/修Bug）

- 开发流程标准与架构说明：[developer-guide.md](/docs/guides/developer-guide.md)
- 配置字段权威参考（查表）：[config-schema.md](/docs/reference/config-schema.md)
- 参数验证与约束补充：[validation.md](/docs/reference/validation.md)
- Slurm 使用与排障：[slurm.md](/docs/guides/slurm.md)
- 文档贡献指南：[contributing.md](/docs/policy/contributing.md)

### 1.3 研究者（关注算法假设/设计权衡/实验结论）

- 系统设计与模式分析：[DP-EVA_Design_Report.md](/docs/design/DP-EVA_Design_Report.md)
- DeepMD 依赖决策记录：[2026-02-04-deepmd-dependency.md](/docs/architecture/decisions/2026-02-04-deepmd-dependency.md)
- 描述符模长假设与实验报告：[modulo-hypothesis.md](/docs/reports/modulo-hypothesis.md)

## 2. 文档分层与权威来源（避免重复/冲突）

- `docs/reference/`：权威“查表类”参考（字段列表、校验规则）。只在这里维护全量字段说明。
- `docs/guides/`：主线操作指南（Quickstart/CLI/配置/Slurm/测试专题等）。遇到字段解释只链接到 `docs/reference/`。
- `docs/architecture/`：架构与关键技术决策（ADR）。
- `docs/governance/`：文档治理交付物（规划、审计、追踪矩阵、工具配置）。
- `docs/reports/`：一次性分析/实验/评审结论（默认只追加）。
- `docs/archive/`：历史与弃用文档（只读）。必须在文件头显式标注适用版本与是否已落地。

## 3. 文档维护规则（建议作为团队约定）

- 变更 `src/` 的用户接口、配置字段、关键目录结构时，PR 必须同步更新：
  - `docs/reference/*`（若涉及字段/校验）
  - `docs/guides/*`（若涉及使用方式/流程）
- 文档必须以“单一权威来源”为原则：同一份字段字典不允许在多处复制粘贴维护。
- `docs/archive/` 内容默认不回写，除非修正事实性错误（需要保留修订说明）。
