# 最终文档集结构设计方案

- Status: active
- Audience: Maintainers / Developers
- Last-Updated: 2026-02-19

本文档给出当前版本的文档集最终结构（含保留/新增），并说明分类原则、命名规范、版本号规则与未来扩展预留目录。

## 1. 分类原则

- 按 Intent（功能）分层：Guides / Reference / Architecture / Decisions(ADR) / Reports / Policy / Archive / Templates / Assets
- 按 Cadence（更新频率）治理：
  - 高频：Guides（接口/目录结构变更必须同步）
  - 中频：Reference（字段/校验变更必须同步）
  - 低频：Architecture/Decisions/Reports（多为追加或小修）
- 单一权威来源：字段字典与校验规则仅在 `/docs/reference/*` 维护；其他文档只引用不复制。
- 多入口策略：同一主题只保留一个主文档，避免多入口导致内容分叉。

## 2. 命名规范

- 目录名：全小写，短语用 `-` 连接（如 `guides/testing`）。
- 主文档：语义清晰、稳定（如 `quickstart.md`、`config-schema.md`）。
- 报告类：可使用语义名或日期前缀（如 `YYYY-MM-DD-*.md`）。
- 禁止在仓库内保留旧入口跳转页；旧路径的兼容性应通过站点侧 301 或发布说明解决。

## 3. 版本号与状态规则

- 每篇 active 文档包含 `Status/Audience/Last-Updated` 元信息。
- `Status` 使用：`active`（现行）、`draft`（草稿）、`archived`（历史/仅供参考）。
- 旧路径兼容性应通过站点侧 301 或发布说明解决。

## 4. 文档结构树（保留/新增）

说明：

- `[A]` = Active（现行权威/入口）
- 树状图 PNG：/docs/img/docs-structure-tree.png

```text
/docs
├── README.md [A]                      文档总导航（单一入口）
├── DOCS_CLASSIFICATION.md [A]         分类维度与规范
├── _templates [A]
│   ├── page.md
│   ├── adr.md
│   └── report.md
├── guides [A]
│   ├── README.md
│   ├── installation.md                安装与环境准备（新增）
│   ├── quickstart.md
│   ├── cli.md
│   ├── configuration.md
│   ├── slurm.md
│   ├── troubleshooting.md
│   ├── developer-guide.md
│   └── testing [A]
│       ├── README.md
│       ├── integration-slurm.md
│       ├── integration-slurm-plan.md
│       ├── integration-config-templates.md
│       └── multi-datapool-artifacts.md
├── reference [A]
│   ├── README.md
│   ├── config-schema.md
│   └── validation.md
├── architecture [A]
│   ├── README.md
│   ├── design-report.md [draft]       architecture 落点占位
│   └── decisions [A]
│       └── 2026-02-04-deepmd-dependency.md
├── reports [A]
│   ├── README.md
│   └── modulo-hypothesis.md
├── policy [A]
│   ├── README.md                      目录说明（新增）
│   ├── maintenance.md
│   ├── quality.md
│   └── contributing.md                文档贡献指南（新增）
├── design [A]
│   ├── README.md                      目录说明（新增）
│   ├── DP-EVA_Design_Report.md
│   └── modulo_distribution.png
├── main [A]
│   ├── README.md                      目录说明（新增）
│   ├── DP-EVA_Docs_Catalog.md
│   ├── DP-EVA_Docs_Completeness_SelfCheck_Report.md
│   ├── DP-EVA_Docs_Completion_List.md
│   ├── DP-EVA_Documentation_System_Planning_Report.md
│   ├── docs-review-report.md
│   ├── documentation-accuracy-audit.md
│   ├── feature-doc-matrix.md
│   ├── link-check-report.md
│   ├── readme-coverage-report.md
│   └── standardized-link-mapping.json 标准化链接映射表（新增）
└── archive [A]
    ├── README.md
    ├── Code_Review_Report_v2.7.1.md
    ├── Variable_Review_Report.md
    └── refactoring
        ├── SLURM_WORKFLOW_DESIGN.md
        ├── Training_Module_Refactoring_Report_v2.9.0.md
        ├── Runner_Interface_Refractor_Plan.md
        └── DP-EVA_Labeling_Module_Plan.md
```

## 5. 未来扩展预留目录

- `/docs/glossary/`：术语表（DataPool/System/Descriptor 等）
- `/docs/migrations/`：迁移指南（字段重命名、弃用策略、破坏性变更）
- `/docs/changelog/`：release notes 与兼容性说明
- `/docs/tools/`：文档检查脚本（链接检查、示例抽取与验证）
