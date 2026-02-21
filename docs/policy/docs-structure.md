# 文档结构与命名规范（Docs Structure & Naming）

- Status: active
- Audience: Maintainers / Developers
- Last-Updated: 2026-02-19

本文档定义当前版本的文档集结构与命名规范，并说明分类原则、状态规则与未来扩展预留目录。

## 1. 分类原则

- 按 Intent（功能）分层：Guides / Reference / Architecture / Decisions(ADR) / Reports / Policy / Governance / Archive / Templates / Assets
- 按 Cadence（更新频率）治理：
  - 高频：Guides（接口/目录结构变更必须同步）
  - 中频：Reference（字段/校验变更必须同步）
  - 低频：Architecture/Decisions/Reports（多为追加或小修）
- 单一权威来源：字段字典与校验规则仅在 `/docs/reference/*` 维护；其他文档只引用不复制。
- 多入口策略：同一主题只保留一个主文档，避免多入口导致内容分叉。

## 2. 命名规范

- 目录名：全小写，短语用 `-` 连接（如 `guides/testing`）。
- 主文档：语义清晰、稳定（如 `quickstart.md`、`config-schema.md`）。
- 审计/计划类报告：使用 `YYYY-MM-DD-<topic>.md`（例如 `2026-02-19-doc-accuracy-audit.md`）。
- 禁止在仓库内保留旧入口跳转页；旧路径兼容性应通过站点侧 301 或发布说明解决。

## 3. 状态规则

- 每篇 active 文档包含 `Status/Audience/Last-Updated` 元信息。
- `Status` 使用：`active`（现行）、`draft`（草稿）、`archived`（历史/仅供参考）。

## 4. 文档结构树

- 树状图 PNG：/docs/img/docs-structure-tree.png

```text
/docs
├── README.md
├── DOCS_CLASSIFICATION.md
├── _templates
│   ├── page.md
│   ├── adr.md
│   └── report.md
├── guides
│   ├── README.md
│   ├── installation.md
│   ├── quickstart.md
│   ├── cli.md
│   ├── configuration.md
│   ├── slurm.md
│   ├── troubleshooting.md
│   ├── developer-guide.md
│   └── testing
│       ├── README.md
│       ├── integration-slurm.md
│       ├── integration-slurm-plan.md
│       ├── integration-config-templates.md
│       └── multi-datapool-artifacts.md
├── reference
│   ├── README.md
│   ├── config-schema.md
│   └── validation.md
├── architecture
│   ├── README.md
│   ├── design-report.md
│   └── decisions
│       └── 2026-02-04-deepmd-dependency.md
├── reports
│   ├── README.md
│   └── modulo-hypothesis.md
├── policy
│   ├── README.md
│   ├── maintenance.md
│   ├── quality.md
│   ├── contributing.md
│   └── docs-structure.md
├── design
│   ├── README.md
│   ├── DP-EVA_Design_Report.md
│   └── modulo_distribution.png
├── governance
│   ├── README.md
│   ├── plans
│   │   └── 2026-02-18-doc-system-planning.md
│   ├── audits
│   │   ├── 2026-02-19-doc-accuracy-audit.md
│   │   └── 2026-02-19-docs-review-report.md
│   ├── traceability
│   │   └── feature-doc-matrix.md
│   ├── inventory
│   │   └── docs-catalog.md
│   └── tools
│       └── link-normalization.json
└── archive
    ├── README.md
    ├── Code_Review_Report_v2.7.1.md
    ├── Variable_Review_Report.md
    ├── governance
    │   ├── 2026-02-18-docs-completion-list.md
    │   ├── 2026-02-19-readme-coverage.md
    │   └── audits
    │       ├── 2026-02-18-docs-selfcheck.md
    │       └── 2026-02-19-link-check.md
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

