# 文档系统治理改进路线图 (Docs Governance Roadmap)

- Status: active
- Audience: Maintainers / Developers
- Last-Updated: 2026-03-09

本文档基于 v0.6.0 文档治理审计结果及 Sphinx 自动化构建调研，规划下一步文档系统的改进路径。

## 1. 现状审计总结 (Audit Summary)

### 1.1 责任矩阵 (Responsibility Matrix)

| 文档目录 | 创建 (Create) | 审核 (Review) | 维护 (Maintain) | 归档 (Archive) | 现状风险 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `docs/policy/` | Maintainer | Maintainer | Maintainer | - | 稳定 |
| `docs/reference/` | Dev | Maintainer | Dev | - | **High**: 手工维护导致与代码脱节 |
| `docs/guides/` | Dev/User | Dev | Dev | - | Medium: 内容更新滞后于功能 |
| `docs/plans/` | Feature Owner | Maintainer | Feature Owner | Docs Admin | **High**: 归档不及时，混淆视听 |
| `AGENTS.md` | - | - | - | - | **Critical**: 与 `.trae/skills` 职责冲突，且冗余 |
| `README.md` | Maintainer | All | Maintainer | - | Medium: 承担了过多的 Quickstart 职责 |

### 1.2 问题清单与改进建议

| 问题类型 | 具体表现 | 改进建议 | 优先级 |
| :--- | :--- | :--- | :--- |
| **冲突 (Conflict)** | `docs/reference/config-schema.md` 与 `src/dpeva/config.py` 脱节。 | **Sync**: 引入 Sphinx + Pydantic 插件自动生成文档。 | **P0** |
| **冲突 (Conflict)** | `AGENTS.md` 与 `.trae/skills/*.md` 双重定义技能。 | **Merge**: 废弃 `AGENTS.md` 的技能描述，仅保留指向 Skills 的链接。 | **P0** |
| **冗余 (Redundancy)** | `README.md`, `AGENTS.md`, `installation.md` 三处重复安装命令。 | **Split**: `README` 仅保留核心介绍，安装指引统一收敛至 `installation.md`。 | **P1** |
| **规划 (Planning)** | `docs/governance/plans` 与 `docs/plans` 入口分散。 | **Merge**: 将治理计划统一迁移至 `docs/plans/governance/`。 | **P2** |
| **结构 (Structure)** | `docs/logo_design` 游离于规范之外；Archive 命名不规范。 | **Move**: 迁移至 `docs/assets/logo`；Archive 补全日期前缀。 | **P2** |

## 2. 改进路线图 (Roadmap)

### Phase 1: 结构瘦身与规范化 (v0.6.1 - Completed)
**目标**: 消除多头权威，收敛信息入口，为 Sphinx 接入做准备。

- [x] **Action 1**: 重构 `README.md`，移除具体安装步骤，改为引用链接。
- [x] **Action 2**: 改造 `AGENTS.md` 为纯导航页，内容迁移至 `.trae/skills/`。
- [x] **Action 3**: 建立 `docs/plans/governance/` 目录，归并分散的治理计划。
- [x] **Action 4**: 标准化目录结构：
    - 移动 `docs/logo_design/` -> `docs/assets/logo/`。
    - 重命名 `docs/archive/` 下的历史文件（增加 `YYYY-MM-DD` 前缀）。
    - 更新 `docs-catalog.md` 清单。

### Phase 2: Sphinx 自动化构建体系 (v0.7.0 - Completed)
**目标**: 建立代码即文档 (Docs as Code) 的自动化流水线，对齐 DeepModeling 生态。

- [x] **Setup 1**: 初始化 Sphinx 环境 (`docs/conf.py`, `docs/index.rst`)。
    - 主题：`sphinx_book_theme`。
    - 插件：`myst_parser` (Markdown支持), `sphinx.ext.autodoc` (API), `sphinxcontrib.autodoc_pydantic` (配置)。
- [x] **Setup 2**: 建立文档分层索引 (`docs/index.rst`):
    - **User Guide**: `docs/guides/` (Install, Quickstart, CLI, Slurm)。
    - **Reference**: `docs/reference/` (由 Pydantic 模型自动生成的 Config Schema)。
    - **Developer Guide**: `docs/guides/developer-guide.md`, `docs/architecture/`。
    - **API Reference**: `docs/api/` (自动生成的 Python API)。
- [x] **Setup 3**: 配置 GitHub Actions (`.github/workflows/docs.yml`)。
    - 触发：Push to `main` / Release Tag。
    - 动作：Build -> Deploy to `gh-pages`。

### Phase 3: 质量体系与模板 (v0.7.x - Completed)
**目标**: 建立长效机制，量化文档质量。

- [x] **Metric 2**: **Freshness** (核心文档无 >30 天未更新)。
    - Implemented script: `scripts/check_docs_freshness.py`
- [x] **Template**: 建立标准模板库 (`docs/_templates/`)：
    - 架构设计 (ADR) - Existing
    - 用户指南 (How-to) - Created
    - 故障排查 (Troubleshooting) - Created

### Phase 4: 归一化与闭环 (v0.6.0 - Completed)
**目标**: 在 v0.6.0 正式发布前消除“双重事实”风险，强化自动化闭环。

- [x] **Consolidation**: 废弃静态 `config-schema.md`，全面转向 Sphinx 生成的 API 文档。
    - 在 `docs/reference/README.md` 中添加重定向说明。
    - 确保 `docs/api/config.rst` 输出质量达到 SSOT 标准。
- [x] **Enforcement**: 将质量检查集成到 CI。
    - 在 `.github/workflows/docs.yml` 中增加 `check_docs_freshness.py` 步骤。
    - 设置非阻塞性警告（Warning-level）以培养习惯。
- [x] **Population**: 提升 API 文档覆盖率。
    - 扫描 `src/dpeva`，为所有核心模块（Sampling, UQ, Submission）补全 `.rst` 索引文件。
    - 制定 "Docstring First" 开发规范。

## 3. 资源需求

- **人员**: 文档负责人 1 名 (兼)，DevOps 工程师 0.5 名 (CI 配置)。
- **工具**: Sphinx, GitHub Actions, Pre-commit hooks。

## 4. 交付物

1.  **在线文档站点** (GitHub Pages)。
2.  **自动化配置** (`docs/conf.py`, `.github/workflows/docs.yml`)。
3.  **瘦身后的文档库** (无冗余、无冲突)。
