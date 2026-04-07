---
title: Repository Audit
status: active
audience: Developers / Maintainers
last-updated: 2026-04-01
owner: Docs Owner
---

# 2026-04-01 仓库质量审查报告（Repository Audit）

- Status: active
- Audience: Developers / Maintainers
- Last-Updated: 2026-04-01

本文档整合本轮四份分域审查结果：`源码与测试`、`文档与项目元信息`、`脚本与 .trae/skills`、`CI/CD`。所有结论均回落到当前磁盘中的实际文件证据，覆盖 `src/`、`tests/`、`docs/`、`.trae/skills/`、`scripts/`、`.github/workflows/`、`README.md`、`AGENTS.md`、`pyproject.toml` 以及规格任务追踪。

## 1. 综合结论

- 当前未发现需要立即冻结仓库的 P0/Critical 问题；核心源码分层、Pydantic 配置契约、安全工具与主要单元测试资产仍处于可维护状态。
- 本轮真实高风险集中在“承诺了质量标准，但没有被门禁持续兑现”：
  - **High / P1**：CI/CD 没有任何面向 Python 主线质量的 `ruff` / `pytest` / `mypy` 门禁，而 `pyproject.toml` 与 `AGENTS.md` 已明确声明这些开发依赖与检查方式。
  - **High / P1**：文档治理脚本默认跳过活跃的 `docs/plans/` 与 `docs/reports/`，导致计划、审查报告、owner 覆盖与链接质量没有被当前治理门禁真实约束。
  - **High / P1**：根级 `README.md` 仍把占位文档站点和 `docs/source/*.rst` 源文件暴露为读者入口，文档入口与当前发布形态不一致。
  - **Medium / P2**：测试资产中仍存在硬编码绝对路径，且测试文档声明的覆盖率门禁没有实际自动化承接，削弱“测试可信度”。
  - **Medium / P2**：`scripts/` 与 `.trae/skills/` 中存在发布流程说明漂移与孤立脚本，维护知识没有完全对齐当前版本结构。
- 综合判断：仓库问题主要不在核心算法，而在治理资产之间的“断链”。优先级应为：先补 CI/CD 与文档门禁，再收敛入口文档与脚本/技能漂移，最后处理源码里的静默异常与测试便携性问题。

## 2. 四份审查结果整合

| 分域审查 | 覆盖对象 | 主要结论 | 代表证据 |
|---|---|---|---|
| 1. 源码与测试 | `src/`、`tests/` | 架构边界总体清晰，但存在静默吞错与测试便携性问题 | [cli.py](../../src/dpeva/cli.py)、[collect.py](../../src/dpeva/workflows/collect.py)、[analysis.py](../../src/dpeva/workflows/analysis.py)、[test_labeling_rotation_bug.py](../../tests/integration/test_labeling_rotation_bug.py) |
| 2. 文档与项目元信息 | `docs/`、`README.md`、`AGENTS.md`、`pyproject.toml` | 文档规范已定义，但入口文档与治理策略执行强度失配 | [README.md](../../README.md)、[quality.md](../policy/quality.md)、[AGENTS.md](../../AGENTS.md)、[pyproject.toml](../../pyproject.toml) |
| 3. 脚本与 `.trae/skills` | `scripts/`、`.trae/skills/` | 治理脚本存在活跃目录豁免，发布辅助说明与当前版本纪元不一致 | [doc_check.py](../../scripts/doc_check.py)、[check_docs_freshness.py](../../scripts/check_docs_freshness.py)、[release_helper.py](../../scripts/release_helper.py)、[release-helper/SKILL.md](../../.trae/skills/release-helper/SKILL.md) |
| 4. CI/CD | `.github/workflows/` | 只有文档相关工作流，没有源码质量主线；文档工作流还有重复与职责分裂 | [docs-check.yml](../../.github/workflows/docs-check.yml)、[doc-lint.yml](../../.github/workflows/doc-lint.yml)、[docs-deploy.yml](../../.github/workflows/docs-deploy.yml) |

## 3. 八维度评估总览

| 维度 | 当前判断 | 严重级别 | 优先级 | 证据 |
|---|---|---|---|---|
| 1. 代码风格与规范一致性 | 仓库已声明 `ruff`、`mypy`、`pytest` 作为开发基线，但声明尚未进入 CI 实际门禁 | High | P1 | [pyproject.toml](../../pyproject.toml)、[AGENTS.md](../../AGENTS.md)、[docs-check.yml](../../.github/workflows/docs-check.yml) |
| 2. 架构设计与模块边界 | CLI → Workflow → Manager 的主链路边界清晰；局部初始化阶段仍有静默异常 | Medium | P2 | [cli.py](../../src/dpeva/cli.py)、[collect.py](../../src/dpeva/workflows/collect.py)、[analysis.py](../../src/dpeva/workflows/analysis.py) |
| 3. 安全与外部命令执行 | 路径安全工具与安全测试存在，当前未见新的命令注入级问题 | Low | P3 | [security.py](../../src/dpeva/utils/security.py)、[command.py](../../src/dpeva/utils/command.py)、[test_path_traversal.py](../../tests/security/test_path_traversal.py) |
| 4. 性能与执行效率 | 核心工作流未见新的明显性能阻断；CI 与文档构建存在重复安装/重复构建开销 | Medium | P2 | [docs-check.yml](../../.github/workflows/docs-check.yml)、[docs-deploy.yml](../../.github/workflows/docs-deploy.yml) |
| 5. 测试完整性与覆盖率 | 测试目录完整，但真实门禁、便携性与覆盖率承诺之间存在明显缺口 | High | P1 | [tests/unit](../../tests/unit)、[tests/integration](../../tests/integration)、[UNIT_TESTS.md](../../tests/unit/UNIT_TESTS.md) |
| 6. 文档准确性与完整性 | 规范文档存在，但 README 入口、治理脚本豁免与部分版本叙述已发生漂移 | High | P1 | [README.md](../../README.md)、[quality.md](../policy/quality.md)、[developer-guide.md](../guides/developer-guide.md) |
| 7. 脚本可维护性与技能一致性 | 脚本与技能文档之间存在事实漂移，部分辅助脚本未接入任何流水线 | Medium | P2 | [gate.sh](../../scripts/gate.sh)、[verify_docs.sh](../../scripts/verify_docs.sh)、[release_helper.py](../../scripts/release_helper.py)、[release-helper/SKILL.md](../../.trae/skills/release-helper/SKILL.md) |
| 8. CI/CD 配置正确性与效率 | 当前 Actions 只覆盖 docs 治理，且对根 README、`.trae/skills`、一般源码改动缺少对应质量触发器 | High | P1 | [doc-lint.yml](../../.github/workflows/doc-lint.yml)、[docs-check.yml](../../.github/workflows/docs-check.yml)、[docs-deploy.yml](../../.github/workflows/docs-deploy.yml) |

## 4. 分级问题清单

| 编号 | 问题 | 严重级别 | 优先级 | 影响范围 | 直接证据 |
|---|---|---|---|---|---|
| F1 | GitHub Actions 未建立 Python 主线质量门禁，`ruff` / `pytest` / `mypy` 仅停留在文档与依赖声明中 | High | P1 | `src/`、`tests/`、`pyproject.toml`、开发流程 | [pyproject.toml](../../pyproject.toml) 仅声明 `ruff`/`mypy` 依赖；[AGENTS.md](../../AGENTS.md) 要求运行 `pytest tests/unit` 与 `ruff check .`；`.github/workflows/` 中仅存在 docs 相关工作流 |
| F2 | `doc_check.py` 与 `check_docs_freshness.py` 默认豁免活跃的 `docs/plans/`、`docs/reports/`，导致治理门禁与质量标准脱节 | High | P1 | `docs/plans/`、`docs/reports/`、owner 覆盖、链接完整性 | [doc_check.py](../../scripts/doc_check.py) 的 `IGNORE_DIRS` 含 `plans`、`reports`；[check_docs_freshness.py](../../scripts/check_docs_freshness.py) 默认排除 `plans`、`reports`；[quality.md](../policy/quality.md) 要求 active 文档 owner 覆盖率 100% |
| F3 | 根级 README 仍引用 `https://dpeva.readthedocs.io` 占位链接和 `docs/source/api/config.rst` 源路径 | High | P1 | 用户入口、外部读者认知、文档导航 | [README.md](../../README.md) 的 Documentation 段落仍暴露占位链接和 `.rst` 源文件路径 |
| F4 | 测试资产存在便携性与治理失真：集成测试硬编码本机绝对路径，单测文档声称的覆盖率门禁没有 CI 承接 | Medium | P2 | `tests/integration`、`tests/unit/UNIT_TESTS.md`、跨环境复现 | [test_labeling_rotation_bug.py](../../tests/integration/test_labeling_rotation_bug.py) 使用 `/home/...` 绝对路径；[UNIT_TESTS.md](../../tests/unit/UNIT_TESTS.md) 声明“CI 阻断/警告”覆盖率门槛，但当前工作流未实现 |
| F5 | 发布辅助知识漂移：`.trae/skills/release-helper` 仍指向 `Current Era (v0.4.x)`，与当前开发指南 `v0.7.x` 不一致 | Medium | P2 | `.trae/skills`、发布流程、维护培训 | [release-helper/SKILL.md](../../.trae/skills/release-helper/SKILL.md) 仍要求定位 `Current Era (v0.4.x)`；[developer-guide.md](../guides/developer-guide.md) 当前版本历史标题已为 `Current Era (v0.7.x)` |
| F6 | 发布与质量脚本未形成闭环：`gate.sh`、`verify_docs.sh`、`audit.py` 存在，但未被任何现有工作流调用 | Medium | P2 | `scripts/`、团队执行路径、CI 一致性 | [gate.sh](../../scripts/gate.sh)、[verify_docs.sh](../../scripts/verify_docs.sh)、[audit.py](../../scripts/audit.py) 在 `.github/workflows/` 中均无引用 |
| F7 | `collect` / `analysis` 在回填 `config_path` 时使用静默 `except Exception: pass` | Medium | P2 | `src/dpeva/workflows`、Slurm 自提交、故障定位 | [collect.py](../../src/dpeva/workflows/collect.py)、[analysis.py](../../src/dpeva/workflows/analysis.py) 的初始化阶段直接吞掉异常 |
| F8 | 文档工作流职责重叠：文档检查与部署链路同时承担构建职责，维护面分裂 | Medium | P2 | `.github/workflows`、CI 耗时、发布稳定性 | [docs-check.yml](../../.github/workflows/docs-check.yml) 与 [docs-deploy.yml](../../.github/workflows/docs-deploy.yml) 都执行 docs 构建，但目标职责不同 |

## 5. 优先级与行动项

| 优先级 | 行动项 | 目标 | 对应问题 |
|---|---|---|---|
| P1 | 新增统一 Python 质量工作流，最少接入 `ruff check .`、`pytest tests/unit`，并决定是否把 `mypy` 作为告警或阻断项 | 让 `pyproject.toml` 与 `AGENTS.md` 中承诺的开发基线变成 PR 门禁 | F1 |
| P1 | 收紧文档治理脚本，至少把活跃 `docs/plans/` 与 `docs/reports/` 纳入 metadata、owner、freshness 校验 | 消除“活跃治理文档不受治理”的自相矛盾 | F2 |
| P1 | 重写根 README 的 Documentation 入口，移除占位站点与 `.rst` 源路径 | 统一仓库对外文档入口与导航语言 | F3 |
| P2 | 修复测试便携性，移除集成测试中的硬编码本机路径，并让覆盖率声明与真实 CI 保持一致 | 提升测试跨环境可复现性与信任度 | F4 |
| P2 | 收敛发布辅助知识：统一 `release_helper.py`、`.trae/skills/release-helper`、`developer-guide.md` 的版本纪元与更新路径 | 让脚本与技能真正可指导维护者执行发布 | F5 |
| P2 | 决定 `gate.sh` / `verify_docs.sh` / `audit.py` 的归宿：接入 CI、整合到主脚本，或下线冗余入口 | 降低“存在但无人执行”的治理噪声 | F6 |
| P2 | 将静默异常改为显式日志或窄化异常类型 | 提升 Slurm 自提交与配置回填问题的可诊断性 | F7 |
| P2 | 合并或分层整理现有文档构建/发布流水线 | 减少重复安装、重复构建与配置漂移 | F8 |

## 6. 关键证据解读

### 6.1 核心实现没有失控

- [cli.py](../../src/dpeva/cli.py) 维持了统一入口，工作流按命令懒加载，模块边界相对清楚。
- [config.py](../../src/dpeva/config.py) 仍是单一配置权威入口，`pyproject.toml` 的打包和入口脚本也保持简洁。
- [security.py](../../src/dpeva/utils/security.py) 与 [test_path_traversal.py](../../tests/security/test_path_traversal.py) 表明路径安全防线仍被显式维护。

### 6.2 真实短板在“治理资产彼此脱节”

- [AGENTS.md](../../AGENTS.md) 和 [pyproject.toml](../../pyproject.toml) 已定义开发检查基线，但 `.github/workflows/` 没有任何源码质量工作流与之对齐。
- [doc_check.py](../../scripts/doc_check.py) 与 [check_docs_freshness.py](../../scripts/check_docs_freshness.py) 都把活跃的 `plans`、`reports` 排除在外，直接削弱了 [quality.md](../policy/quality.md) 的执行力。
- [README.md](../../README.md) 仍把占位 docs 链接与 `.rst` 源文件当成用户入口，说明对外入口文案没有跟随实际文档体系演进。
- [test_labeling_rotation_bug.py](../../tests/integration/test_labeling_rotation_bug.py) 的绝对路径表明部分测试仍依赖作者本机目录，和“仓库可复现”的目标冲突。
- [release-helper/SKILL.md](../../.trae/skills/release-helper/SKILL.md) 与 [developer-guide.md](../guides/developer-guide.md) 的版本纪元不一致，说明 `.trae/skills` 也需要纳入真实维护链路。

## 7. 索引与验证记录

- 已复核 [docs/source/index.rst](../source/index.rst) 的当前导航策略：本次仅修订既有报告内容，不新增、不移动文档，因此无需调整 `toctree`；现有索引明确说明 `docs/reports/` 不纳入主导航稳定入口。
- 本轮文档修订后的验证命令：
  - `ruff check .`
  - `python3 scripts/doc_check.py`
  - `python3 scripts/check_docs_freshness.py --days 90`
  - `make -C docs html SPHINXOPTS="-W --keep-going"`

## 8. 审计结论

本轮修订后的真实结论是：DP-EVA 的“软件本体”仍可演进，但“治理系统”还没有把既有质量标准闭环起来。若只看源码，仓库状态容易被高估；若把测试、文档、脚本、技能与 CI/CD 一起看，当前最需要做的不是大规模重构，而是先把已经写进文档和工具链的规则真正落到自动化门禁中。
