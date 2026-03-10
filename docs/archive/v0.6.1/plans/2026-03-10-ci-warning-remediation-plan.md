---
title: CI WARNING 治理修复方案（可追溯执行版）
status: active
audience: Developers / Maintainers
last-updated: 2026-03-10
owner: Docs Owner
---

# CI WARNING 治理修复方案（可追溯执行版）

**版本**: v1.1  
**日期**: 2026-03-10  
**状态**: Closed  
**目标**: 将当前文档构建 WARNING 从 77 降至 0，并完成文档治理稳态化收尾（责任归属、提交流程、CI 触发面、脚本鲁棒性）。

## 1. 现状确认（基于本次复核）

### 1.1 复核命令

```bash
cd docs
make clean
make html SPHINXOPTS="--keep-going"
```

### 1.2 复核结果（2026-03-10）

- 构建成功，当前 WARNING 总数：**77**
- 分类统计：
  - `toc.not_included`: **19**
  - `myst.xref_missing`: **54**
  - `misc.highlighting_failure`: **4**

### 1.3 对《2026-03-10-ci-warning-analysis.md》结论确认

| 项目 | 报告结论 | 当前确认 | 结论 |
| :--- | :--- | :--- | :--- |
| Mermaid 依赖接入 | 已修复 | `pyproject.toml` 已含 `sphinxcontrib-mermaid`，`docs/source/conf.py` 已启用扩展 | **部分完成**（依赖层完成，文档中仍有 `mermaid` 代码块被当作未知 lexer） |
| CI 阻塞恢复 | 已修复 | `.github/workflows/docs-check.yml` 已移除 `-W` | **已完成** |
| 部分 TOC 缺失修复 | 已修复 | 仍存在 `toc.not_included=19` | **部分完成** |
| 剩余问题约 77 条 | 待修复 | 本次 clean build 仍为 77 | **已确认** |

## 2. 治理目标与验收标准

### 2.1 阶段目标

1. **阶段A（降噪）**：77 → ≤20，优先清理高频 `myst.xref_missing`
2. **阶段B（清零）**：≤20 → 0，完成 TOC 收口与语法告警清零
3. **阶段C（门禁恢复）**：CI 恢复 `-W`，并将治理 lint 从软失败改为硬失败

### 2.2 最终验收标准

- `make html SPHINXOPTS="-W --keep-going"` 退出码为 0
- `docs-check.yml` 在 PR/Push 均启用 `-W` 严格模式
- `doc-lint.yml` 的结构检查与新鲜度检查不再 `continue-on-error`
- 方案中所有任务状态为 `DONE`，并附带执行证据

## 3. 修复任务分解（可追溯）

| Task ID | 问题类型 | 目标范围 | 修复动作 | 验收命令 | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CIW-001 | XREF | `docs/README.md`、`docs/{architecture,guides,policy,reference}/README.md` | 统一内部文档链接为 Sphinx 可解析形式（优先使用文档相对路径，不再引用仓库绝对路径） | `make html SPHINXOPTS="--keep-going"` | DONE |
| CIW-002 | XREF | `docs/guides/testing/*.md`、`docs/architecture/design-report.md` | 将指向 `src/`、`tests/`、`examples/` 的链接改为明确策略：文档内链使用 doc 引用，仓库文件链接改为代码仓库 URL 或 `literalinclude` | `make html SPHINXOPTS="--keep-going"` | DONE |
| CIW-003 | TOC | `docs/source/index.rst`、`docs/source/**/index.rst` | 建立“应纳管文档清单”，补齐 toctree；确认为历史文档的文件迁移至 `archive` 或加入 `exclude_patterns` | `make html SPHINXOPTS="--keep-going"` | DONE |
| CIW-004 | Lexing | `docs/guides/developer-guide.md`、`docs/plans/governance/2026-03-09-doc-system-maintenance-execution-detail.md`、`docs/architecture/decisions/2026-02-04-deepmd-dependency.md` | 修复 `mermaid` 与代码块高亮：Mermaid 改为 ` ```{mermaid}` 指令；修复 TOML/YAML 示例非法字符 | `make html SPHINXOPTS="--keep-going"` | DONE |
| CIW-005 | CI 门禁 | `.github/workflows/docs-check.yml`、`.github/workflows/doc-lint.yml` | 分阶段恢复严格门禁：先启用 doc-lint 硬失败，再启用 docs-check `-W` | GitHub Actions PR 验证 | DONE |
| CIW-006 | 治理固化 | `scripts/doc_check.py`、`docs/policy/contributing.md` | 增加“禁止仓库绝对路径文档内链”检查，更新贡献规范并给出示例 | `python3 scripts/doc_check.py` | DONE |
| CIW-007 | 稳态化收尾 | `docs/policy/*`、`.github/*`、`docs/governance/inventory/*`、`scripts/doc_check.py` | 固化稳态运行基线、Owner 矩阵、PR 模板、CodeOwners、CI 触发面与 linkcheck 策略 | `python3 scripts/doc_check.py` + `make html` + `make linkcheck` | DONE |

## 4. 执行顺序与依赖

| 批次 | 任务 | 依赖 | 退出条件 |
| :--- | :--- | :--- | :--- |
| Batch-1 | CIW-001, CIW-002 | 无 | `myst.xref_missing` 显著下降（目标 ≤15） |
| Batch-2 | CIW-004, CIW-003 | Batch-1 | `toc.not_included=0` 且 `misc.highlighting_failure=0` |
| Batch-3 | CIW-006, CIW-005 | Batch-2 | `-W` 恢复且 CI 全绿 |
| Batch-4 | CIW-007 | Batch-3 | 稳态化基线文档、责任归属与流程门禁全部落地并本地验证通过 |

## 5. 回写与进度标记规则

### 5.1 任务状态枚举

- `TODO`：未开始
- `DOING`：进行中
- `BLOCKED`：受阻
- `DONE`：已完成并通过验收

### 5.2 回写要求

每完成一个 Task，必须在本文件同步回写：

1. 更新“修复任务分解”中的 `状态`
2. 在“执行日志”新增一条记录
3. 更新“告警基线变化”

## 6. 执行日志（用于持续回写）

| 日期 | Task ID | 动作摘要 | 告警变化（前→后） | 证据（命令/CI 链接） | 结果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-10 | BASELINE | 建立 clean build 基线 | 77 → 77 | `make clean && make html SPHINXOPTS="--keep-going"` | DONE |
| 2026-03-10 | CIW-001 | 修复核心入口文档与 README 链接策略，消除 Sphinx XREF 误判 | 77 → 49 | `make clean && make html SPHINXOPTS="--keep-going"` | DONE |
| 2026-03-10 | CIW-002 | 修复 testing/design 文档中的代码与示例链接策略，统一为仓库 URL | 49 → 22 | `make clean && make html SPHINXOPTS="--keep-going"` | DONE |
| 2026-03-10 | CIW-004 | 修复 Mermaid/TOML/YAML 高亮告警 | 22 → 19 | `make clean && make html SPHINXOPTS="--keep-going"` | DONE |
| 2026-03-10 | CIW-003 | 在 `docs/source/index.rst` 建立隐藏 toctree 收口 orphan 文档 | 19 → 0 | `make clean && make html SPHINXOPTS="--keep-going"` | DONE |
| 2026-03-10 | CIW-006 | 增加绝对路径链接治理检查并补齐文档 front matter 与目录 README | 0 → 0 | `python3 scripts/doc_check.py` | DONE |
| 2026-03-10 | CIW-005 | 恢复 CI 严格门禁（docs-check `-W`，doc-lint 硬失败） | 0 → 0 | `make clean && make html SPHINXOPTS="-W --keep-going"` | DONE |
| 2026-03-10 | CIW-007 | 完成稳态化收尾：Owner 矩阵、PR 模板、CODEOWNERS、CI 触发面补齐、linkcheck 策略收敛、脚本鲁棒性增强 | 0 → 0 | `python3 scripts/doc_check.py` + `python3 scripts/check_docs_freshness.py --days 90` + `make html SPHINXOPTS="-W --keep-going"` + `make linkcheck SPHINXOPTS="-W --keep-going -D linkcheck_ignore='https://github.com/QuantumMisaka/dpeva/.*'"` | DONE |

## 7. 告警基线变化

| 日期 | 总数 | TOC | XREF | LEX | 备注 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 2026-03-10 | 77 | 19 | 54 | 4 | 初始基线（本方案建立时） |
| 2026-03-10 | 49 | 19 | 27 | 3 | 完成核心与专题文档链接策略修复 |
| 2026-03-10 | 22 | 19 | 0 | 3 | XREF 清零 |
| 2026-03-10 | 0 | 0 | 0 | 0 | 完成 TOC 收口与高亮修复 |
| 2026-03-10 | 0 | 0 | 0 | 0 | 完成治理稳态化收尾并验证通过 |

## 8. 风险与应对

| 风险 | 触发条件 | 应对策略 |
| :--- | :--- | :--- |
| 历史文档链接债务大 | archive/plans 旧文档存在大量失效引用 | 对历史文档采用“归档隔离策略”，不影响 active 文档门禁 |
| 路径策略不一致反复回归 | 开发者继续写仓库绝对路径链接 | 在 `doc_check.py` 增加规则并纳入 CI 硬门禁 |
| 一次性全量修复冲突大 | 同时改动文档入口与索引文件 | 采用 Batch 批次小步提交，每批独立验收 |

## 9. 完成定义（DoD）

- [x] `myst.xref_missing = 0`
- [x] `toc.not_included = 0`
- [x] `misc.highlighting_failure = 0`
- [x] `docs-check.yml` 恢复 `-W`
- [x] `doc-lint.yml` 去除 `continue-on-error`
- [x] 本文所有 Task 状态均为 `DONE`

## 10. 稳态化收尾交付物

- 制度：`docs/policy/maintenance.md`、`docs/policy/quality.md`、`docs/policy/contributing.md` 已补齐稳态门禁与提交流程
- 责任：新增 `docs/governance/inventory/owners-matrix.md`，建立目录级 Owner 责任映射
- 流程：新增 `.github/PULL_REQUEST_TEMPLATE.md`，文档治理检查清单纳入 PR 必填
- 审查：新增 `.github/CODEOWNERS`，将治理关键路径纳入强制评审
- 自动化：`docs-check.yml` 与 `doc-lint.yml` 完成触发面补齐；`doc_check.py` 增加 Owner 覆盖率提示并清理异常吞没
- 稳定性：linkcheck 对仓库 GitHub URL 使用忽略策略，避免分支未合并与网络波动导致 PR 随机失败
