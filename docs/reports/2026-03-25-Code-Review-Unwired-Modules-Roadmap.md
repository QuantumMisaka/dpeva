---
title: Unwired Modules Remediation Roadmap
status: active
audience: developers
last-updated: 2026-03-25
owner: DP-EVA Maintainers
---

# 2026-03-25 未接入模块修复优先级路线图

## 1. 目标

- 将“模块存在但未接入标准工作流链路”的风险收敛为可执行、可验收的治理序列。
- 保持现有主链路稳定（`train/infer/analysis/feature/collect/label/clean`）前提下推进治理，不引入破坏性行为。

## 2. 输入依据

- 工作流入口映射与候选审查：`2026-03-25-Code-Review-Workflow-Entry-Mapping.md`
- Collection 出图审计与可生成性状态：`Collection Workflow 全量出图审计报告.md`
- 已落地修复：Collection Candidate parity 两图已接入标准 `collect` 主链路。

## 3. 优先级分层（P0/P1/P2/P3）

| 优先级 | 目标 | 当前结论 | 处置策略 | 验收标准 |
|---|---|---|---|---|
| P0 | 主链路功能缺失或错误接线 | 当前未发现新的 P0（Candidate parity 已修复） | 持续门禁，出现即热修 | 相关 workflow 单测通过且关键输出存在 |
| P1 | 影响可观测性或可维护性的接线不清 | Collection 中已补齐 rescaled 依赖门控与日志 | 继续补齐日志语义与断言 | 异常/跳过路径日志可定位，测试覆盖 |
| P2 | 可复核资产不足（文档/审计缺口） | 已有两份审计报告，但跨报告引用可进一步标准化 | 统一报告模板与交叉链接 | 报告含“模块位置/状态/证据/建议/优先级” |
| P3 | 包级聚合/占位模块定位不清（低风险） | 候选主要集中在 `__init__` 聚合/占位模块 | 保留并文档化，不引入业务逻辑 | 开发文档明确“公共 API/占位”定位 |

## 4. 下一阶段执行清单

### 4.1 P1（建议本迭代完成）
- 为 Collection 的 rescaled 缺失路径增加更细粒度日志标签（含触发条件字段）。
- 在 workflow 级测试中增加“日志消息关键字 + 产物不存在”双断言，避免静默回归。

### 4.2 P2（建议并行推进）
- 在 `docs/reports/` 的相关报告中补充互链段落（入口映射报告 ↔ 出图审计报告 ↔ 本路线图）。
- 固化“未接入模块审查表”模板，后续新模块复用同一结构。

### 4.3 P3（按需推进）
- 对以下包级文件补充最小职责说明，避免误用：
  - `src/dpeva/workflows/__init__.py`
  - `src/dpeva/analysis/__init__.py`
  - `src/dpeva/io/__init__.py`
  - `src/dpeva/labeling/__init__.py`
  - `src/dpeva/sampling/__init__.py`
  - `src/dpeva/uncertain/__init__.py`

## 5. RACI（执行责任）

| 项目 | Responsible | Accountable | Consulted | Informed |
|---|---|---|---|---|
| Collection 接线回归门禁 | Workflow Maintainer | Tech Lead | QA | 全体开发 |
| 审计报告模板与互链 | Docs Owner | Tech Lead | Workflow Owner | 全体开发 |
| 包级 API/占位定位文档化 | Module Owner | Architect | Docs Owner | 全体开发 |

## 6. 建议门禁命令

```bash
pytest tests/unit/workflows/test_collect_refactor.py tests/unit/uncertain/test_visualization.py
ruff check .
python3 scripts/doc_check.py
python3 scripts/check_docs_freshness.py --days 90
make -C docs html SPHINXOPTS="-W --keep-going"
```
