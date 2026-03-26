---
title: Unwired Module Audit Template
status: active
audience: Maintainers / Developers
last-updated: 2026-03-25
owner: Docs Owner
---

# 未接入模块审查模板

用于统一“模块存在但未接入标准工作流链路”的审查格式，适用于 `train/infer/analysis/feature/collect/label/clean` 等模块治理场景。

## 1. 审查元信息

- 审查主题：
- 审查日期：
- 审查范围：
- 审查人：
- 关联变更（PR/Issue/Plan）：

## 2. 入口到实现映射

| CLI/入口 | 配置模型 | Workflow | 核心模块 | 关键证据 |
|---|---|---|---|---|
| 示例：`collect` | `CollectionConfig` | `CollectionWorkflow` | `UQManager` / `SamplingManager` | `src/dpeva/workflows/collect.py#L...` |

## 3. 未接入候选清单（结构化）

| 模块位置 | 预期所属工作流 | 当前接入状态 | 证据摘要 | 处置建议 | 优先级 |
|---|---|---|---|---|---|
| `src/dpeva/xxx.py` | collect | 未接入/保留占位/已接入 | 导入链路、调用链路、测试证据 | 接入/保留/移除 | P0/P1/P2/P3 |

## 4. 判定规则

- `已接入`：存在从入口到模块的可追溯主链路调用，且有测试或运行证据。
- `保留占位`：不在主链路中，但承担命名空间、导出聚合或预留职责。
- `未接入`：模块具备业务功能但缺少主链路调用。

## 5. 建议动作

- P0：主链路功能缺失，必须立即修复并回归验证。
- P1：可观测性或门控缺失，当前迭代修复。
- P2：文档、追踪或模板能力不足，计划迭代补齐。
- P3：低风险占位/聚合模块，文档化职责边界。

## 6. 验证与门禁

```bash
pytest tests/unit/workflows/test_collect_refactor.py tests/unit/uncertain/test_visualization.py
ruff check .
python3 scripts/doc_check.py
python3 scripts/check_docs_freshness.py --days 90
make -C docs html SPHINXOPTS="-W --keep-going"
```
