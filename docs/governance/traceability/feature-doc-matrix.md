---
title: Feature-to-document traceability matrix
status: active
audience: Developers
last-updated: 2026-09-05
owner: Docs Owner
---

# 功能-文档双向追踪矩阵（Feature ↔ Docs）

本矩阵是供人审阅的索引；机器校验的唯一数据源是
[`capability-evidence.json`](capability-evidence.json)。每个 capability ID
必须在 JSON 注册表中恰好出现一次，并具备代码、测试、文档、负责人和现存证据路径。
本表不从源码文本推断功能状态，也不替代 capability manifest 的
`supported`/`experimental`/`unsupported`/`blocked-upstream` 声明。

| Capability ID | 代码实现 | 文档/recipe 入口 | 最新证据 |
|---|---|---|---|
| `workflow.train` | `src/dpeva/cli.py`; `src/dpeva/workflows/train.py` | `docs/guides/cli.md`; `examples/recipes/training/config_train.json` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `workflow.infer` | `src/dpeva/cli.py`; `src/dpeva/workflows/infer.py` | `docs/guides/cli.md`; `examples/recipes/inference/config_infer.json` | `docs/reports/2026-09-04-run-contract-pilot-report.md` |
| `workflow.feature` | `src/dpeva/cli.py`; `src/dpeva/workflows/feature.py` | `docs/guides/cli.md`; `examples/recipes/feature_generation/config_feature.json` | `docs/reports/2026-09-04-run-contract-pilot-report.md` |
| `workflow.collect` | `src/dpeva/cli.py`; `src/dpeva/workflows/collect.py` | `docs/guides/cli.md`; `examples/recipes/collection/config_collect_normal.json` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `workflow.analysis` | `src/dpeva/cli.py`; `src/dpeva/workflows/analysis.py` | `docs/guides/cli.md`; `examples/recipes/analysis/config_analysis.json` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `workflow.label` | `src/dpeva/cli.py`; `src/dpeva/workflows/labeling.py` | `docs/guides/cli.md`; `examples/recipes/labeling/config_cpu.json` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `workflow.clean` | `src/dpeva/cli.py`; `src/dpeva/workflows/data_cleaning.py` | `docs/guides/cli.md`; `examples/recipes/data_cleaning/config_clean_all_thresholds.json` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `run.contract` | `src/dpeva/run/context.py`; `src/dpeva/run/recorder.py`; `src/dpeva/run/status.py` | `docs/guides/developer-guide.md`; `examples/recipes/README.md` | `docs/reports/2026-09-04-run-contract-pilot-report.md` |
| `lineage.evaluation-card` | `src/dpeva/run/dataset.py`; `src/dpeva/run/model.py`; `src/dpeva/evaluation/card.py` | `docs/guides/cli.md`; `examples/recipes/evaluation/config_eval_card.json` | `docs/reports/2026-09-04-dataset-lineage-eval-card-acceptance.md` |
| `compatibility.deepmd-3.2` | `src/dpeva/compatibility/deepmd-3.2.json`; `src/dpeva/compatibility/adapter.py`; `src/dpeva/compatibility/attestation.py` | `docs/guides/developer/deepmd-kit-sai-build.md`; `docs/reports/templates/deepmd-3.2-qualification.md` | `docs/reports/2026-09-04-deepmd-3.2-compatibility.md` |

## 维护规则

- 新增或变更对外能力时，先更新 JSON 注册表，再同步本索引和契约测试矩阵。
- `evidence_path` 必须指向已提交的具体文件；路径存在不等于能力已经通过科学或兼容性晋级。
- 旧版归档能力仍可保留在归档文档中，但不应在本表新增没有当前代码、测试和文档入口的条目。
