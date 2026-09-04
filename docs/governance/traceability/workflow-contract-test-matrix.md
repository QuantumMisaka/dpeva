---
title: Workflow contract-to-test traceability matrix
status: active
audience: Developers
last-updated: 2026-09-05
owner: Docs Owner
---

# 工作流-契约测试矩阵（Workflow Contract ↔ Tests）

本矩阵是供人审阅的索引，与
[`feature-doc-matrix.md`](feature-doc-matrix.md) 使用相同的 capability ID
集合。机器校验由 [`capability-evidence.json`](capability-evidence.json) 和
`scripts/check_traceability.py` 完成；本表不复制或推断源码中的能力状态。

## 契约与测试入口

| Capability ID | 对外契约/入口 | 最小测试证据 | 最新证据 |
|---|---|---|---|
| `workflow.train` | `dpeva train <config>`；训练目录与完成状态 | `tests/unit/workflows/test_train_workflow_init.py`; `tests/unit/workflows/test_workflow_completion_marker.py` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `workflow.infer` | `dpeva infer <config>`；推理结果与完成状态 | `tests/unit/workflows/test_infer_workflow_exec.py`; `tests/unit/workflows/test_workflow_completion_marker.py` | `docs/reports/2026-09-04-run-contract-pilot-report.md` |
| `workflow.feature` | `dpeva feature <config>`；特征文件与完成状态 | `tests/unit/workflows/test_feature_workflow_submission.py`; `tests/unit/workflows/test_feature_workflow_env.py` | `docs/reports/2026-09-04-run-contract-pilot-report.md` |
| `workflow.collect` | `dpeva collect <config>`；采集数据与完成状态 | `tests/unit/workflows/test_collection_workflow_submission.py`; `tests/unit/workflows/test_collect_workflow_routing.py` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `workflow.analysis` | `dpeva analysis <config>`；分析日志和统计产物 | `tests/unit/workflows/test_analysis_workflow.py` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `workflow.label` | `dpeva label <config>`；标注阶段和作业日志 | `tests/unit/workflows/test_labeling_workflow.py`; `tests/unit/workflows/test_slurm_logging.py` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `workflow.clean` | `dpeva clean <config>`；清洗后的数据集 | `tests/unit/workflows/test_data_cleaning_workflow.py` | `docs/reports/2026-09-04-integration-failure-classification.md` |
| `run.contract` | run manifest 的 identity、状态、事件、产物和失败语义 | `tests/unit/run/test_context.py`; `tests/unit/run/test_recorder.py`; `tests/integration/test_run_contract_pilot.py` | `docs/reports/2026-09-04-run-contract-pilot-report.md` |
| `lineage.evaluation-card` | 数据谱系、模型引用和六维 evaluation card | `tests/unit/run/test_dataset_lineage.py`; `tests/unit/run/test_model_ref.py`; `tests/unit/evaluation/test_card.py`; `tests/integration/test_evaluation_card_cli.py` | `docs/reports/2026-09-04-dataset-lineage-eval-card-acceptance.md` |
| `compatibility.deepmd-3.2` | DeepMD 3.2 CLI contract 与 capability evidence | `tests/unit/compatibility/test_deepmd_matrix.py`; `tests/unit/compatibility/test_deepmd_adapter.py`; `tests/contract/deepmd/test_cli_contract.py` | `docs/reports/2026-09-04-deepmd-3.2-compatibility.md` |

## 统一完成语义

工作流只有在进程/作业成功、声明产物通过验证且 run manifest 到达
`finished` 时才算完成。`DPEVA_TAG: WORKFLOW_FINISHED` 只是日志锚点，不能
单独证明成功；`sbatch` 返回 JobID 只建立 `submitted`，必须继续验证最终作业
状态和最小产物。

## 维护规则

- 契约测试必须对应注册表中的 `test_paths`，并保持与功能-文档矩阵的 ID 一致。
- 新 capability 先补可执行测试和现存证据，再登记路径；注册表门禁只验证 schema、路径和归属，不把路径存在误读为功能通过。
- integration、DeepMD 或 SAI 证据的成功与否由各自报告记录，不能由 unit gate 或本矩阵代替。
