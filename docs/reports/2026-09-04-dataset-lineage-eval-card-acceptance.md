---
title: Dataset Lineage and Evaluation Card Phase 2A Acceptance
status: accepted
audience: Developers / Scientific Owner
last-updated: 2026-09-05
owner: Scientific Owner
---

# Dataset Lineage and Evaluation Card Phase 2A 验收记录

## 结论

Plan C 的 Phase 2A 验收门通过。当前证据证明的是数据谱系记录、模型引用和候选评估卡片的证据管线可用；本报告不把缺失评测当作零值，也不把管线通过解释为模型科学质量通过。

## Fresh gate 证据

以下命令均在隔离工作树 `/home/james/work/ft2dp-dpeva/dpeva/.worktrees/governance-deepmd-32`、`dpeva-dpa4` 环境中重新执行：

```text
conda run -n dpeva-dpa4 ruff check src tests scripts
All checks passed!

conda run -n dpeva-dpa4 pytest tests/unit -q
735 passed in 25.71s

conda run -n dpeva-dpa4 pytest tests/integration/test_e2e_cycle.py tests/integration/test_evaluation_card_cli.py -q
7 passed in 3.39s

conda run -n dpeva-dpa4 python -c "import json; from dpeva.config import EvaluationCardConfig; EvaluationCardConfig.model_validate(json.load(open('examples/recipes/evaluation/config_eval_card.json')))"
exit 0

conda run -n dpeva-dpa4 pytest tests/integration/test_evaluation_card_cli.py -q
5 passed in 0.19s

git diff --check
exit 0
```

## 已交付 artifact 与 schema

- 数据谱系模型与计数守恒校验：`src/dpeva/run/dataset.py`，`DatasetManifest` / `DatasetParent` schema `1.0`。12,105 + 4,317 = 16,422 的回归边界已由 unit 测试覆盖。
- 标注整合输出：调用方指定的 `<merged_training_data_path>/` 下包含导出的数据、当前代兼容指针 `dataset-manifest.json`、不可变的 `dataset-manifest-<generation>.json` 和 `integration_summary.json`。summary 返回 `dataset_manifest_path`，并记录生成代与 SHA-256。
- 模型证据引用：`src/dpeva/run/model.py`，`ModelArtifactRef` schema `1.0`；明确区分 checkpoint/frozen/exportable/pretrained-alias 及 regular/EMA 角色。
- 候选评估卡片：`src/dpeva/evaluation/card.py`，`EvaluationCard` schema `1.0`，固定六个 metrics 维度；通过 `src/dpeva/cli.py` 的 `dpeva eval-card CONFIG.json` 生成调用方配置的 `evaluation-card.json`。输出采用不可覆盖发布语义。
- 可移植 recipe：`examples/recipes/evaluation/config_eval_card.json`；不包含 campaign-local 绝对路径或 FT2DP 任务状态。
- 集成测试中的具体 evidence fixture 路径为相对测试临时根目录的 `evidence/model-ref.json`、`evidence/surface.json`，生成输出为 `artifacts/evaluation-card.json`；测试验证配置相对路径解析和六维卡片生成。

## 显式缺失维度

本次 eval-card 集成 fixture 只提供 `surface_slice` 证据。因此以下五个维度明确为 `not-run`，其 `value` 保持 `null`，不是数值零：

- `in_domain_cumulative`
- `iter11_last_wave`
- `historical_domain`
- `matpes_retention`
- `training_cost`

配置了但不存在、格式错误或包含非有限 JSON 数值的证据会标记为 `failed` 并保留 `evidence_ref`；这不是本次 fixture 的失败。

## 已知限制与边界

- Phase 2A 只组装已有证据，不执行训练、推理、评测或科学排名。
- 当前没有断言候选模型在 in-domain、历史域、MatPES、surface 或 Fischer–Tropsch 下的数值性能。
- 未提供下游审查输入时，`downstream_feedback_ref` 只作为引用保留，不会被自动生成或推断。
- 本计划不引入 campaign database、自动科学排序、调度器编排或 Phase 3 算法。
- 整合清单的逻辑 parent 目前是 `existing-training` / `new-labeled`；父清单引用的进一步固化留待后续批准的契约变更。

This gate validates evidence plumbing, not scientific superiority or downstream Fischer–Tropsch acceptance.
