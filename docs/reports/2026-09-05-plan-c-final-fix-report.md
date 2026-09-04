---
title: Plan C Final Review Fix Report
status: ready-for-parent-review
audience: Project Maintainers / Scientific Owner
last-updated: 2026-09-05
owner: Scientific Owner
---

# Plan C 终审修订记录

本轮修订仅收敛 Dataset Lineage 与 Evaluation Card 的证据边界；没有引入 campaign database、科学
排名或 Phase 3 算法。

## 已处理终审意见

- R4：`DatasetManifest` 现在要求 `validation_result`（rule version、status、counts、sources、
  intersections、type-map 结果）以及结构化 `intersection_summary`。来源未声明、计数不闭合、
  重复/交集没有解释、或 type-map 冲突均失败。显式坐标签名去重保留 overlap/removal evidence；
  不启用去重时发现重复会在导出前失败。
- 评估指标：递归检查 dict/list 中的所有 float，覆盖 `NaN`、`Infinity`、`-Infinity` 和合法 JSON
  数值溢出 `1e999`；坏证据映射为 `failed` 且保留其 evidence reference，其他维度继续组装。
- 可移植引用：读取阶段使用解析后的本地路径，写卡阶段将 model、dataset、metric evidence 和
  本地下游反馈写成相对卡片目录的 POSIX 引用；URI 原样保留。相对路径只提供可搬迁定位，不被
  当作不可变性机制；不可变性仍由引用目标、校验和及验证契约承担。
- 大数据 bundle：保留 sibling staging、进程可见原子发布和 Linux `renameat2(RENAME_NOREPLACE)`
  竞争不覆盖语义；移除不完整且高成本的目录/staging crash durability 声称，不递归 fsync dpdata
  树。JSON 原子写入中的文件级 flush 不外推为整个 bundle 的 durability 保证。
- Recipe：明确 `config_eval_card.json` 是需填充真实 artifact 的模板；CLI 文档给出 filled config
  的形式，不再把模板呈现为可直接运行的候选交接。

## Ruling / 偏差

- Ruling：validation rule version 固定为 `1.0`；validation result 是持久化的最小闭环结果，
  不引入独立 validation engine。
- Ruling：交集证据采用真实 dpdata `System.sub_system()` 拆出的逐帧 identity，identity 覆盖
  canonical coords/cells、PBC/nopbc、atom types/type map/names 和 shape；标签字段另行纳入
  label identity。方法版本为 `frame-identity-v1`，不宣称逐帧全局 hash 或科学等价性。
- Ruling：整合发布不再报告 `PublicationDurabilityError`；失败边界是 export/manifest/summary
  写入失败、目标已存在、或平台不支持 `renameat2(RENAME_NOREPLACE)`。
- 偏差：candidate package 的搬迁前提是被引用的 evidence artifacts 与 card 按原相对布局共同搬迁；
  本命令不复制外部证据，也不伪造 checksum。

## Verification

在隔离 worktree `/home/james/work/ft2dp-dpeva/dpeva/.worktrees/governance-deepmd-32`、环境
`dpeva-dpa4` 中：

```text
pytest tests/unit/run/test_dataset_lineage.py tests/unit/evaluation/test_card.py tests/unit/labeling/test_integration.py tests/integration/test_evaluation_card_cli.py tests/integration/test_e2e_cycle.py -q
77 passed（含真实 dpdata 帧 identity/冲突测试与 relocation 解析）
pytest tests/unit -q
748 passed
pytest tests/integration/test_e2e_cycle.py tests/integration/test_evaluation_card_cli.py -q
7 passed
pytest tests/integration -q
41 passed, 7 skipped
python -c "... EvaluationCardConfig.model_validate(...) ..."
exit 0
python scripts/doc_check.py
PASSED
ruff check src tests scripts
All checks passed!
git diff --check
exit 0
```

Plan C 的新增验收边界包括 12,105 + 4,317 = 16,422、显式去重、未解释重复失败、持久化
validation/intersection evidence、递归非有限值拒绝、失败 metric 保留 evidence、相对引用和
candidate package 目录搬迁后的解析契约。
