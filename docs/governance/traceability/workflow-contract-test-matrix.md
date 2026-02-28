# 工作流-契约测试矩阵（Workflow Contract ↔ Tests）

- Status: active
- Audience: Maintainers / Developers
- Last-Updated: 2026-02-21

本矩阵把“对外可观测契约”（产物路径、日志完成标记、失败定位入口）映射到“可执行测试”（unit / integration），用于：

- 发布前核对：文档契约是否仍被代码满足
- 测试建设：缺口优先级排序与用例补齐

## 1. 契约锚点（统一完成标记）

| 契约项 | 约定 | 代码实现 |
|---|---|---|
| 工作流完成锚点 | `DPEVA_TAG: WORKFLOW_FINISHED` | `/src/dpeva/constants.py` |

## 2. CLI 工作流契约（产物 + 标记 + 覆盖测试）

| 工作流 | 入口命令 | 最小产物（存在性断言） | 完成标记（日志） | Unit 覆盖 | Integration 覆盖 |
|---|---|---|---|---|---|
| Feature | `dpeva feature <cfg>` | `savedir/` 下 `.npy`（单池）或 `savedir/<pool>/` 下 `.npy`（多池） | `eval_desc.log`（或 slurm 输出日志）包含完成标记 | `tests/unit`（补齐：命令尾部 marker） | `tests/integration/test_slurm_multidatapool_e2e.py`（Feature-候选/训练集） |
| Train | `dpeva train <cfg>` | `work_dir/0..N-1/` + `model.ckpt.pt`（或等价模型产物） | `work_dir/<i>/train.out` 包含完成标记 | `tests/unit/workflows/test_train_workflow_init.py`（初始化/编排） +（补齐：脚本尾部 marker） | `tests/integration/test_slurm_multidatapool_e2e.py`（Train） |
| Infer | `dpeva infer <cfg>` | `work_dir/<i>/<task_name>/results.e.out`（或前缀等价输出） | `work_dir/<i>/<task_name>/test_job.log` 包含完成标记 | `tests/unit/workflows/test_infer_workflow_exec.py`（提交契约） +（补齐：命令尾部 marker） | `tests/integration/test_slurm_multidatapool_e2e.py`（Infer） |
| Collect | `dpeva collect <cfg>` | `root_savedir/dataframe/df_uq_desc_sampled-final.csv` | `collect_slurm.out`（slurm）或本地日志包含完成标记 | `tests/unit/workflows/test_collect_logging_fix.py`（校验/约束） +（补齐：完成标记日志） | `tests/integration/test_slurm_multidatapool_e2e.py`（Collect） |
| Analysis | `dpeva analysis <cfg>` | `output_dir/analysis.log` + 统计/图表文件（如 `metrics.json`） | 无统一标记约定（以 `analysis.log` 成功结束为准） | 建议补齐（解析/输出目录行为） | 未纳入 |

## 3. 分层测试建议（落地原则）

| 层级 | 目的 | 推荐断言粒度 |
|---|---|---|
| Unit（纯 Python） | 稳定覆盖“不可依赖外部环境”的核心逻辑 | 配置校验、路径解析、命令构造、完成标记拼接、关键异常分支 |
| Integration（依赖 Slurm/DeepMD） | 验证真实生产链路可跑通 | 最小产物存在 + 日志完成标记出现（避免依赖队列状态） |
