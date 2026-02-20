# DP-EVA 集成测试最小配置模板（Slurm）

- Status: active
- Audience: Developers / Infra
- Last-Updated: 2026-02-18

对应可直接复制的模板文件位于：

- [tests/integration/slurm_multidatapool/configs](/tests/integration/slurm_multidatapool/configs)

本模板以“多数据池生产目录”语义为基准（`other_dpdata_all/`、`sampled_dpdata/`、`desc_pool/`、`desc_train/`、`0..3/`、`test_val/`），并尽量压缩到 Pydantic 模型要求的最小字段集。

## 0. 相关方

- 开发者：维护配置字段、模板与测试链路
- 平台维护：提供 Slurm 队列策略（partition/qos/account）
- 使用者：复制模板并按集群参数与目录结构运行

## 1. 通用约定

- 所有路径推荐写为相对路径，并通过 CLI 自动做路径解析。
- Slurm 资源参数（partition/qos/gpu）在不同集群差异较大，模板仅保留最小字段；如需固定队列，请在 `submission.slurm_config` 内补齐 `partition/qos`。
- 完成标记用于链式编排：Worker 成功结束时会输出 `DPEVA_TAG: WORKFLOW_FINISHED`。

## 2. Feature（候选池描述符）

- 模板：`feature_pool.json`
- 输入：`other_dpdata_all/` + `DPA-3.1-3M.pt`
- 输出：`desc_pool/`

## 3. Feature（训练集描述符）

- 模板：`feature_train.json`
- 输入：`sampled_dpdata/` + `DPA-3.1-3M.pt`
- 输出：`desc_train/`

## 4. Train（Ensemble 微调）

- 模板：`train.json`
- 关键输入：
  - `input_json_path`：DeepMD-kit 训练输入（集成测试会用裁剪版 input.json）
  - `training_data_path`：`sampled_dpdata/`
  - `base_model_path`：`DPA-3.1-3M.pt`
- 关键输出：`0..3/` 及 `train.out`（含完成标记）

## 5. Infer（候选池推理）

- 模板：`infer.json`
- 输入：`other_dpdata_all/` + `0..3/model.ckpt.pt`
- 输出：`0..3/test_val/` 及 `test_job.log`（含完成标记）

## 6. Collect（普通采样）

- 模板：`collect_normal.json`
- 输入：
  - `desc_dir=desc_pool/`
  - `testdata_dir=other_dpdata_all/`
  - `testing_dir=test_val`（推理结果目录名）
- 输出：`root_savedir=dpeva_uq_result/`（csv/png/dpdata 导出）

## 7. Collect（联合采样）

- 模板：`collect_joint.json`
- 输入追加：
  - `training_data_dir=sampled_dpdata/`
  - `training_desc_dir=desc_train/`
- 输出：`root_savedir=dpeva_uq_result_joint/`

## 8. Analysis（推理结果统计）

- 模板：`analysis.json`
- 输入：`result_dir=0/test_val`
- 输出：`analysis/`

## 9. 异常处理

- 模板字段校验失败：优先对照 `docs/reference/*` 与校验规则定位缺失/类型错误。
- Slurm 资源字段不兼容：在 `slurm_config` 中补齐集群要求的字段，并调整 GPU/CPU 申请。
- 输出路径不一致：确认 `work_dir/project/savedir/root_savedir` 与提交目录一致，并检查权限/配额。

## 10. 变更记录

- 2026-02-18：建立集成测试最小配置模板索引与目录/产物约定，作为测试专题的可复用资产入口。
