# DP-EVA 集成测试最小配置模板（Slurm）

对应的可直接复制模板文件位于：[tests/integration/slurm_multidatapool/configs](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration/slurm_multidatapool/configs)。

本模板以“多数据池生产目录”语义为基准（`other_dpdata_all/`、`sampled_dpdata/`、`desc_pool/`、`desc_train/`、`0..3/`、`test_val/`），并尽量压缩到 Pydantic 模型要求的最小字段集。

## 1. 通用约定

- 所有路径推荐写为相对路径，并通过 CLI 自动做路径解析。
- Slurm 资源参数（partition/qos/gpu）在不同集群差异较大，模板仅保留最小字段；如需固定队列，请在 `submission.slurm_config` 内补齐 `partition/qos`。
- “完成标记”用于链式编排：Worker 成功结束时会输出 `DPEVA_TAG: WORKFLOW_FINISHED`（Train/Infer/Feature 已统一；Collect 需补齐该标记用于集成测试串联）。

## 2. Feature（候选池描述符）

- 模板：[feature_pool.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration/slurm_multidatapool/configs/feature_pool.json)
- 输入：`other_dpdata_all/` + `DPA-3.1-3M.pt`
- 输出：`desc_pool/`

## 3. Feature（训练集描述符）

- 模板：[feature_train.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration/slurm_multidatapool/configs/feature_train.json)
- 输入：`sampled_dpdata/` + `DPA-3.1-3M.pt`
- 输出：`desc_train/`

## 4. Train（Ensemble 微调）

- 模板：[train.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration/slurm_multidatapool/configs/train.json)
- 关键输入：
  - `input_json_path`：DeepMD-kit 训练输入（集成测试会用裁剪版 input.json）。
  - `training_data_path`：`sampled_dpdata/`
  - `base_model_path`：`DPA-3.1-3M.pt`
- 关键输出：`0..3/` 及 `train.out`（含完成标记）

## 5. Infer（候选池推理）

- 模板：[infer.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration/slurm_multidatapool/configs/infer.json)
- 输入：`other_dpdata_all/` + `0..3/model.ckpt.pt`
- 输出：`0..3/test_val/` 及 `test_job.log`（含完成标记）

## 6. Collect（普通采样）

- 模板：[collect_normal.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration/slurm_multidatapool/configs/collect_normal.json)
- 输入：
  - `desc_dir=desc_pool/`
  - `testdata_dir=other_dpdata_all/`
  - `testing_dir=test_val`（推理结果目录名）
- 输出：`root_savedir=dpeva_uq_result/`（csv/png/dpdata 导出）

## 7. Collect（联合采样）

- 模板：[collect_joint.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration/slurm_multidatapool/configs/collect_joint.json)
- 输入追加：
  - `training_data_dir=sampled_dpdata/`
  - `training_desc_dir=desc_train/`
- 输出：`root_savedir=dpeva_uq_result_joint/`

## 8. Analysis（推理结果统计）

- 模板：[analysis.json](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/tests/integration/slurm_multidatapool/configs/analysis.json)
- 输入：`result_dir=0/test_val`
- 输出：`analysis/`

