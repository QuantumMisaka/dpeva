---
title: Code Review - Collection PCA Background
status: archived
audience: Historians
last-updated: 2026-03-13
owner: AI Assistant
---

# 2026-03-13 Code Review: Collection PCA Background

## 1. 审查范围与结论

- 范围：`CollectionWorkflow` 在采样阶段将 `background_features` 投影到 PCA 可视化空间的处理链路。
- 结论：主根因是 `background_features` 在 `SamplingManager` 中直接进入 `PCA.transform`，未先经过与采样主链一致的 `sampler.scaler.transform`。这会导致背景点与候选点不在同一特征缩放空间，产生可视化偏移与覆盖度误判风险。

## 2. 现象（Phenomenon）

- 现象 P1：`Final_sampled_PCAview.png` 中灰色背景分布相对候选点出现整体拉伸/压缩或方向偏移，且在不同数据尺度任务间不稳定。
- 现象 P2：覆盖度曲线与肉眼观察的“背景包络”在部分任务中不一致，导致对采样代表性的解释出现偏差。

## 3. 证据链（Evidence Chain）

1. **DIRECT 主链使用的是“先标准化再 PCA”**  
   `DIRECTSampler` 定义为 `Pipeline([..., scaler, pca, ...])`，其中 `scaler=StandardScaler()`（`src/dpeva/sampling/direct.py`）。

2. **PCA 组件接口明确要求输入已归一化特征**  
   `PrincipalComponentAnalysis.transform` 的参数名与文档均为 `normalized_features`，语义上要求先做归一化（`src/dpeva/sampling/pca.py`）。

3. **候选样本通过 `sampler.fit_transform(features)` 走完整 pipeline**  
   在 `_run_direct` 中，`res = sampler.fit_transform(features)`，得到用于聚类/采样的 `PCAfeatures`（`src/dpeva/sampling/manager.py`）。

4. **背景样本未走 scaler，直接做 `pca.transform`**  
   `_run_direct` 中直接调用 `sampler.pca.transform(background_features)`；  
   `_run_2_direct` 中直接调用 `sampler.step1_sampler.pca.transform(background_features)`（`src/dpeva/sampling/manager.py`）。

5. **背景 `full_pca_features` 直接进入最终 PCA 图**  
   `plot_pca_analysis(..., full_features=full_pca_features)` 将其作为灰色背景层显示（`src/dpeva/uncertain/visualization.py`）。

6. **因此形成“候选点与背景点预处理不一致”闭环证据**  
   候选点：`scaler -> pca`；背景点：`pca only`。二者坐标系不一致，属于确定性的流程缺陷。

## 4. 根因（Root Cause）

- 根因 R1（主根因）：`background_features` 未经过 `sampler.scaler.transform`，直接输入 `PCA.transform`。
- 根因 R2（触发放大因子）：当描述符各维量纲差异较大时，跳过标准化会显著放大偏移。
- 根因 R3（可观测性不足）：当前缺少“背景预处理一致性”日志与断言，问题可长期静默存在。

## 5. 风险评估（Risk）

- 风险 K1（分析风险，高）：PCA 背景分布与候选分布不可直接比较，覆盖度与代表性分析可能偏离真实情况。
- 风险 K2（决策风险，中）：人工筛查图形时可能误判“候选已覆盖主空间”或“候选离群”。
- 风险 K3（回归风险，中）：`direct` 与 `2-direct` 两条路径均存在同类问题，若只修一处会残留隐患。

## 6. 验证结论（Validation）

1. **静态路径验证**  
   已确认主链与背景链调用路径存在分叉：主链经过 scaler，背景链未经过 scaler。

2. **接口语义验证**  
   `PrincipalComponentAnalysis.transform(normalized_features)` 与其文档语义一致，佐证背景链缺少归一化步骤并非设计预期。

3. **最小复现实验建议**  
   使用同一批 `background_features` 分别计算：  
   `sampler.pca.transform(background_features)` 与  
   `sampler.pca.transform(sampler.scaler.transform(background_features))`，  
   对比 PCA 坐标均值/方差与二维散点偏移量，预期存在显著差异。

4. **回归验证判据建议**  
   修复后应满足：  
   - 背景链与主链均显式记录 `scaler_applied=True`；  
   - `direct` 与 `2-direct` 的背景点分布在多数据尺度任务上更稳定；  
   - 现有采样输出文件数量与关键流程不受破坏。

## 7. 修复建议（Recommendations）

1. 在 `_run_direct` 中改为：  
   `bg_scaled = sampler.scaler.transform(background_features)` 后再 `sampler.pca.transform(bg_scaled)`。

2. 在 `_run_2_direct` 中改为：  
   `bg_scaled = sampler.step1_sampler.scaler.transform(background_features)` 后再 `sampler.step1_sampler.pca.transform(bg_scaled)`。

3. 增加输入校验：  
   背景特征维度需与 `features` 训练维度一致，否则抛出可定位异常。

4. 增加守护测试：  
   - 单测断言“背景链调用 scaler”；  
   - 单测断言修复前后行为差异（旧路径与新路径输出不等）；  
   - 参数化覆盖 `direct` 与 `2-direct` 两条路径。

5. 增加可观测性：  
   记录背景样本数量、维度、是否应用 scaler，以及 PCA 投影后统计摘要。

## 8. 审查结论摘要

- 本问题的主根因已收敛为“背景特征跳过标准化直接进 PCA 变换”，属于流程级实现缺陷。
- 该缺陷会系统性影响 PCA 背景解释质量，建议优先修复并加回归测试，随后再讨论背景口径语义增强。
