---
status: archived
audience: Historians
last-updated: 2026-03-13
owner: DP-EVA Developers
---

# Collection PCA Background Fix Plan

## 根因

- 主根因：`background_features` 在采样阶段直接进入 `PCA.transform`，未经过 `sampler.scaler.transform`。
- 伴生问题：`direct` 与 `2-direct` 两条路径都存在同类处理分叉，导致修复需双路径同步。
- 可观测性不足：当前缺少“背景预处理一致性”日志与测试断言，问题可静默回归。

## 实施步骤

1. 在 `SamplingManager._run_direct` 修复背景投影链路：先 `sampler.scaler.transform`，后 `sampler.pca.transform`。
2. 在 `SamplingManager._run_2_direct` 同步修复：先 `sampler.step1_sampler.scaler.transform`，后 `sampler.step1_sampler.pca.transform`。
3. 增加维度一致性校验：背景特征列数与采样训练特征列数不一致时直接失败并输出可定位错误。
4. 增加结构化日志：记录 `background_shape`、`feature_shape`、`scaler_applied`、`sampler_type`。
5. 补充单元测试：覆盖 `direct`/`2-direct` 背景链应用 scaler 的正向断言与维度失配负向断言。
6. 补充最小回归样例：固定随机种子，比较修复前后 `full_pca_features` 分布差异，确保行为变化可解释且稳定。

## 验证

- 静态验证：确认代码路径中背景链不再出现“直接 `pca.transform(background_features)`”调用。
- 单测验证：运行 `pytest tests/unit`，确认新增断言与既有测试全部通过。
- 定向回归：运行 Collection 相关测试，确认 `direct` 与 `2-direct` 均可生成稳定 PCA 图与采样输出。
- 负向验证：构造背景维度不一致输入，确认系统快速失败并输出可定位报错。
- 日志验证：确认输出 `scaler_applied=True` 及背景/主链特征维度信息。

## 回滚与风险

- 回滚策略：保留修复前实现快照；若发现兼容性问题，可临时回退到旧链路并保留新增日志与告警。
- 主要风险：修复后背景分布可视化会发生预期内变化，可能与历史图形不一致。
- 缓解措施：在发布说明中明确“背景 PCA 坐标系修正”，并附最小对照示例。
- 观测重点：关注 Collection 成功率、背景维度报错率、PCA 图形稳定性与用户侧解释一致性。
