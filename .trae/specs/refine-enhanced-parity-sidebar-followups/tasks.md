# Tasks
- [x] Task 1: 审计 sidebar 跟进优化的现状与边界
  - [x] SubTask 1.1: 对照当前 cohesive energy 与 force 代表图确认新增问题点
  - [x] SubTask 1.2: 明确 scatter 几何扩展与 hexbin 图型替换的兼容边界
  - [x] SubTask 1.3: 固化本轮只改侧栏呈现、不改统计定义的范围

- [x] Task 2: 重构 scatter 的 Predicted Density 横向布局
  - [x] SubTask 2.1: 重新设计 Predicted Density 与 True Density 的宽度呼应关系
  - [x] SubTask 2.2: 保持右侧对齐线稳定并避免压缩主 parity 区域
  - [x] SubTask 2.3: 复核 cohesive energy 场景中的标题、刻度与留白

- [x] Task 3: 将 hexbin Error Distribution 切换为横置 violin
  - [x] SubTask 3.1: 选择并接入 matplotlib 或 seaborn 的横置 violin 实现
  - [x] SubTask 3.2: 保留 zero-error 参考线并统一 error 轴标题语义
  - [x] SubTask 3.3: 控制 violin 的填充、边界与透明度，避免盖过主图区

- [x] Task 4: 重构 hexbin 的 vertical color bar
  - [x] SubTask 4.1: 将 color bar 改为自下而上递增的 vertical 方向
  - [x] SubTask 4.2: 恢复与 hexbin 计数一致的色条文案与刻度语义
  - [x] SubTask 4.3: 允许独立微调 color bar 宽度与上下留白

- [x] Task 5: 补强测试并完成代表图回归
  - [x] SubTask 5.1: 增加 sidebar 几何、violin 与 vertical color bar 的单元测试
  - [x] SubTask 5.2: 运行 visualizer 与 visual_style 相关 pytest、ruff 校验
  - [x] SubTask 5.3: 基于 `/tmp/dpeva_parity_verify` 代表图路径复核最终观感

- [x] Task 6: 回填 spec 验收状态
  - [x] SubTask 6.1: 勾选已完成任务
  - [x] SubTask 6.2: 勾选 checklist 对应检查项
  - [x] SubTask 6.3: 记录与用户新增要求一一对应的验证结论

## Validation Notes
- `Predicted Density` 侧栏通过 `scatter_sidebar_width_scale` 与对齐逻辑放宽横向长度，单测覆盖 cohesive geometry 对齐且代表图已在 `/tmp/dpeva_parity_verify/parity_cohesive_energy_enhanced.png` 复核。
- Hexbin `Error` 侧栏已切换为横置 violin，并保留 `x=0` 零误差参考线；vertical color bar 同步改为 `Counts Per Hexbin` 语义，单测覆盖渲染路径、标签与几何宽度独立性。
- 验证结果：`pytest tests/unit/inference/test_visualizer.py tests/unit/utils/test_visual_style.py` 通过，`ruff check .` 通过，`pytest` 全量运行可复现仓库内既有 8 个非本次改动引入的失败用例。

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 1
- Task 5 depends on Task 2
- Task 5 depends on Task 3
- Task 5 depends on Task 4
- Task 6 depends on Task 5
