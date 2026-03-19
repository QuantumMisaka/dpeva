# Tasks
- [x] Task 1: 建立性能观测基线并量化瓶颈
  - [x] SubTask 1.1: 在 analysis 主流程增加阶段级计时埋点（解析、组分加载、统计、绘图）
  - [x] SubTask 1.2: 基于 `config_analysis_val.json` 输出基线耗时摘要并确认主要慢点
  - [x] SubTask 1.3: 补充日志字段规范，确保后续优化可对比

- [x] Task 2: 修复 with_error 布局 warning 并保持图像质量
  - [x] SubTask 2.1: 调整 `plot_distribution_with_error` 的布局实现，避免 tight_layout 不兼容告警
  - [x] SubTask 2.2: 验证 Energy/Force/Virial/Cohesive 四类 with_error 图均可正常输出
  - [x] SubTask 2.3: 增加测试断言，确保日志不再出现该类 warning

- [x] Task 3: 引入 plot_level 分级与慢图告警机制
  - [x] SubTask 3.1: 在 AnalysisConfig 增加 `plot_level`（basic/full，默认full）与慢图阈值配置
  - [x] SubTask 3.2: 在 analysis manager 中实现 basic/full 两档出图路径并输出模式摘要
  - [x] SubTask 3.3: 为单图耗时超过 60s 的场景输出 warning 并提示切换 basic

- [x] Task 4: 完成回归验证与文档示例更新
  - [x] SubTask 4.1: 新增/更新单元测试覆盖配置解析、plot_level 分支与慢图告警
  - [x] SubTask 4.2: 使用目标配置回归运行并对比优化前后耗时与产物完整性
  - [x] SubTask 4.3: 更新示例配置与说明，明确 basic/full 模式选择建议

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 2
- Task 4 depends on Task 3
