# Tasks
- [x] Task 1: 完成 Collection 作图链路代码审查并输出报告到 docs/reports。
  - [x] SubTask 1.1: 审查 collect.py、sampling/manager.py、uncertain/visualization.py 的数据流
  - [x] SubTask 1.2: 复核 All Data in Pool 与Candidate 的坐标变换一致性
  - [x] SubTask 1.3: 形成证据链、根因结论与风险评估并写入审查文档

- [x] Task 2: 设计并输出可回溯修复计划到 docs/plans。
  - [x] SubTask 2.1: 给出最小修复方案与可选增强方案
  - [x] SubTask 2.2: 定义验收标准、验证路径与失败回滚策略
  - [x] SubTask 2.3: 标注依赖关系与执行顺序，等待用户确认

- [x] Task 3: 按确认后的方案修复背景点 PCA 投影不一致问题。
  - [x] SubTask 3.1: 在 SamplingManager 中统一背景数据标准化与 PCA 投影链路
  - [x] SubTask 3.2: 如适用同步修复 2-DIRECT 对应路径
  - [x] SubTask 3.3: 保持可视化接口与输出文件命名兼容

- [x] Task 4: 增加验证并完成文档索引一致性检查。
  - [x] SubTask 4.1: 增加或更新测试，覆盖背景点与候选点坐标一致性
  - [x] SubTask 4.2: 运行单元测试与必要静态检查
  - [x] SubTask 4.3: 更新 docs/source 下相关 toctree，确保无悬挂引用

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
