# Tasks
- [x] Task 1: 明确多数据池总结图的输出契约
  - [x] SubTask 1.1: 定义新图命名、输出路径与图例信息（图外图注）
  - [x] SubTask 1.2: 确认仅在 joint sampling 条件下触发
- [x] Task 2: 以最小改动接入多数据池总结图出图逻辑
  - [x] SubTask 2.1: 复用现有 PCA 数据与绘图流程实现灰色全集背景+按池区分 sampled 点
  - [x] SubTask 2.2: 在 collect 主流程增加条件触发与日志记录
  - [x] SubTask 2.3: 保证单池模式行为完全兼容
- [x] Task 3: 补充验证与文档
  - [x] SubTask 3.1: 增加多池触发/单池不触发的单元测试
  - [x] SubTask 3.2: 更新 Collection 出图文档说明
  - [x] SubTask 3.3: 通过相关单测与文档门禁

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
