# Tasks
- [x] Task 1: 完成 Collection 低使用价值图像确认与分层清单
  - [x] SubTask 1.1: 基于现有审计结论定义 Core 与 Diagnostic 图清单
  - [x] SubTask 1.2: 明确每类图的生成前置条件与业务用途
- [x] Task 2: 实现 Collection 出图分层配置与默认策略切换
  - [x] SubTask 2.1: 增加配置项控制 Diagnostic 图开关
  - [x] SubTask 2.2: 在 collect 主链路按分层策略执行出图
  - [x] SubTask 2.3: 对跳过图像输出统一 reason 日志
- [x] Task 3: 完成测试与文档更新
  - [x] SubTask 3.1: 补充默认关闭与显式开启 Diagnostic 图测试
  - [x] SubTask 3.2: 更新 Collection 出图说明与操作指引
  - [x] SubTask 3.3: 运行并通过单测与文档门禁

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
