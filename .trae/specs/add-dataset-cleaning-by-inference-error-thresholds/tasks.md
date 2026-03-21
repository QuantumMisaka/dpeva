# Tasks
- [x] Task 1: 设计并接入 clean CLI 子命令
  - [x] SubTask 1.1: 在 `cli.py` 新增 `handle_clean` 执行入口
  - [x] SubTask 1.2: 新增 `dpeva clean <config>` 子命令解析
  - [x] SubTask 1.3: 对齐现有 CLI 错误处理与日志行为

- [x] Task 2: 扩展配置模型与默认值常量
  - [x] SubTask 2.1: 在 `config.py` 新增 `DataCleaningConfig`
  - [x] SubTask 2.2: 在 `constants.py` 新增 clean 默认值常量
  - [x] SubTask 2.3: 明确阈值单位与未配置行为文档描述

- [x] Task 3: 新增独立清洗 workflow 与模块分层
  - [x] SubTask 3.1: 新增 `workflows/data_cleaning.py` 与 `DataCleaningWorkflow`
  - [x] SubTask 3.2: 设计并实现加载指标、构建掩码、导出、统计的分层方法
  - [x] SubTask 3.3: 更新 `workflows/__init__.py` 暴露新 workflow

- [x] Task 4: 建立帧级对齐校验与阈值判定逻辑
  - [x] SubTask 4.1: 校验系统名+帧号映射一致性
  - [x] SubTask 4.2: 对缺帧、重复、乱序提供 fail-fast 异常
  - [x] SubTask 4.3: 实现“任一启用指标超阈值即剔除”规则与全未配置旁路

- [x] Task 5: 集成导出统计与用户案例入口
  - [x] SubTask 5.1: 导出清洗后 dpdata 并输出统计文件
  - [x] SubTask 5.2: 新增 `examples/recipes/data_cleaning` 用户示例配置
  - [x] SubTask 5.3: 更新 `examples/recipes/README.md` 的 clean 入口说明

- [x] Task 6: 补齐单元测试与回归验证
  - [x] SubTask 6.1: 覆盖阈值全启用、部分启用、全未配置路径
  - [x] SubTask 6.2: 覆盖对齐失败与缺失结果文件异常路径
  - [x] SubTask 6.3: 运行相关单测并修复回归问题

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 4
- Task 6 depends on Task 1, Task 2, Task 3, Task 4, and Task 5
