# Tasks

- [x] Task 1: 接入 Collection 含 label 场景的 Candidate parity 出图
  - [x] SubTask 1.1: 在 `collect.py` 的真值误差可用分支中接入 `plot_candidate_vs_error`
  - [x] SubTask 1.2: 复用现有 `_should_plot_force_error` 门控，保证触发条件一致
  - [x] SubTask 1.3: 完善日志，明确生成或跳过 Candidate parity 的原因

- [x] Task 2: 排查并治理 Collection 内“定义存在但未接线”出图逻辑
  - [x] SubTask 2.1: 枚举 `uncertain/visualization.py` 主要绘图函数与调用点
  - [x] SubTask 2.2: 对未接线项按“接入/保留手动/移除”决策并落地
  - [x] SubTask 2.3: 补充对应测试或说明，避免再次失联

- [x] Task 3: 全面审查七类工作流的未接入模块
  - [x] SubTask 3.1: 建立 `train/infer/analysis/feature/collect/label/clean` 入口到实现映射
  - [x] SubTask 3.2: 扫描并识别“模块定义存在但无标准链路调用”的候选项
  - [x] SubTask 3.3: 输出结构化审计结果（状态、证据、建议、优先级）

- [x] Task 4: 增补回归测试与质量门禁
  - [x] SubTask 4.1: 为 Collection 新增/更新 workflow 级测试，断言两张 Candidate parity 图的触发行为
  - [x] SubTask 4.2: 为无 GT/异常 GT 数据路径增加“应跳过”断言
  - [x] SubTask 4.3: 运行相关单元测试并修复回归问题

- [x] Task 5: 交付审查结论与文档同步
  - [x] SubTask 5.1: 在 `docs/reports/` 补充本次未接入模块审查报告
  - [x] SubTask 5.2: 更新既有 Collection 出图审计报告中的可生成性状态
  - [x] SubTask 5.3: 完成文档治理检查与引用完整性检查

# Task Dependencies
- Task 2 depends on Task 1
- Task 4 depends on Task 1
- Task 5 depends on Task 2
- Task 5 depends on Task 3
- Task 5 depends on Task 4
