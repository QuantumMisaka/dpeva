# Tasks
- [x] Task 1: 扩展Analysis配置与入口支持Slurm backend
  - [x] SubTask 1.1: 在配置模型中为Analysis接入统一submission配置
  - [x] SubTask 1.2: 调整CLI/Workflow初始化，保证analysis可拿到绝对配置路径用于自提交
  - [x] SubTask 1.3: 保持local backend兼容行为不变并补齐必要校验

- [x] Task 2: 在Analysis Workflow实现Slurm自提交执行链路
  - [x] SubTask 2.1: 复用现有submission组件生成`submit_analysis.slurm`
  - [x] SubTask 2.2: 构造analysis worker命令并注入内部backend覆盖防止递归提交
  - [x] SubTask 2.3: 完成slurm提交日志与错误处理，保持与Collection风格一致

- [x] Task 3: 更新Analysis recipes为Slurm backend示例并对齐Collection参数
  - [x] SubTask 3.1: 更新`examples/recipes/analysis`中的主案例为slurm backend版本
  - [x] SubTask 3.2: 对齐`submission.slurm_config`字段层级与参数命名
  - [x] SubTask 3.3: 校验示例路径与字段可被当前配置解析

- [x] Task 4: 补充单元测试覆盖Analysis Slurm路径并回归本地分支
  - [x] SubTask 4.1: 新增/更新workflow单测验证slurm提交调用与脚本内容
  - [x] SubTask 4.2: 断言关键参数透传、命令构造与内部backend覆盖逻辑
  - [x] SubTask 4.3: 运行相关单元测试并修复回归问题

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 2
