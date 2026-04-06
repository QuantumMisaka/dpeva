# Tasks
- [x] Task 1: 建立审阅基线与证据地图，明确项目目标、治理规范、核心入口与审阅边界。
  - [x] SubTask 1.1: 盘点 `docs/` 中与项目定位、开发规范、文档治理、架构设计相关的权威文档
  - [x] SubTask 1.2: 盘点 `examples/recipes/` 中各工作流对外契约、最小配置与示例覆盖情况
  - [x] SubTask 1.3: 盘点 `src/` 中 CLI、配置中心、workflow、manager 与共享模块的主链路实现

- [x] Task 2: 输出项目目标意义与治理规范落实度审阅结论，形成详细审阅文档。
  - [x] SubTask 2.1: 归纳项目目标、使用场景、设计哲学与工程治理重点
  - [x] SubTask 2.2: 逐项核查文档规范在代码与 recipes 中的落实状态，并标注“已落实/部分落实/未落实”
  - [x] SubTask 2.3: 识别文档与代码、文档与 recipes、代码与 recipes 之间的偏差与缺口
  - [x] SubTask 2.4: 在 `docs/reports/` 生成符合命名规范的详细审阅报告，并在需要时更新索引
 
- [x] Task 3: 审查并改善项目文档系统，为后续 `AGENTS.md` 改造打基础。
  - [x] SubTask 3.1: 修复 `docs/` 中与核心代码、CLI 契约、recipes 示例不一致的内容
  - [x] SubTask 3.2: 盘点冗余、错漏、过时或可删除的文档文件，并明确保留、合并、归档或删除建议
  - [x] SubTask 3.3: 按项目文档协议完成必要的文档更新、索引同步与治理说明

- [x] Task 4: 提炼代码中已存在但文档未充分表达的工程化实践。
  - [x] SubTask 4.1: 提炼配置与默认值中心化、路径统一解析、CLI 前置校验等实践
  - [x] SubTask 4.2: 提炼工作流独立可执行、阶段化入口、共享底层模块复用、完成标记等实践
  - [x] SubTask 4.3: 为每项实践给出建议的文档落点，判断应进入 `AGENTS.md` 摘要还是进入 `docs/` 详细文档

- [x] Task 5: 制定 `AGENTS.md` 精简改造方案，并联动提出必要的文档系统改进建议。
  - [x] SubTask 5.1: 设计 `AGENTS.md` 的目标读者、信息层级与目录结构
  - [x] SubTask 5.2: 明确保留、删减、改写与新增的模块，避免重复 AI 规则与既有开发规范正文
  - [x] SubTask 5.3: 基于前序文档治理结果，补充仍然需要的 `docs/` 联动改进建议
  - [x] SubTask 5.4: 形成可执行的 AGENTS 改造方案文档或方案章节，并与审阅结论建立引用关系

- [x] Task 6: 依据清单复核交付物完整性，确保结论可执行且与项目文档协议一致。
  - [x] SubTask 6.1: 逐项核验审阅范围、证据、文档治理动作、问题分类、AGENTS 方案与文档联动建议
  - [x] SubTask 6.2: 检查新增或更新文档是否满足命名、归档、索引与可追溯性要求

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 1, Task 3
- Task 5 depends on Task 2, Task 3, Task 4
- Task 6 depends on Task 2, Task 3, Task 4, Task 5
