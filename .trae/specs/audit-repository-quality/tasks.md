# Tasks
- [x] Task 1: 建立审查基线与证据清单，确认目录边界、现有质量约束、可用检查工具与输出模板。
- [x] Task 2: 审查核心源码与测试目录，覆盖代码风格、架构设计、安全、性能、测试完整性与覆盖率。
- [x] Task 3: 审查文档与项目元信息，覆盖 `docs`、`AGENTS.md`、`README.md`、`pyproject.toml` 的准确性、一致性与完整性。
- [x] Task 4: 审查维护脚本与技能脚本，覆盖 `.trae/skills` 与 `scripts` 的可维护性、健壮性、潜在风险与复用性。
- [x] Task 5: 审查 GitHub CI/CD 配置，检查流水线正确性、效率、缓存策略、门禁覆盖与安全性。
- [x] Task 6: 汇总全部发现，按严重级别分级，形成问题清单、修复建议、改进优先级与后续行动项。
- [x] Task 7: 按项目文档协议输出详细审查报告到 `docs/reports/`，并在需要时同步更新 `docs/source/` 索引。
- [x] Task 8: 依据 `checklist.md` 逐项复核审查交付物，确保范围、证据、分级与行动项完整。

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 1
- Task 5 depends on Task 1
- Task 6 depends on Task 2, Task 3, Task 4, Task 5
- Task 7 depends on Task 6
- Task 8 depends on Task 7
