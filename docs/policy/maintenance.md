# 文档版本管理与维护机制（Maintenance）

- Status: active
- Audience: Maintainers
- Last-Updated: 2026-02-18

## 1. 文档版本策略

- 面向用户的 Guide/Reference 默认随 `main` 分支演进。
- 重大接口变更（CLI/配置字段/目录结构）必须在同一 PR 内：
  - 更新相关文档
  - 更新示例（`examples/recipes`）
  - 更新/新增回归测试（unit 或 integration）
- 报告与归档文档默认冻结：新增通过“新文件”形式，不在旧报告中“覆盖式修改结论”。

## 2. Ownership（责任到人/模块）

建议按模块分配文档 Owner：

- CLI/配置总览：接口维护人（CLI Owner）
- Train/Infer/Feature/Collect：各 Workflow 维护人
- Slurm/集群适配：平台/运维协作负责人
- API Reference：配置模型维护人（Pydantic Model Owner）

Owner 可以是角色而非具体姓名；但每篇 `active` 文档必须有 Owner。

## 3. 变更触发规则（何时必须更新文档）

- 新增/修改 Pydantic 配置字段：必须更新 `docs/reference/*`
- 变更输出目录命名、日志文件名、完成标记：必须更新 Slurm 与集成测试相关文档
- 新增 CLI 子命令或更改参数：必须更新 CLI Guide 与 Quickstart

## 4. Review 流程（建议）

- 所有文档变更走 PR Review
- 文档 PR 至少 1 名 Owner Review
- 若变更涉及用户接口或配置字段，建议在 PR 描述中附“迁移说明”与“最小示例”

## 5. 文档过期治理（建议）

- 每季度（或每 2 个 release）做一次 docs 体检：
  - 断链扫描（链接可达性）
  - 过期字段扫描（例如 `num_selection` 这类已废弃字段）
  - 示例可运行性抽检（优先 Quickstart 与集成测试）
