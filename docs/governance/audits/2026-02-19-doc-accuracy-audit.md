# 文档准确性审计报告（Docs Accuracy Audit）

- Status: active
- Audience: Maintainers / Developers
- Last-Updated: 2026-02-19
- Next-Audit: 2026-03-19

## 1. 审计范围

- 文档范围：`/docs/**/*`
- 代码范围：`/src/dpeva/**/*`、`/tests/**/*`
- 对外接口定义：
  - CLI：`dpeva` 子命令
  - 配置：Pydantic 配置模型与校验规则
  - 运行约定：完成标记、关键输出产物路径

## 2. 统计口径

- 总条目数：76
  - CLI 子命令：5
  - Pydantic 配置字段（模型字段并集）：60
  - `submission.slurm_config` 推荐键（约定层）：11
- 已验证条目数：76
- 差异条目数：0（已在本轮修复并复核）

## 3. 差异条目列表与修复记录

| 类别 | 发现问题 | 修复动作 | 影响 |
|---|---|---|---|
| Reference | `training_mode` 默认值与代码不一致 | 将默认值改为 `init` 并补齐 `template_path` 字段说明：`/docs/reference/config-schema.md` | 避免用户误以为默认继续训练 |
| Reference | `uq_trust_mode` 可选值与校验说明不一致 | 将枚举补齐 `no_filter` 并同步校验说明：`/docs/reference/config-schema.md`、`/docs/reference/validation.md` | 避免信任区筛选开关行为不明 |
| Docs | 项目内 Markdown 链接存在 `file:///...` 与目录相对路径 | 全量标准化为根相对路径（以 `/` 开头），并删除旧入口文档避免多入口分叉 | 提升可移植性与渲染一致性 |

## 4. 风险项（需进入 Issue 队列，可能阻塞发布）

- 运行时警告：Pydantic 报告 `model_path/model_head` 与 protected namespace 冲突（导入时可见）。建议创建 Issue 并评估是否通过 `model_config` 消除警告，避免 CI 噪音与潜在破坏性升级风险。

## 5. 可执行示例验证

本环境完成的验证：

- `pip install -e .` 可安装
- `import dpeva`、`import dpeva.config` 可导入

未在本环境执行的示例（原因）：

- 依赖 Slurm 队列与 DeepMD-kit `dp` 命令的端到端示例：需要在具备对应环境的 CI/集群上验证。

建议的 CI 落地：

- 新增文档示例抽取与验证脚本（跑通 `dpeva --help`、以及不依赖 Slurm/DeepMD 的轻量命令）
- 在具备 DeepMD/Slurm 的环境中单独启用 integration 示例验证（与现有 `tests/integration` 同步）

## 6. 追踪矩阵

- 功能-文档矩阵：/docs/governance/traceability/feature-doc-matrix.md

