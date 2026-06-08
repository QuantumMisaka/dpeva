---
title: 文档治理快速上手（开发者与 AI）
status: active
audience: Developers / Maintainers
last-updated: 2026-03-12
owner: Docs Owner
---

# 文档治理快速上手（开发者与 AI）

本文用于帮助人类开发者与 AI 开发者在首次参与 DP-EVA 贡献时，快速掌握文档治理最小闭环。

## 1. 先读哪里

- 贡献规则：`../policy/contributing.md`
- 维护机制：`../policy/maintenance.md`
- 质量标准：`../policy/quality.md`
- 治理总览：`../governance/README.md`
- Owner 责任矩阵：`../governance/inventory/owners-matrix.md`

## 2. 什么时候必须更新文档

出现以下变更时，必须在同一 PR 更新文档：

- CLI 子命令或参数变化
- `src/dpeva/config.py` 字段或约束变化
- 输出目录、日志文件名、完成标记变化
- `examples/recipes` 结构或关键示例变化

## 3. 提交前最低检查

```bash
python3 scripts/doc_check.py
python3 scripts/check_docs_freshness.py --days 90
make -C docs html SPHINXOPTS="-W --keep-going"
```

## 4. PR 必填治理信息

- 使用 `.github/PULL_REQUEST_TEMPLATE.md`
- 说明是否涉及对外契约变更
- 给出已更新的文档路径
- 若涉及治理文件，需通过 CODEOWNERS 审查

## 5. AI 开发者执行建议

- 先检索 `docs/guides`、`docs/policy`、`docs/governance` 的现有规则再改动
- 避免复制字段定义，始终链接到 `docs/reference/*`
- 修改文档后必须复跑治理检查并回写验证结果
- 优先补充导航入口与操作步骤，避免新增孤立文档

## 6. AI 草案与项目文档边界

- 执行期草案默认放 `.trae/documents/`，用于快速迭代与中间记录。
- 仅当文档达到“可共享、可复盘、可发布”状态，才迁入 `docs/plans/` 或 `docs/reports/`。
- 版本发布或任务闭环后，统一归档到 `docs/archive/vX.Y.Z/{plans,reports}/`。
