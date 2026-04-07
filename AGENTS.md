---
alwaysApply: false
description: DP-EVA 项目开发最小入口
---
# DP-EVA

DP-EVA 是面向 DPA 大模型高效微调的主动学习框架。主链路由训练、推理、特征、采集、标注、分析、清洗等独立工作流构成，目标是在尽量降低数据标注成本的前提下提升模型性能。

本文件只保留项目开发的最小入口。详细规范、门禁、生命周期与架构说明统一进入 `docs/guides/developer-guide.md`。

## Read First

- 项目总入口：`docs/README.md`
- 开发主手册：`docs/guides/developer-guide.md`
- CLI 契约：`docs/guides/cli.md`
- 配置与路径：`docs/guides/configuration.md`
- 文档治理快速上手：`docs/guides/docs-governance-quickstart.md`
- 配置模板：`examples/recipes/README.md`

## Key Signals

- 配置以 `src/dpeva/config.py` 与 Reference 为准
- 主链路以 `src/dpeva/cli.py -> src/dpeva/workflows/*.py` 为准
- 工作流可独立启动，但应共享底层模块，不在 workflow 层复制逻辑
- 对外契约变化必须在同一 PR 同步更新文档与 `examples/recipes/`
- 详细工程契约、文档生命周期与质量门禁统一进入 `docs/guides/developer-guide.md`
