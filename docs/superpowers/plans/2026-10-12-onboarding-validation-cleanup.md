---
title: Onboarding and development validation cleanup
status: completed
audience: Developers / Maintainers
last-updated: 2026-10-12
owner: Project Maintainer
---

# 新人示例与开发验证收敛

来源：维护者于 2026-10-12 接受入职文档审查，并要求将可复用内容沉淀到 unittest 和相对路径 examples，随后有序删除重复 smoke/验收资产。

基线：`a4f55ff`。本计划仅记录本轮迁移与验证，不建立新的验收框架。

## 决定与边界

- 用户路径收敛为一份 `examples/recipes/quickstart/` Local 示例；JSON 路径相对配置文件，文档只解释准备、执行和产物，不复制另一套配置。
- 测试以行为为准：在既有 owning suite 承接有价值回归，删除只锁历史报告文字或重复验证的检查。保留真实 subprocess、Slurm、DeepMD contract 等跨边界测试，不把它们全部降格为 mock 单测。
- 删除先确认承接测试/示例及活跃引用；移出的文件保存在 `~/scratch`，Git 历史仍可追溯。冻结科学资格证据及有效 capability manifest 不在清理范围。
- 修复教程路径、作业内环境脚本路径、从零安装后端说明和版本文档站内链接。仅在有依据时调整实现，不扩展科学能力、不训练、不提交集群作业、不移动发布 tag。

## 工作包

- [x] 示例：建立唯一可运行的四阶段 Local recipe；复用已跟踪夹具；以轻量配置加载和真实路径契约回归验证，避免新增烟测执行器。
- [x] 开发验证：盘点临时生成器、孤立采样测试、绘图验收脚本、一次性硬件比较包装脚本及历史版本验收；将有效行为合并到相应测试，移除重复资产并更新活跃引用。
- [x] 文档：将 smoke/config-template 重复内容收敛到 quickstart + examples；修复 SAI 加载路径和环境准备，恢复同版本站内导航；明确贡献与测试入口。
- [x] 集成：检查迁移映射、净变更和剩余引用；运行相关测试、完整 unit/integration 与文档门禁；独立终审，处理有效问题。

## 验收

1. 从仓库外加载发布的示例 JSON，输入可定位、输出落到约定工作区；无需运行真实训练。
2. SAI 环境加载说明不依赖任务 cwd；干净环境有明确 CPU/PT 安装入口，预置集群环境边界清楚。
3. 被删除文件的有效行为已有承接，引用不存在悬空；不新增另一套 smoke/验收 runner。
4. 相关门禁通过，真实硬件验证未运行的限制如实记录。

## 迁移及验证记录

备份位置：`/home/james/scratch/dpeva-validation-cleanup-20261012/`，保留原相对目录。

| 移出资产 | 承接或保留边界 |
| --- | --- |
| `docs/guides/smoke-demo.md`、`testing/integration-config-templates.md` | Quickstart 引用唯一 `examples/recipes/quickstart/` 配置；不再复制 JSON |
| `tests/integration/data/input.json` | 唯一训练模板为 quickstart/input.json；真实 E2E 仍通过 helper 覆盖所需步数与数据 |
| `tests/test_birch_clustering.py` | 空输入、样本不足、warning 处理进入 `tests/unit/sampling/test_clustering_full.py` |
| `tests/integration/test_v080_atst_acceptance.py` | 真实 CLI/stub backend 进入 `test_exploration_cli.py`；label prepare 进入 `test_e2e_cycle.py`；删除历史报告文字断言 |
| `tests/gen_test_data.py`、`tests/integration/setup_data.py` | 使用已跟踪小夹具与既有 data_minimizer，不再维护一次性数据生成器 |
| `scripts/collection_plot_regression.py` | 既有 visualization、collection_io 测试覆盖当前绘图与导出行为 |
| `scripts/check_docs.py`、`scripts/verify_docs.sh` | 统一 `scripts/run_gate.py` 与 gates.toml，移除重复入口的文字断言 |
| `scripts/validation/compare_eval_desc_embed.py`、三个对应 Slurm 包装脚本、`scripts/env/dpeva-dpa4-embed-test.env` | 移除依赖旧环境/私有路径的一次性比较工具；冻结数值比较报告与 Git 基线保留，不声称由现行资格验证替代其数值等价证据 |

### 实施裁决

- 显式 `collect.project` 原来遗漏统一路径解析，现与其他路径字段一样相对配置文件解析。已在 configuration/CLI 中说明迁移；未给定默认值及直接 Python 对象不变。新增回归先观察失败，再验证修复。
- 示例测试从仓库外读取原样复制的真实 JSON，检查输入、输出及四阶段连接，使用临时工作目录；不创建另一套执行器。Python CI 路径触发范围包含 recipes，保证示例修改触发相应测试。
- 本轮初次运行旧集成测试时，因环境具有真实 Torch，Local 用例曾进入 Feature 子进程；发现后中断。检查未产生 descriptor、训练日志或 checkpoint，也无残留计算进程，未提交 Slurm。为消除此默认启动风险，Local 真实训练测试增加 `DPEVA_RUN_LOCAL_ITEST=1` 显式开关；Slurm 原开关保留。后续统一门禁不启用真实计算。
- 历史发布/科学报告的旧文件名称保持历史事实，当前指南与测试入口已迁移。DeepMD qualification 与科学证据未改。


### 统一验证（2026-10-12）

在隔离工作树 `refactor/onboarding-validation-cleanup`，使用 `dpeva-dpa4` 环境及 `PYTHONPATH=src:.`：

- `python scripts/run_gate.py local`：exit 0；Ruff / audit 通过；983 passed、3 skipped，branch coverage 83.64%（阈值 80%）。
- `python scripts/run_gate.py integration`：exit 0；46 passed、8 skipped；跳过包含显式硬件/真实训练开关和未提供的外部回归数据。
- `python scripts/run_gate.py docs_pr`：exit 0；文档审计、时效检查、Sphinx 警告即错误构建、产物检查、链接检查、traceability 均通过。
- `git diff --check`：通过。原始日志在 `/tmp/dpeva-cleanup-{local,integration,docs_pr}.log`。
- 验证范围为配置连接、常规测试和文档；本轮未完成真实四阶段训练与集群端到端复跑，不据此新增科学能力声明。

独立终审：`gpt_final_reviewer` 已核对全部 tracked/untracked 改动及上述原始日志，未发现 P1/P2 阻断问题。清理实施完成；独立终审时工作树保留，尚未提交或合入。全新 CPU 环境安装与实际四阶段运行未复验。

### 本地合入授权（2026-10-12）

维护者确认验证通过后合入本地 main。合入前 main 仍为干净基线 a4f55ff，复用以上未变代码的验证与独立审查；同时移除 CODEOWNERS 对已删除 verify_docs.sh 的残留。提交后执行 fast-forward 合入并核对提交一致性及重点路径回归。远端发布/推送及分支保护配置不在本次操作范围。
