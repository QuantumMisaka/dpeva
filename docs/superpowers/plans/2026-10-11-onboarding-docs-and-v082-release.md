---
title: Onboarding Docs & v0.8.2 Release Implementation Plan
status: active
audience: Developers / AI Agents
last-updated: 2026-10-11
owner: Project Maintainer
---

# 入职文档补丁与 v0.8.2 发布 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ①补齐社区新用户（特别是 SAI 与通用 Slurm 用户）的入职入口：模型与数据获取指引、自带夹具的冒烟演示、SAI 快速上手、README 文档站入口；②修复阻断发布路径的既有 freshness 红状态；③把已完成的 0.8.2 兼容性收尾落成正式版本（annotated tag + GitHub Release）；④全程不改代码与能力边界，为 2026-10-14 DeepModeling 社区研讨会提供稳定引用点。

**Requirement source:** 维护者（刘照清）2026-10-11 指示：“P0仓库补丁和 v0.8.2 release 请你派发给 dpeva 开发者”。本计划确认开发目标、内容与验收模式；发布与补丁均在此授权内；Task 4 的前置修复在本机取证（2026-10-11）后列为发布路径必备项。

**Architecture:** 只做文档层改动（README、docs/guides、docs/README 索引、tests/integration/data/README.md 更正、docs/reports 发布记录、本计划归档）与 Git tag / GitHub Release 发布动作。不改动 `src/**`、`pyproject.toml` 与 `src/dpeva/compatibility/*.json`。新页面沿用 docs front-matter（status/audience/last-updated/owner）与现有 guides 风格，中文为主、英文术语保留；直接提交 main（与近期实践一致），远端以 CI 为权威。

**Tech Stack:** Markdown/Sphinx 文档体系、`scripts/run_gate.py` 门禁、GitHub Actions、GitHub CLI（`gh`，已登录 QuantumMisaka）、git tag。

## Global Constraints（硬性边界）

- 仅允许文档与发布改动；`src/**`、`pyproject.toml`、`src/dpeva/compatibility/*.json`、`tests/**`（唯一例外 `tests/integration/data/README.md` 的更正）均不得改动；`docs/governance/**` 既有记录不得改动（唯一例外为 Task 4 清单内的 `docs/governance/tools/unwired-module-audit-template.md` 复核刷新）。
- 不 bump 版本（保持 0.8.2）；不新增能力声明；不启动新训练、不提交集群作业。
- freshness 修复只做“逐份复核 + `last-updated` 刷新”级别；不得把 living 文档改成 `status: record` 来规避审查，不得用机械批量方式改写内容。
- 不 force push、不重写历史、不删除分支与 tag；不动其他仓库，不涉及 personal 仓库与飞书文档。
- 仓库可能有并行会话：开工先 `git status --short --branch` 与 `git log --oneline -5`；发现非预期已修改文件先停并报告，不回退他人修改。
- 事实与 URL 以本计划为准；超出本计划的不确定性先停、报告、等裁决，不自行发明。
- 门禁失败时不得把失败提交推送到 main；不得在 CI 红的状态下打 tag。

## 关键决策（已拍板，执行相不回摆）

- **D1 发布点** = 本文档补丁落地后的 main HEAD（基线 `5b0294a`，version=0.8.2；证据：`docs/reports/2026-09-05-v082-compatibility-closeout.md`、远端 tag 仅至 v0.8.1）。若发现仓库规定明确指向另一发布点，以规定为准并在报告中说明。
- **D2 顺序** = 先文档补丁与前置修复（Task 1–6），后发布（Task 8），使 release 包含入职材料且发布路径全绿。
- **D3 回退** = 若 2026-10-13 前文档补丁仍无法绿色落地，将补丁剥离出发布点，直接发布当前 HEAD 并报告。
- **D4 体例** = 不整篇重写既有文档，只做增补与修正；新页面中文为主。
- **D5 发布路径前置** = `Deploy Docs` 的 release-validation 运行完整 `release` profile（含 `docs_freshness`）；2026-10-11 取证：freshness 因 10 个 living 文档超期而失败（exit=1），定时治理任务 2026-09-28/10-05 亦因此失败。**Task 4 必须先行修复，否则 main/tag 的文档部署均会被跳过**。

## Tasks

### Task 0: 基线核对与门禁了解

- [ ] `git status --short --branch`、`git log --oneline -5`；确认 HEAD=`5b0294a` 且与 origin/main 同步；确认本文件是预期中的未提交文件。
- [ ] 读 `docs/guides/docs-governance-quickstart.md`（docs_pr/release 门禁与治理要求）、`docs/superpowers/plans/2026-09-04-deepmd-3-2-compatibility-lane.md`（计划体例）、`docs/reports/2026-09-05-v082-compatibility-closeout.md`（发布口径与边界）。
- [ ] 跑 `python scripts/doc_check.py`（应 PASS）与 `python scripts/check_docs_freshness.py --days 90`（当前 exit=1，记录 10 个超期文件清单与既有失败 run，作为 Task 4 的基线）。

验收：基线记录进入最终报告；无未知脏文件。

### Task 1: 模型与数据获取指引

Files: `README.md`（Quick Start 附近新增小节）、`docs/guides/quickstart.md`（§3.1 前置条件增补）、`docs/guides/installation.md`（如需一行交叉链接）。

内容（以本计划给定字符串为准，勿臆造 URL 细节）：

- 预训练模型 DPA-3.1-3M：AIS Square 模型库 `aissquare.com/models`（id=343）。
- 铁基 FTS 数据集 FT2DP-dataset-FeCHO-v1：AIS Square 数据集 `aissquare.com/datasets`（id=306）。
- 仓库自带测试夹具 `tests/integration/data/DPA-3.1-3M.pt`（约 47MB，随仓库跟踪）可用于冒烟；外部模型可用 `DPEVA_DPA3_MODEL_PATH` / `DPEVA_DPA4_MODEL_PATH` 覆盖。
- 指向 Task 2 的 `docs/guides/smoke-demo.md`。

- [ ] 完成并以 `doc_check` 校验。

### Task 2: 冒烟演示（Smoke Demo）文档化 + 夹具 README 更正

Files: 新建 `docs/guides/smoke-demo.md`；修正 `tests/integration/data/README.md`；`docs/README.md` 索引增行。

内容：

- 基于夹具（1 帧 `sampled_dpdata/122`、5 帧 `other_dpdata_all/amourC/C2`、模型 `DPA-3.1-3M.pt`）与 `docs/guides/testing/integration-config-templates.md` 模板，给出 Local backend 最小闭环命令序列（feature→train→infer→collect），标注预期产物、`DPEVA_TAG: WORKFLOW_FINISHED` 与“仅冒烟、非科学结果、帧数极少”的限制。
- 更正 `tests/integration/data/README.md` 中 “Large DeePMD pretrained models ... intentionally not tracked” 的与事实不符表述：`DPA-3.1-3M.pt` 自 `da7babf` 起被跟踪（47,176,032 字节）；DPA4 模型仍为外部夹具；保留环境变量覆盖说明。

- [ ] 完成并以 `doc_check` 校验；命令至少在 Task 7 完成可执行性核对。

### Task 3: SAI 快速上手（用户向）

Files: 新建 `docs/guides/sai-quickstart.md`；`docs/README.md` 索引增行；`docs/guides/slurm.md` 若缺则增加一句交叉链接。

内容（汇总既有事实，不复制字段定义；逐条给出处链接）：

- 环境：`dpeva-dpa4` conda 环境；构建指南 `docs/guides/developer/deepmd-kit-sai-build.md`；`scripts/env/dpeva-dpa4.env`。
- 安装与自检：clone → `pip install -e .` → `dpeva doctor`。
- 配置要点：`submission.env_setup`、partition/qos；模板 `examples/recipes/labeling/config_sai_abacus_slurm.json`。
- 已知坑（出处 `docs/guides/slurm.md`）：flood-gpu/rush-gpu 的 1GPU 请求可能被 QOSMinGRES 拒（用 4GPU MPI 类）；`source /etc/profile` 必须在 #SBATCH 指令之后、module load 之前；`launcher_mode="mpi_abacus"` 会自动插入 SAI rank-map；array 提交上限；长作业走 8V100V0。

- [ ] 完成并以 `doc_check` 校验；与 `slurm.md` 逐条对照一致。

### Task 4: 修复 freshness 前置红状态（发布路径必备）

背景：`docs_freshness` 是 `docs_pr` 与 `release` profile 的组成项；当前 10 个 living 文档超期导致 exit=1，进而使（a）定时「Documentation Governance」失败、（b）`Deploy Docs` 的 release-validation 会失败并跳过部署。

Files（每个仅做“复核 + `last-updated` 刷新”，内容如需小修保持最小）：

- `docs/governance/tools/unwired-module-audit-template.md`（106 天）
- `docs/guides/testing/multi-datapool-artifacts.md`（106 天）
- `docs/guides/README.md`（102 天）
- `docs/guides/developer/deepmd-kit-sai-build.md`（102 天）
- `docs/guides/developer/README.md`（102 天）
- `docs/policy/README.md`（99 天）
- `docs/reference/validation.md`（99 天）
- `docs/superpowers/README.md`（99 天）
- `docs/plans/README.md`（99 天）
- `docs/guides/slurm.md`（98 天；本任务与 Task 3 交叉，注意合并不冲突）

步骤：

- [ ] 逐份打开复核：确认内容仍准确、链接有效；仅更新 `last-updated: 2026-10-11`（必要时做与本次补丁一致的极小幅修正，如需修正请在报告说明）。
- [ ] 跑 `python scripts/check_docs_freshness.py --days 90`，确认 exit=0；再跑 `python scripts/doc_check.py` 确认 PASS。
- [ ] 提交：`docs: refresh reviewed living documents to restore freshness gate`。

验收：freshness exit=0（本地 + 推送后 CI「Documentation Governance」的 freshness 步骤恢复绿色）。

### Task 5: README 文档站入口

Files: `README.md`。

内容：Documentation 区块加入文档站 `https://quantummisaka.github.io/dpeva/latest/`（根路径 404，必须链接 `/latest/`）。

- [ ] 完成；`curl -I` 复核 200。

### Task 6: 门禁、提交与推送

- [ ] 跑 `python scripts/run_gate.py docs_pr`（文档必跑项，须全绿）；本地依赖不足无法全跑时记录未覆盖项，以 GitHub Actions CI 为准；另跑 `python scripts/release_helper.py --check`。
- [ ] 按主题分批提交（英文 conventional 风格），至少包含本计划归档与 Task 1–5 内容；提交信息示例：`docs: add model/data acquisition guide and docs-site links`、`docs: add smoke demo and correct fixture README`、`docs: add SAI quickstart for new users`、`docs: refresh reviewed living documents to restore freshness gate`、`docs(plans): add onboarding and v0.8.2 release plan`。
- [ ] push main（远端 `https://github.com/QuantumMisaka/dpeva`；https 推送失败时先执行 `gh auth setup-git`）；确认 CI（至少 python-quality、docs-check、Documentation Governance 的 push 运行）绿并记录链接。

验收：`git diff --stat 5b0294a..HEAD` 证明本任务只含文档文件；CI 绿。

### Task 7: 冒烟干跑（受限）

- [ ] 在 `~/scratch` 下 fresh clone + venv，按仓库既有 wheel acceptance 方式（见 closeout 报告：system-site-packages venv + `pip install --no-deps -e .`）验证 entry-point help 与版本导入；不安装重型 GPU 依赖。
- [ ] 若本机存在可用 `dp` 环境（检查 `which dp`），按 Task 2 命令实跑最小闭环并记录耗时；否则记录限制（未执行、原因）。

验收：命令、输出摘要、耗时与限制进入报告。

### Task 8: 发布 v0.8.2

- [ ] 发布点复核：HEAD 为 main 最新、`release_helper.py --check` 通过、CI 绿（含 freshness 修复后的 governance 运行）；尽力本地执行 `python scripts/run_gate.py release`，不可行时记录并以 CI（docs-deploy 的 release-validation）为准。
- [ ] `git tag -a v0.8.2 -m "..."`（消息摘要 closeout 要点 + 本次入职文档）；push tag；`git ls-remote --tags` 复核可见。
- [ ] `gh release create v0.8.2 --title "v0.8.2" --notes-file <file>`（notes 基于 closeout 报告：兼容性收尾、fail-closed 行为、3 supported / 8 experimental / 4 unsupported / 2 blocked-upstream 边界、安装包络、“不含新科学资格”；附加本次入职文档说明与安装命令）。
- [ ] 验证：tag 指向、Release URL、docs-deploy（main 与 tag）成功（release-validation 全绿）、`/latest/` 200、`/v0.8.2/` 出现（轮询等待部署完成，上限合理）。

验收：以上六项证据齐备（命令输出/HTTP 码/链接）。

### Task 9: 发布记录

- [ ] 发布完成后新增 `docs/reports/2026-10-11-v082-release-cut.md`（体例参照既有 reports；front-matter 完整），记录：发布点 commit、tag/Release 链接、门禁与 CI 结果、已知限制；追加提交 main（不进入 tag，属发布后记录）。
- [ ] 勾选本计划内完成的 checkbox 并随最终一批提交归档。

验收：`doc_check` 通过；报告链接进入最终汇报。

## Acceptance（总验收模式）

- A1 新增/修正文档落库，`docs_pr`（或 CI 等价）通过；新增页面 front-matter 完整、无个人绝对路径；**freshness exit=0，10 个超期 living 文档完成复核刷新**。
- A2 远端 `v0.8.2` annotated tag 存在且指向发布点；GitHub Release 页面存在并含 notes。
- A3 `python scripts/release_helper.py --check` 通过，版本面一致（0.8.2）。
- A4 docs-deploy 对 tag 与 main 成功（release-validation 全绿）；`https://quantummisaka.github.io/dpeva/latest/` 200；`/v0.8.2/` 可访问。
- A5 冒烟干跑证据（或明确限制）齐备。
- A6 本任务对 `src/**` 与 `pyproject.toml` 零改动（diff 证据）。
- A7 报告含：逐 Task 状态、提交 SHA 列表、门禁/CI 链接、遗留问题、未触碰范围清单。
