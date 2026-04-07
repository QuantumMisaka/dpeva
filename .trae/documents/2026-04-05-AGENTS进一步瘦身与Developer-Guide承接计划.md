# AGENTS 进一步瘦身与 Developer Guide 承接计划

## Summary

本计划用于继续收敛 `AGENTS.md` 的职责边界，使其从“项目目标 + 关键契约 + 最小治理协议”进一步瘦身为**AI 与人类开发者快速理解项目开发的关键入口**。详细开发规范、门禁、生命周期与执行细节统一下沉到 `docs/guides/developer-guide.md`，避免 `AGENTS.md` 与项目文档系统形成高度重复。

本轮目标：

1. 将当前 `AGENTS.md` 中仍然偏“规范正文”的内容迁移或吸收到 `docs/guides/developer-guide.md`；
2. 将 `AGENTS.md` 压缩为“项目定位 + 最小阅读路径 + 极简契约摘要 + 不重复原则”；
3. 保证 `AGENTS.md` 成为 `developer-guide.md` 的入口，而不是第二份开发规范。

## Current State Analysis

### 1. 当前 `AGENTS.md` 的状态

已通过只读核查确认，当前 [AGENTS.md](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/AGENTS.md) 已经比旧版精炼很多，但仍然保留了五块较重内容：

- `Read First`
- `Core Engineering Contracts`
- `Documentation Lifecycle`
- `Quality Gates`
- `Do Not Duplicate`

其中以下内容本质上已经超出“入口页”职责：

- `Core Engineering Contracts` 中的 6 条项目契约正文
- `Documentation Lifecycle` 的完整规则与路径协议
- `Quality Gates` 的完整命令清单

这些内容虽然精炼，但仍然构成了一套独立规范层，容易与 `developer-guide.md`、`docs-governance-quickstart.md`、`docs/policy/*` 的细节正文产生竞争关系。

### 2. 当前 `developer-guide.md` 的承接能力

已核查 [developer-guide.md](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/docs/guides/developer-guide.md)，其当前已具备以下基础：

- 项目概述与目标
- 架构实践与模块边界
- 开发流程标准
- 系统架构与目录结构
- 包级导出层与主链路入口边界
- 大量核心模块与工作流细节

但它仍缺少一组“可以完整接住 AGENTS 下沉内容”的显式章节：

- 面向开发者的“核心工程契约”汇总节
- 更清晰的“文档生命周期 / 过程资产迁移规则”开发者视角摘要
- 更集中且可引用的“质量门禁与提交流程”章节

因此本轮最合理的动作不是继续让 `AGENTS.md` 承载这些内容，而是**增强 `developer-guide.md` 的总控角色**。

### 3. 与现有文档体系的关系

已核查：

- `docs/README.md` 已承担 docs 总导航与文档分层说明；
- `docs/guides/docs-governance-quickstart.md` 已承担文档治理最小闭环；
- `docs/policy/contributing.md`、`maintenance.md`、`quality.md` 已承担规则正文；
- `docs/guides/cli.md`、`configuration.md` 已承担具体接口与配置说明。

因此新的职责边界应为：

- `AGENTS.md`：项目开发的超短入口页
- `developer-guide.md`：开发规范、工程契约、架构与流程的主入口
- 其余 guides/policy/reference：专项正文

### 4. 当前高重复风险点

本轮已确认以下重复风险：

1. `AGENTS.md` 的 `Documentation Lifecycle` 与 `docs/README.md` / `docs-governance-quickstart.md` / `policy/*` 高度重叠
2. `AGENTS.md` 的 `Quality Gates` 与 `developer-guide.md` 的开发流程标准、文档治理快速上手重叠
3. `AGENTS.md` 的 `Core Engineering Contracts` 虽然正确，但目前没有在 `developer-guide.md` 中形成同等密度的承接章节，导致它仍像“独立规范正文”

## Proposed Changes

### A. 继续瘦身 `AGENTS.md`

#### 目标

把 `AGENTS.md` 压缩为真正的入口页，只保留首次打开仓库时必须在 30 秒内知道的内容。

#### 文件范围

- `AGENTS.md`

#### 改造方向

保留：

- 项目一句话定位
- `Read First`
- 极简版 `Core Signals` / `Key Signals`
- 一段“详细规则请进入 developer-guide”式跳转
- 极简的“不重复”边界

下沉：

- 完整 `Core Engineering Contracts`
- 完整 `Documentation Lifecycle`
- 完整 `Quality Gates`

删除或继续压缩：

- 明细级规范句子
- 带命令的门禁正文
- 文档生命周期的完整路径协议列表

#### 建议结构

建议改成 4 个模块：

1. `Purpose`
2. `Read First`
3. `Key Signals`
4. `Boundary`

其中 `Key Signals` 只保留非常短的 4-5 条摘要，例如：

- 配置以 `src/dpeva/config.py` + Reference 为准
- 主链路以 `cli.py -> workflows/*.py` 为准
- 对外契约变化必须同 PR 更新文档
- 详细规范统一进入 `docs/guides/developer-guide.md`

### B. 扩充 `developer-guide.md` 承接 AGENTS 下沉内容

#### 目标

把原本还放在 `AGENTS.md` 的开发规范性正文，迁移到 `developer-guide.md`，让其真正成为开发者主入口。

#### 文件范围

- `docs/guides/developer-guide.md`

#### 建议新增或重组章节

1. 新增“开发者快速入口 / Read First”小节
   - 明确本页与 `AGENTS.md`、`docs/README.md`、`docs-governance-quickstart.md` 的关系

2. 新增“核心工程契约”小节
   - 配置中心化
   - 路径显式解析
   - 工作流独立可执行与共享底层模块
   - 执行入口与导出入口分离
   - 对外契约联动更新
   - 数据格式约束

3. 新增或重组“文档生命周期与过程资产迁移规则”
   - `docs/reports/`
   - `docs/plans/`
   - `.trae/documents/`
   - `docs/archive/`
   - `docs/source/` 索引检查

4. 新增或重组“质量门禁与提交前验证”
   - `ruff check src tests scripts`
   - `pytest tests/unit`
   - `python3 scripts/doc_check.py`
   - `python3 scripts/check_docs_freshness.py --days 90`
   - `make -C docs html SPHINXOPTS="-W --keep-going"`

#### 具体承接原则

- `developer-guide.md` 负责“解释与总览”
- `docs/policy/*` 继续作为规则正文权威来源
- `docs/reference/*` 继续作为字段和校验正文权威来源
- 不把 policy/reference 的全文复制进 developer-guide，只做开发者视角的摘要和入口绑定

### C. 明确 `AGENTS.md -> developer-guide.md` 的入口关系

#### 目标

让读者在 `AGENTS.md` 中清楚知道：这是入口，不是完整手册。

#### 文件范围

- `AGENTS.md`
- `docs/guides/developer-guide.md`

#### 实施方式

- 在 `AGENTS.md` 首屏明确写出：
  - “本文件仅保留项目开发最小入口；详细规范、门禁、生命周期与架构说明请进入 `docs/guides/developer-guide.md`”
- 在 `developer-guide.md` 顶部补一行反向关系：
  - “若只需快速建立项目心智模型，请先读 `AGENTS.md`；若需要开发细则，请继续阅读本页”

### D. 保持与现有文档体系去重

#### 目标

避免把 AGENTS 瘦身后又把重复内容堆进 developer-guide，导致新的重复层出现。

#### 文件范围

- `docs/guides/developer-guide.md`
- `docs/guides/docs-governance-quickstart.md`
- `docs/README.md`

#### 决策

- 本轮以“承接 AGENTS 下沉内容”为主，不额外重写 `docs/README.md`
- 若 `developer-guide.md` 新增章节与 `docs-governance-quickstart.md` 出现明显重复，只保留：
  - `developer-guide.md`：开发者为什么需要这些规则、在哪看细节
  - `docs-governance-quickstart.md`：文档治理执行清单

### E. 验证与完成标准

#### 目标

确保实施后 `AGENTS.md` 明显更短，且 `developer-guide.md` 真正接住下沉内容。

#### 完成判定

实施完成后应满足：

- `AGENTS.md` 的篇幅与信息密度明显下降
- `AGENTS.md` 不再独立承载完整质量门禁与文档生命周期正文
- `developer-guide.md` 能直接承接开发者需要的工程契约与门禁信息
- 打开 `AGENTS.md` 时不会再产生“这是第二份 developer-guide”的感觉

## Assumptions & Decisions

### 已确认事实

- 当前 `AGENTS.md` 已经过一次瘦身，但仍然承载较多规范正文
- `developer-guide.md` 已是开发者主入口，且拥有足够承接空间
- `docs/README.md` 与 `docs-governance-quickstart.md` 已承担更上层和更专项的导航职责
- 你的明确要求是：**所有能放入 `developer-guide.md` 的都放入里面，仅在 `AGENTS.md` 中留下必要精髓**

### 本轮边界

- 本轮主要改 `AGENTS.md` 与 `docs/guides/developer-guide.md`
- 不计划大幅改写 `docs/README.md`
- 不计划重写 policy/reference 正文
- 不新增新文档，只在现有入口间重新分工

### 核心决策

- `AGENTS.md` 是入口，不是规范正文
- `developer-guide.md` 是开发主手册
- policy/reference/guides 专项页继续是细分权威来源

## Verification

实施完成后验证步骤：

1. 内容验证
   - 对比 `AGENTS.md` 改造前后，确认章节数与正文长度下降
   - 复查 `AGENTS.md` 不再包含完整门禁命令列表和完整文档生命周期协议
   - 复查 `developer-guide.md` 已新增或吸收核心工程契约、质量门禁和文档生命周期摘要

2. 去重验证
   - 搜索 `AGENTS.md` 中是否仍大段复述 `developer-guide.md` 的正文
   - 搜索 `developer-guide.md` 是否只是简单复制 `policy/*` 或 `reference/*` 全文

3. 质量验证
   - `python3 scripts/doc_check.py`
   - `python3 scripts/check_docs_freshness.py --days 90`
   - `make -C docs html SPHINXOPTS="-W --keep-going"`

4. 交付物验证
   - `AGENTS.md` 成为超短入口页
   - `developer-guide.md` 成为承接开发规范的主入口
