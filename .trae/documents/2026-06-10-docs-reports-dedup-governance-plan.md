---
title: docs/reports 去冗余与治理落实方案
status: draft
audience: Developers / Maintainers
last-updated: 2026-06-10
owner: Docs Owner
---

# docs/reports 去冗余与治理落实方案

## 1. Summary

目标：基于当前 `docs/reports/` 的真实内容与项目既有文档治理规则，给出一套可执行的去冗余与稳态治理方案，使该目录满足以下成功标准：

- 报告目录不再把“总报告、专题报告、阶段性闭环记录、实验系列报告”无差别并列为 active。
- 每个保留在 `docs/reports/` 的 active 报告都满足当前治理基线：有 front matter、有 owner、生命周期清晰、长期可复用。
- 已闭环或强过程性的报告从 active 视图中退出，按既有归档规范迁入 `docs/archive/.../reports/`。
- `README.md` 能明确表达报告类型、状态、父子关系与归档策略，减少读者误读。
- 后续新增报告有统一模板与准入规则，避免重复问题再次发生。

## 2. Current State Analysis

### 2.1 现有文件与类型分布

`docs/reports/` 当前包含：

- `README.md`
- `2026-04-01-Code-Review-Repository-Audit.md`
- `2026-04-02-Code-Review-Unit-Test-Audit.md`
- `2026-04-05-Code-Review-Agents-Docs-Governance.md`
- `2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md`
- `2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md`

从内容上可分为两类：

- 审查治理类：`2026-04-01`、`2026-04-02`、`2026-04-05`
- 实验对比类：`2026-06-07`、`2026-06-08`

### 2.2 已确认的冗余与治理问题

1. `2026-04-01-Code-Review-Repository-Audit.md` 已明确声明“整合本轮四份分域审查结果”，说明它是总报告而非普通并列报告。
2. `2026-04-02-Code-Review-Unit-Test-Audit.md` 与 `2026-04-01` 在“测试完整性与覆盖率”主题上明显重叠，但当前缺少“专题展开/附录”标识。
3. `2026-04-05-Code-Review-Agents-Docs-Governance.md` 含大量“本轮处理”“已修复”“已关闭”“第二阶段治理闭环”等过程性内容，更符合“阶段性治理收口记录”，不适合作为长期 active 报告继续并列展示。
4. `2026-06-07` 与 `2026-06-08` 不是内容重复，但章节结构、叙事方式和指标组织高度同构，缺少“系列报告”层级。
5. `docs/reports/README.md` 当前只有 Active Reports 列表，未区分“总报告 / 专题报告 / 实验系列 / 待归档”。
6. `2026-06-07` 与 `2026-06-08` 当前缺少 YAML front matter，不满足 `scripts/doc_check.py` 的 active 文档治理要求。
7. `docs/governance/inventory/owners-matrix.md` 尚未显式覆盖 `docs/reports/*`，与 `docs/policy/maintenance.md` 中 “Reports (Audit/Exp)” 的 RACI 约定不完全闭环。

### 2.3 已有治理规则与约束

本方案必须遵守当前仓库中已存在的治理规则：

- `docs/guides/docs-governance-quickstart.md`
  - `.trae/documents/` 放执行期草案，`docs/reports/` 只放“可共享、可复盘、可发布”的收敛结果。
  - 版本发布或任务闭环后统一归档到 `docs/archive/vX.Y.Z/{plans,reports}/`。
- `docs/policy/quality.md`
  - Report 属于“一次性结论”类文档，要求结论明确、范围与局限性明确。
  - `status: active` 文档必须具备完整元信息和 owner。
- `docs/policy/maintenance.md`
  - 报告默认冻结，新增通过新文件，不覆盖式重写旧结论。
  - 代码审查报告闭环后，随版本发布归档到 `docs/archive/vX.Y.Z/reports/`。
  - 任何物理移动必须同步更新索引。
- `docs/source/index.rst`
  - `docs/reports/` 当前明确“不纳入主导航稳定入口”，因此治理重点是仓库内目录可读性与索引准确性，不是 Sphinx 主导航扩展。
- `scripts/doc_check.py`
  - `docs/reports/` 不在忽略目录中，因此其中 active 文档必须满足 front matter、链接、owner 等检查。

## 3. Assumptions & Decisions

### 3.1 决策原则

- 保留“长期复用价值高、结论稳定、可被反复引用”的文档在 `docs/reports/` active 视图中。
- 将“已闭环的治理过程记录”和“仅作为专题证据、且不需要高频独立访问的报告”从 active 平铺视图中退出。
- 不删除历史结论；采用“父子关系重构 + 版本归档 + README 分类”组合治理，而不是简单删文件。
- 对实验报告不强制合并正文，但必须建立系列化治理和统一模板，降低未来重复写作成本。

### 3.2 文件级处理决策

#### 保留为 active 总报告

- `docs/reports/2026-04-01-Code-Review-Repository-Audit.md`
  - 角色：4 月审查总报告
  - 原因：已具备整合性与高层索引能力，适合保留为入口文档
  - 要求：在文首或摘要区补充“关联专题报告”字段，明确引用 `2026-04-02`

#### 保留但降级为专题报告

- `docs/reports/2026-04-02-Code-Review-Unit-Test-Audit.md`
  - 角色：`2026-04-01` 的测试专题展开
  - 原因：仍有细粒度证据价值，但不应被 README 当作与总报告同级的主结论入口
  - 要求：在 front matter 或文首增加 `Related` / `Parent Report` 语义，标明其隶属 `2026-04-01`

#### 迁出 active，归档为闭环治理记录

- `docs/reports/2026-04-05-Code-Review-Agents-Docs-Governance.md`
  - 角色：文档治理阶段性闭环记录
  - 原因：内容大量体现“本轮处理/已修复/已关闭”，生命周期已从“现行结论”转为“历史治理证据”
  - 目标路径：`docs/archive/v0.7.1/reports/2026-04-05-Code-Review-Agents-Docs-Governance.md`
  - 说明：本方案采用仓库当前已存在的最新归档版本目录 `v0.7.1`。若实施时项目已有更新发布目录，则改用当时最新 release 归档目录，但迁移规则不变。

#### 保留为 active 系列实验报告

- `docs/reports/2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md`
- `docs/reports/2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md`
  - 角色：采样对比系列下的两篇 active 实验报告
  - 原因：结论对象不同，不应归档；但需建立统一模板和系列标签
  - 要求：
    - 补充 YAML front matter
    - 统一元信息字段
    - 在文首补“Series / Related Reports”
    - 在 README 中以“系列实验报告”分组展示

## 4. Proposed Changes

### 4.1 更新 `docs/reports/README.md`

目的：把当前平铺式 active 列表改为“类型 + 生命周期 + 关系”的索引页。

具体修改：

- 保留当前目录用途与维护策略说明。
- 新增“报告分类规则”：
  - 总报告（Summary Report）
  - 专题报告（Focused Report）
  - 系列实验报告（Experiment Series）
  - 归档报告（Archived Report）
- 将“Active Reports”重构为以下分组：
  - `Active Summary Reports`
  - `Active Focused Reports`
  - `Active Experiment Series`
  - `Archive Candidates / Archived`
- 在每条索引中增加简短标签：
  - `Type`
  - `Status`
  - `Parent/Series`
  - `Archive Policy`
- 将 `2026-04-05` 从 active 列表移除，并在“Archived Reports”或“Recently Archived”区域保留一条索引。
- 为 `2026-04-01` 显式标记“关联专题：2026-04-02”。
- 为 `2026-06-07`、`2026-06-08` 显式标记“系列：Sampling Comparison Series”。

为什么这样做：

- 不改变文档站导航策略；
- 直接降低同类文档平铺造成的冗余感；
- 将文档治理语义显式化，便于后续审查与归档。

### 4.2 更新 `docs/reports/2026-04-01-Code-Review-Repository-Audit.md`

目的：把其稳定为 4 月审查的唯一主入口。

具体修改：

- 在 front matter 中补充与专题报告的 `related` 或等效说明字段。
- 在摘要或“审查结果整合”段落中增加一句：`2026-04-02` 为测试专题展开，供深入证据查阅。
- 不重写原有结论，不改变其作为审计报告的冻结属性。

为什么这样做：

- 让总报告与专题报告建立明确的父子关系；
- 避免 README 之外的读者在直接打开正文时仍感到重复。

### 4.3 更新 `docs/reports/2026-04-02-Code-Review-Unit-Test-Audit.md`

目的：将其从“并列主报告”重定位为“专题报告”。

具体修改：

- 保持正文结论不变。
- 在 front matter 中补充 `Related` 或 `Parent Report` 指向 `2026-04-01`。
- 在标题下增加一段短说明：本报告是 4 月仓库总审查中的测试专题展开，不单独代表全仓库总体结论。

为什么这样做：

- 减少读者把它误读为与 `2026-04-01` 同层级的总体审计；
- 保留其细节价值，不强行归档。

### 4.4 迁移 `docs/reports/2026-04-05-Code-Review-Agents-Docs-Governance.md`

目的：让已闭环的治理收口文档退出 active 目录。

具体修改：

- 物理迁移到 `docs/archive/v0.7.1/reports/`。
- 将 front matter 中 `status: active` 改为 `status: archived`。
- 在标题附近补充“历史治理记录 / 不作为现行规范来源 / 参考当前规则入口”的说明。
- 如 `docs/archive/v0.7.1/README.md` 或对应索引存在，则同步补充索引；若该版本目录下尚无 reports 索引，则新增最小 README 索引页。
- 在原 `docs/reports/README.md` 中更新索引状态。

为什么这样做：

- 与 `docs/policy/maintenance.md` 中“闭环后随版本发布归档”的规定一致；
- 清除 active 目录中的过程性噪声；
- 保留历史证据，不丢失治理轨迹。

### 4.5 更新 `docs/reports/2026-06-07-DPA4-Mini-MACE-Sampling-Comparison.md`

目的：使其满足 active 文档门禁，并纳入系列治理。

具体修改：

- 增加 YAML front matter，至少包含：
  - `title`
  - `status: active`
  - `audience`
  - `last-updated`
  - `owner`
- 在文首元信息中补充：
  - `Series: Sampling Comparison`
  - `Related: 2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md`
- 保持主体分析不变，只统一结构和元信息命名。

为什么这样做：

- 当前文件不满足 `doc_check` 的 front matter 要求；
- 系列关系显式化后，可降低“同模板重复文档”的管理成本。

### 4.6 更新 `docs/reports/2026-06-08-DPA4-Neo-Air-Plus-Sampling-Comparison.md`

目的：与 `2026-06-07` 形成统一的 active 实验报告规范。

具体修改：

- 同步补齐 YAML front matter。
- 同步补 `Series` 与 `Related Reports` 元信息。
- 统一文首元信息字段命名，避免 `Owners/Related/Date` 与 front matter 并存时语义重复。

为什么这样做：

- 当前文件同样不满足治理门禁；
- 与 `2026-06-07` 对齐后，后续新增采样报告可以直接复用模板。

### 4.7 更新 `docs/governance/inventory/owners-matrix.md`

目的：把 `docs/reports/*` 正式纳入 Owner 覆盖矩阵。

具体修改：

- 在“模块 Owner 映射”表中新增一行：
  - `docs/reports/*`
  - 范围：项目级审查报告与实验结论
  - Owner：`Auditor / Researcher` 或项目内部约定的统一 Owner 角色
  - 审查角色：`Tech Lead`
- 在维护策略中补一句：新增 active 报告或变更其状态时，必须同步检查 Owner 覆盖与 README 索引。

为什么这样做：

- 当前政策文档已经定义 Reports 的 RACI，但矩阵未显式落表；
- 该变更能补齐“规则存在但清单未落地”的治理断点。

### 4.8 可选更新 `docs/policy/maintenance.md`（仅在实施时发现需要时执行）

目的：如果 `docs/reports/README.md` 的新分类需要制度层支撑，则补充最小规则。

触发条件：

- 实施时若发现现有规则仍不足以约束“总报告 / 专题报告 / 系列实验报告”的使用边界，则在 `7.1 代码审查报告` 和相邻段落中加两条：
  - 总报告可引用专题报告，但专题报告不应与总报告在同级索引中作为并列总结入口呈现。
  - 已闭环的治理型报告应优先归档，不长期保留在 active 列表。

默认策略：

- 若仅靠 `docs/reports/README.md` 和 `owners-matrix.md` 即可完成治理闭环，则不改该政策文件，避免过度扩散改动范围。

## 5. Implementation Sequence

1. 先更新 `docs/reports/README.md`，确定新的类型与状态分组。
2. 再处理 4 月报告：
   - 固化 `2026-04-01` 的总报告定位
   - 将 `2026-04-02` 标记为专题报告
   - 迁移 `2026-04-05` 至 archive
3. 处理 6 月实验报告：
   - 补齐 front matter
   - 统一系列元信息
4. 更新 `docs/governance/inventory/owners-matrix.md`
5. 检查是否需要最小更新 `docs/policy/maintenance.md`
6. 运行治理验证并确认索引一致性

## 6. Verification

实施后应执行以下验证：

- `python3 scripts/doc_check.py`
  - 预期：`docs/reports/` 下 active 文档全部通过 front matter、链接、绝对路径检查
- `python3 scripts/doc_check.py --strict-owner --strict-owner-dir reports`
  - 预期：`docs/reports/` 下 active 文档 owner 覆盖率为 100%
- `python3 scripts/check_docs_freshness.py --days 90`
  - 预期：迁移后的 active 文档仍满足 freshness；archive 目录默认不参与检查
- `make -C docs html SPHINXOPTS="-W --keep-going"`
  - 预期：无因归档迁移或链接调整导致的 toctree / 链接警告

同时做以下人工复核：

- `docs/reports/README.md` 能一眼区分总报告、专题报告、系列实验报告、归档项
- 从 `2026-04-01` 可以顺利跳转到 `2026-04-02`
- `2026-04-05` 不再出现在 active 列表中，但仍可从 archive 索引找到
- `2026-06-07` 与 `2026-06-08` 的开头元信息一致，且不再缺少 front matter

## 7. Expected Outcome

落地后，`docs/reports/` 的目标状态为：

- active 目录仅保留 1 篇总审查报告、1 篇测试专题报告、2 篇系列实验报告和 1 个清晰的 README 索引；
- 已闭环治理文档转入 archive；
- 报告类文档补齐 front matter 与 owner 治理；
- README 和 Owner 矩阵共同承担目录治理职责；
- 未来新增报告可以沿用同一分类规则与模板，避免再次出现“平铺并列 + 生命周期混乱”的冗余问题。
