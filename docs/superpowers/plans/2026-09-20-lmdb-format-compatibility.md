---
title: LMDB 数据格式兼容 Implementation Plan
status: proposed
audience: Developers / AI Agents
last-updated: 2026-09-20
owner: Workflow Owner (IO) / Compatibility Owner
---

# LMDB 数据格式兼容 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `deepmd/lmdb` 在 DP-EVA 内成为可探测、可读、可诊断、可声明（能力矩阵）的受支持数据格式：读侧消除全部静默降级与静默截断，训练/评测路径按上游真实能力声明并前置校验，写出侧与体系身份策略按决策门收敛，最终按仓库既有证据规则晋级能力。

**Spec:** `docs/reports/2026-09-19-lmdb-read-support.md`（研究侧提案，status: proposed）+ 本计划 §1 实测基线（2026-09-19/20 复核并修正了该提案的三处表述）。

**Authorization:** 主人 2026-09-20 要求编排完整兼容开发计划。计划落盘不改变代码；执行前需确认 §2 的 D1–D5，其中 D4/D5 影响 PR 切分。

**Architecture:** 单一格式探测原语（对齐 deepmd `is_lmdb`）→ 读侧路由（LMDB 只经 `dpdata.MultiSystems`，禁止单体系 API）→ 帧身份索引（`frame → system_id/nloc/frame_idx`）→ 能力矩阵 `data_format` 维度 + `DeepMDAdapter` preflight fail-closed → 写出侧按 D4 决定是否纳入 `output_format` 契约。

**Tech Stack:** Python 3.10+、dpdata≥1.1（`deepmd/lmdb` 格式）、lmdb + msgpack（dpdata 的传递依赖）、deepmd-kit 3.2（`pt` / `pt-expt`）、pytest、Slurm/SAI V100、现有 `scripts/run_gate.py` 门禁体系。

## Global Constraints

- 不改变 `deepmd/npy` / `deepmd/npy/mixed` 的既有行为、默认输入输出格式与数值语义（唯一例外是 D1 授权的失败语义）。
- LMDB 读取必须经 `dpdata.MultiSystems.from_file(fmt="deepmd/lmdb")`；任何单体系 API（`dpdata.LabeledSystem` / `dpdata.System`）都不得用于 LMDB 路径——它对多组成 LMDB 只返回第一个组成组且仅发 `warnings.warn`。
- 对外名称统一 canonical `deepmd/lmdb`。裸 `lmdb` 别名在 dpdata 1.0.2 指向语义不同的 legacy 插件，只允许作为输入兼容映射并在日志中规范化，不作为 dpeva 内部常量。
- 不新增 dpeva 直接依赖；`lmdb`/`msgpack` 由 `dpdata>=1.1` 传递引入，必须在依赖边界文档中写明。
- 能力晋级遵守既有 `CapabilityRecord` 规则：`supported` 需要 `verification_command`（可收集 pytest node）+ `required_evidence` + `evidence_ref`（`cpu-contract` 与/或 `sai-v100-qualification`，后者还需 `sai_verification_cases`）；CPU 证据不能代表 GPU/Slurm；SAI qualification 需要显式授权与不可变证据目录。
- 每步验证用当前有效证据（`superpowers:verification-before-completion`）；涉及远端作业的实验必须带作业号与不可变产物路径。

---

## 1. 实测基线（2026-09-19/20，本机复现）

以下事实决定设计，后续任务不得与之矛盾。命令均在仓库根执行。

### 1.1 dpdata 侧语义

| 事实 | 证据 |
|---|---|
| 格式名：`deepmd/lmdb`（canonical）与 `lmdb`（legacy 别名，1.0.2 下是另一个插件） | dpdata 1.1.0 `dpdata/plugins/lmdb.py`；1.0.2 下 `fmt="lmdb"` → `KeyError: 'system_info'`，`fmt="deepmd/lmdb"` → `NotImplementedError` |
| `MultiSystems.from_file(<lmdb>, fmt="deepmd/lmdb")` 可用；分组键是**组成（atom_numbs）**，不是原始目录 | dpdata 1.1.0 `LMDBFormat._group_frames`；`bulk_ref_v1` 16 组/20 帧、`legacy_fecho_aligned_clean` 589 组/26 122 帧、`complement_validation_v5` **841 组**/2 604 帧（dpeva npy 路径为 **964 体系**） |
| 读回时按 global type 稳定排序并同步置换所有原子字段 | `LMDBFormat._canonicalize_frame`；`bulk_ref_v1` 20 帧中 6 帧与 npy 存储顺序不同，排序后 max\|Δcoord\|=0 |
| 读回**不含** `real_atom_types` / `real_atom_names` | 实测 keys：`atom_names, atom_numbs, atom_types, cells, coords, energies, forces, orig, spins, virials` |
| `LabeledSystem(<lmdb>)` 只返回第一个组成组 + 1 条 warning | 实测：`bulk_ref_v1` 2/20、`complement_validation_v5` **1/2 604**、`fe5c2_keyprobe_v1_cleaned_v3` 23/128、`g3_*`/`iron_carbide_dz_v1` 1/6–1/10 |
| 读侧默认 `max_frames=100 000`，超出直接抛错 | `DEFAULT_MAX_FRAMES`；外部转换器已为其加 `--verify-max-frames` 开关 |
| `mixed_type=True` 读回全局 type_map（缺失元素计数为 0） | 实测 `complement_validation_v5` → `['C','Fe','H','O']` + `[15,0,12,3]` |
| 写侧：`MultiSystems.to("deepmd/lmdb", ...)` 按组成合并；`dpdata.formats.deepmd.lmdb.dump_systems(systems, ...)` 按输入顺序保留 `frame_system_ids`（供 `prob_sys_size`） | dpdata 1.1.0 `to_multi_systems` / `dump_systems` docstring |
| 写侧：mixed 源必须同时提供 `real_atom_types` 与 `real_atom_names` | `_prepare_atom_source`：`"Mixed-type data requires both real_atom_types and real_atom_names."` |
| 依赖：dpdata 1.1.0 的 `Requires-Dist` 含 `lmdb>=2.0.0`、`msgpack`（非 extra） | 1.1.0 wheel METADATA |

### 1.2 deepmd-kit 3.2 侧支持矩阵（GA = 3.2.0，env `ft2dp-post`；dev = 3.2.0b1.dev67，env `dpeva-dpa4`）

**结论先说**："deepmd-kit 3.2 已支持 LMDB"这一判断对 **train / test 成立**，对 **eval-desc / embed 不成立**；后者在 GA、dev 与上游 master 三处都没有 LMDB 分支。

| 命令 / 路径 | LMDB | 依据（含本轮实跑） |
|---|---|---|
| `dp train`（`training_data` / `validation_data`） | ✅ 仅当 `systems` 是**单个字符串**路径；neighbor-stat 亦走 `make_neighbor_stat_data` | GA 与 dev 源码 `pt/entrypoints/main.py`：`isinstance(training_systems, str) and is_lmdb(...)` → `LmdbDataset` |
| `dp test`（`-s`） | ✅ 整个 LMDB = 1 个数据源，按 nloc 分组评测；`-d` 明细行序 = nloc 组升序 | 实跑（env `dpeva-dpa4`，`dp --pt test -m ft2dp-dpa4-mini-regular-50k.pt -s bulk_ref_v1.lmdb -n 0`）：输出 `# testing system : …/bulk_ref_v1.lmdb`、`LmdbTestData type remapping: LMDB ['Fe','O','C'] -> model […], remap=[25 7 5]`、`# mixed-nloc LMDB: testing 13 groups: {1: 1, 2: 2, …}`；随后仅因登录节点无 GPU（`Failed to load libcuda.so`）中断，与格式无关。另有外部 `validation-set/lmdb/READING_DP_TEST_OUTPUT.md` |
| `dp eval-desc` | ❌ | 实跑：`dp --pt eval-desc -m <model> -s bulk_ref_v1.lmdb` → `RuntimeError: Did not find valid system`（`eval_desc.py:77`，早于模型加载）；对照 `-s cleaned_v1`（npy 容器）越过该检查、进入邻居计算。源码：`expand_sys_str` + `DeepmdData`，LMDB 相关引用 0 处（GA、dev、master 均为 0） |
| `dp embed` | ❌ | 实跑同上 → `RuntimeError: Did not find valid system`（`embedding.py:127`）；`direct` 判定：`expand_sys_str(<lmdb>)` → `[]`，`DeepmdData(<lmdb>)` → `FileNotFoundError: No set.* is found` |
| `dp freeze` / `change-bias` | 与数据格式无关 | — |
| LMDB 判定规则 | `str(path).endswith(".lmdb") or Path(path, "data.mdb").is_file()` | `dpmodel/utils/lmdb_data.py::is_lmdb` |

上游状态（2026-09-20 查询 GitHub API）：master 的 `entrypoints/eval_desc.py`、`entrypoints/embedding.py`、`utils/data.py` 对 lmdb 的引用数均为 0；issue/PR 搜索 `repo:deepmodeling/deepmd-kit lmdb eval-desc` 命中 0 条。即该缺口既非"版本落后"，也不在上游在途计划中。已提交上游跟踪 issue：[deepmodeling/deepmd-kit#6034](https://github.com/deepmodeling/deepmd-kit/issues/6034)（含复现、根因与输出布局建议）。若 D2 选择"等待上游"，manifest 记录可据此登记 `blocked-upstream` 并引用该编号；否则按 `unsupported` 登记，DP-EVA 侧仍保持提交前拒绝。

### 1.3 DP-EVA 现状缺陷清单（本计划要闭合的对象）

| # | 位置 | 缺陷 |
|---|---|---|
| F1 | `src/dpeva/io/dataset.py:46,105` | 候选格式硬编码，LMDB 不可发现 |
| F2 | `src/dpeva/io/dataset.py:127` | 零体系时静默 `return []` |
| F3 | `src/dpeva/io/dataset.py:96-101` | 单体系优化分支先于多体系分支 → 若把 LMDB 加入候选列表，`LabeledSystem` 会**静默只返回第一个组成组**（实测最坏 1/2 604 帧） |
| F4 | `src/dpeva/workflows/labeling.py:196,208`、`training/managers.py:64`、`feature/managers.py:43-56`、`io/dataset.py:120-123` | 体系探测只看 `type.raw`/`type_map.raw`/`set.000`，LMDB 目录一律被判为非体系 |
| F5 | `training/managers.py:62-83` | 容器目录展开为子体系列表 → 内含 `.lmdb` 的容器会被展开成 list，而上游 `dp train` 的 LMDB 路径只接受单个字符串 |
| F6 | `feature/managers.py:100-185` | 默认 `feature_exporter="eval_desc"`（`dp eval-desc`）/`embed` 在 3.2 GA 不支持 LMDB，且无前置校验 → 白跑 Slurm 后才失败 |
| F7 | `io/dataproc.py:248-300`、`uncertain/manager.py:100-150`、`workflows/data_cleaning.py:49` | 逐帧 `dataname` 归属依赖 `-d` 明细的注释行序（npy 体系序）；LMDB 下注释行是 nloc 组、行序按 nloc 升序 → 归属错误或静默跳过 |
| F8 | `labeling/integration.py:444-458` | `_frame_identity` 哈希存储顺序的 `atom_types`+`coords` → npy 与 lmdb 读回顺序不同，跨格式去重静默漏检 |
| F9 | `compatibility/deepmd-3.2.json`、`io/dataset.py` 调用方（`src` 内 14 处调用点 + `tools/deepmd_verify_common.py` 4 处）、`tests/unit/io/test_dataset.py:91-94` | manifest 只有 `train + deepmd/lmdb`（experimental/planned）；`DeepMDAdapter` 调用方全部走 `for_legacy_unchecked`，`data_format` 维度实际未被检查；现有单测**断言**非法路径返回 `[]` |

顺带发现（不属本计划范围，但需登记）：`validation-set/complement_validation_v5/cleaned_v5/provenance/` 缺 `type.raw` 使 npy 侧 `MultiSystems.from_file` 直接抛 `FileNotFoundError`，dpeva 目前靠子目录扫描兜底——与报告中"payload 内含非体系子目录"是同一类问题。

### 1.4 已完成的预检（降低 D5 风险）

- dpdata 1.1.0（PYTHONPATH 注入，env `dpeva-dpa4`）：`pytest tests/unit -q` → **950 passed / 31.45 s**。
- GA lane（env `ft2dp-post`：deepmd-kit 3.2.0 + dpdata 1.1.0 原生）：`pytest tests/unit -q` → **945 passed, 5 skipped**（skip 为缺 torch 的 DPOSE 用例）。
- 结论：单元层对 dpdata 下限提升不敏感；真实数据路径（`dp train`/`dp test` + 真实 LMDB）仍需 Phase 5 的端到端证据。

---

## 2. 决策门（执行前需确认；括号内为推荐默认）

**D1 — 零体系失败语义（推荐：新增 `on_empty="error"` 且默认 `error`）。**
`src` 内 14 处调用点中有 3 类容忍空结果：`labeling._load_dataset_map` 多池扫描（当前 warn+skip）、`io/collection.py::count_frames`（统计，异常返回 0）、`inference|analysis::load_composition_info`（异常返回 `None` 并降级为均值扣除）。推荐默认 `error` + 显式豁免上述三类，并在豁免处把"降级"写进日志与产物元数据。代价：多池模式下"空池"从跳过变为整流程失败（行为变更，需在 PR 描述中标注）。

**D2 — LMDB 能力声明范围（推荐：`test`/`train` 先 `experimental`，读侧不进 deepmd 矩阵）。**
读侧（dpdata 路径）是 dpeva 自有契约，进 `docs/reference/upstream-software.md` 与 `dpeva doctor`；`dp train`/`dp test` 的 LMDB 走 `capability/deepmd-3.2.json` 的 `data_format: "deepmd/lmdb"` 记录，先 experimental，凭 Phase 5 证据晋级 `supported`。`dp eval-desc`/`dp embed` 上游三处（GA/dev/master）实测均无 LMDB 支持，需在两种登记方式中选一：`unsupported`（dpeva 不再等待，仅前置拒绝）或 `blocked-upstream`（需先提交上游 issue 并把编号写入 `upstream_issue`）。无论选哪种，dpeva 都必须在提交作业前 fail-closed。

**D3 — 体系身份承诺等级（推荐：帧级等价为一等承诺；体系级仅在存在身份映射时承诺）。**
LMDB 是扁平帧存储，按组成分组读回；原始目录级体系身份只能来自 `__metadata__.frame_system_ids`（帧号）或 sidecar（名字）。推荐：dpeva 承诺"帧集合与逐帧数值等价"，体系分组以组成分组为准；凡依赖 `target_systems=` 或逐体系指标的消费方，必须提供身份映射，否则显式报错或显式降级并在产物中标注口径。

**D4 — 写出侧范围（推荐：本次不把 LMDB 纳入 `output_format` 契约）。**
仅提供 dpeva 内的转换/自校验工具（移植 `tools/npy2lmdb.py` 能力，基于 `dump_systems` 保身份），`VALID_LABELING_OUTPUT_FORMATS` 与 `config.py` validator 不变。若纳入契约，需要同步扩展常量、validator、两个写出点（`labeling/manager.py:533-540`、`labeling/integration.py:175`）、recipes、文档与测试。

**D5 — dpdata 下限策略（推荐：收紧为 `dpdata>=1.1`）。**
依据 §1.4 的 950/945 通过证据，收紧下限比"保留下限 + 运行时探测"更简单且避免 1.0.2 的 `lmdb` 别名语义混淆；同时 `dpeva doctor` 仍报告实际版本以支持预置环境（`--no-deps`）场景。

---

## 3. 目标兼容矩阵（"完全兼容"的可达边界）

| 消费路径 | 配置入口 | LMDB 目标等级 | 依赖任务 |
|---|---|---|---|
| `dpeva train` 训练/验证数据 | `TrainConfig.training_data_path` → `input.json: training.{training_data,validation_data}.systems` | 原生透传（单字符串） | T8 |
| `dpeva infer` 候选池 `dp test` | `InferConfig.data_path` | 原生（deepmd 侧支持） | T8/T10 |
| `dpeva analysis`（model_test / dataset） | `AnalysisConfig.data_path` / `dataset_dir` | 读侧支持 + 结果归属需映射 | T1/T10 |
| `dpeva clean` | `CleanConfig.dataset_dir` | 读 + 帧归属需映射 + 导出仍为 npy | T1/T10/T11 |
| `dpeva feature`（`eval_desc`/`embed` exporter） | `FeatureConfig.data_path` | **不兼容**（上游 GA 实测）；dpeva 必须前置 fail-closed 或改走 in-process | T9 |
| `dpeva feature`（in-process generator） | `feature/generator.py` | 读侧支持（dpdata） | T1 |
| `dpeva collect` | `CollectConfig.testdata_dir` | 读侧支持（组成分组语义） | T1/T12 |
| `dpeva label`（候选集探测 + 导出） | `LabelingConfig.input_data_path`、`output_format` | 读侧支持；导出仍 npy（受 D4） | T1/T8/T13 |
| `dpeva label` integration（读+写+去重） | `existing_training_data_path`、`merged_training_data_path` | 读支持 + 身份规范化；写受 D4 | T12/T13 |
| `dpdata` 侧只读消费者（analysis/UQ 校验/统计） | — | 读侧支持，组成分组语义 | T1/T12 |

---

## 4. 任务

### Phase 1 — 读侧契约与失败语义（PR-1）

#### Task 1: 单一格式探测原语与 LMDB 读路由

**Files:** `src/dpeva/io/dataset.py`（探测 + 路由）、新增 `tests/unit/io/test_dataset_lmdb.py`

**Behavior:**
- 新增 `is_lmdb_path(path) -> bool`，规则与 deepmd 完全一致：`str(path).endswith(".lmdb") or Path(path, "data.mdb").is_file()`。
- 新增 `detect_dataset_kind(path)`：`lmdb | single_system | container | missing`，判定用统一体系标记（`type.raw`/`type_map.raw`/`set.000`/`data.mdb`），供 T8/T9/T12 复用（取代 F4 的四处散落判断）。
- `fmt` 规范化：接受 `deepmd/lmdb`（canonical）与 `lmdb`（别名，记 INFO 日志后规范化）；其余未知格式保持现状。
- `load_systems` 新增 LMDB 分支，且该分支必须**先于**单体系优化分支（F3）：`dpdata.MultiSystems.from_file(path, fmt="deepmd/lmdb", max_frames=None)`；显式拒绝在 LMDB 上调用 `_load_single_path`。
- 读回后断言帧数等于 `__metadata__.nframes`（读元数据一次即可），不一致立即报错。

**Dependencies:** D5（dpdata≥1.1）。

**Verification:**
- 新单测：探测器（含空目录、仅 `data.mdb` 文件、`.lmdb` 后缀、npy 体系目录、容器目录）；断言 LMDB 输入**不会**调用 `_load_single_path`（用 `patch` 计数）。
- 真实小 LMDB 契约用例（dpdata≥1.1 + `lmdb` guard；`pytest.importorskip`）：断言系统数/帧数与期望一致，且 20 帧集全量返回（防 F3 回归：断言帧数 ≠ 第一个组成组帧数）。

#### Task 2: 零体系的 fail-loud 与豁免清单

**Files:** `src/dpeva/io/dataset.py`、`src/dpeva/workflows/labeling.py`、`src/dpeva/io/collection.py`、`src/dpeva/inference/managers.py`、`src/dpeva/analysis/managers.py`、`tests/unit/io/test_dataset.py`

**Behavior:**
- 新增 `DatasetLoadError(ValueError)`，信息含路径、探测到的 kind、尝试过的格式、`dpdata`/`deepmd-kit` 版本、以及"若是 LMDB 需要 dpdata≥1.1"的指引。
- `load_systems(..., on_empty: Literal["error","warn"] = "error")`（受 D1）；`on_empty="warn"` 保留旧语义但必须 `logger.warning`。
- 豁免处显式传参：`labeling._load_dataset_map`（多池扫描）、`collection.count_frames`、`inference/analysis.load_composition_info`；后两者在降级时记录"结果将使用均值扣除/跳过组合信息"的显式提示。
- 更新 `tests/unit/io/test_dataset.py:91-94`（现断言返回 `[]`）为断言抛 `DatasetLoadError`。

**Dependencies:** T1、D1。

**Verification:** 新增/更新单测覆盖 `error`/`warn` 两分支；`pytest tests/unit -q` 全绿；`workflows/labeling.py` 的多池语义变更在 PR 描述中标注。

#### Task 3: 依赖边界、doctor lane 与能力预检（读侧）

**Files:** `pyproject.toml`、`src/dpeva/run/doctor.py`、`src/dpeva/utils/env_check.py`、`tests/unit/test_dependency_contracts.py`、`tests/unit/run/*`（就近）

**Behavior:**
- `pyproject.toml`：`dpdata>=1.0.0` → `dpdata>=1.1`（D5）；同步 `docs/guides/installation.md`。
- `dpeva doctor` 增加一条独立 lane：`dpdata` 版本检查 + LMDB 读能力结论（"lmdb-read: ok/unavailable(needs dpdata>=1.1)"），并说明 `lmdb`/`msgpack` 由 dpdata 传递引入。
- 依赖契约测试新增 dpdata 断言（风格沿用 `tests/unit/test_dependency_contracts.py` 现有用例）。

**Verification:** `dpeva doctor --json` 在 `dpeva-dpa4`（dpdata 1.0.2）与 `ft2dp-post`（1.1.0）两侧输出符合预期；`python scripts/run_gate.py` 相关层通过。

#### Task 4: 文档联动（读侧）

**Files:** `docs/reference/upstream-software.md`、`docs/guides/installation.md`、`docs/guides/configuration.md`、`docs/guides/cli.md`、`docs/guides/troubleshooting.md`

**Behavior:** 写明支持的输入格式清单（`deepmd/npy`、`deepmd/npy/mixed`、`deepmd/lmdb`）、dpdata 下限与 LMDB 依赖、LMDB 的体系分组语义与失败诊断入口；`upstream-software.md` §7 增加版本下限列（当前该表无版本列）。

**Verification:** `python scripts/run_gate.py docs_pr`（含 `docs_audit`/`docs_freshness`/`docs_build`/`docs_linkcheck`）。

### Phase 2 — 能力矩阵与治理登记（PR-2）

#### Task 5: `data_format` 维度落地与 LMDB 能力记录

**Files:** `src/dpeva/compatibility/deepmd-3.2.json`、`src/dpeva/compatibility/adapter.py`、`src/dpeva/training/managers.py`、`src/dpeva/inference/managers.py`、`src/dpeva/feature/managers.py`、`tests/unit/compatibility/*`

**Behavior:**
- 新增/更新记录：`test + deepmd/lmdb`、`train + deepmd/lmdb`（已有，保持 experimental/planned 直到 Phase 5）、`eval-desc + deepmd/lmdb`（`unsupported`）、`embed + deepmd/lmdb`（`unsupported`）。`unsupported` 记录用于前置拒绝，不携带 `verification_command`。
- 让调用方从配置派生 `data_format`（新增 `data_format_for(path) -> "deepmd/lmdb" | "deepmd/npy"`），并在 `DeepMDAdapter.test/train/...` 传入完整 `CapabilityKey`；未登记组合默认 fail-closed，`experimental` 需要显式 `allow_experimental`。

**Dependencies:** D2。

**Verification:** 单测覆盖：LMDB + `eval-desc` 在提交命令前抛 `CapabilityUnavailable`；LMDB + `test` 在 experimental 下需显式开关；manifest schema 校验（重复键、planned 不带命令、unsupported 不带证据引用）。

#### Task 6: 追踪矩阵登记

**Files:** `docs/governance/traceability/capability-evidence.json`、`docs/governance/traceability/feature-doc-matrix.md`、`docs/governance/traceability/workflow-contract-test-matrix.md`

**Behavior:** 新增/扩展条目（`code_paths`/`test_paths`/`documentation_paths`/`owner`/`evidence_path`），把 LMDB 读能力与 Phase 5 的契约测试节点、文档路径绑定；更新两个矩阵文档对应行。

**Verification:** `python scripts/check_traceability.py`。

### Phase 3 — 消费链对齐（PR-3）

#### Task 7: 训练/评测路径透传

**Files:** `src/dpeva/training/managers.py:35-90`、`tests/unit/training/*`

**Behavior:**
- `resolve_data_path`：当 `systems` 是字符串且 `is_lmdb_path` 为真时，跳过后缀的目录展开，保留单字符串（上游要求），日志说明"LMDB 作为单一数据源"。
- 容器目录内含 `.lmdb`：按 D2 记录状态处理——若该记录为 `unsupported`/未登记则 fail-closed，并提示"每个任务只能有一个 LMDB 数据源，或改用 npy/mixed"。
- 同样处理 `validation_data`；训练 LMDB + 验证 npy 的混搭保持上游语义（上游已在 pt entrypoint 处理）。

**Verification:** 单测断言 `systems` 保持字符串、容器内含 `.lmdb` 时前置报错；现有训练单测不回归。

#### Task 8: feature exporter 前置校验（受 D3）

**Files:** `src/dpeva/feature/managers.py:95-190`、`src/dpeva/config.py`（如新增 exporter 选项）、`tests/unit/feature/*`

**Behavior:**
- `feature_exporter in {"eval_desc","embed"}` 且输入为 LMDB → 在提交作业前抛 `CapabilityUnavailable`，信息给出两条可行路径：(a) 使用 npy/mixed 副本；(b) 使用 in-process generator（`get_feature`/`load_systems` 路径）。
- 若 D3 选择"支持 in-process LMDB 特征导出"，则新增 exporter 选项并在同一 PR 内实现（写出 HDF5/npy 时明确帧序：按组成分组顺序，并在产物中记录口径）。
- 若 D2 选择 `blocked-upstream`：在同一 PR 内提交上游 issue（标题需含 `dp eval-desc`/`embed` + LMDB），并把编号写入 manifest 记录的 `upstream_issue` 字段；PR 描述中记录 2026-09-20 的三处实测依据。

**Verification:** 单测覆盖拒绝分支与（若实现）in-process 分支；§1.2 的 GA 实测结论在测试注释中标注为上游依据。

#### Task 9: `dp test -d` 逐帧归属（帧身份索引）

**Files:** `src/dpeva/io/dataproc.py:234-360`、新增 `src/dpeva/io/lmdb_index.py`、`src/dpeva/uncertain/manager.py:100-150`、`src/dpeva/workflows/data_cleaning.py:38-60`、`tests/unit/io/test_dataproc_lmdb.py`

**Behavior:**
- 新增 `read_lmdb_frame_index(path)`：从 `__metadata__` 读 `nframes`/`frame_nlocs`/`frame_system_ids`/`frame_idx_fmt`，产出 (a) storage 帧序、(b) `dp test` 评测序（nloc 升序 + 组内 storage 序）、(c) `frame → (system_id, nloc)` 映射；可选消费 `<set>.frame_index.json` 获取**名字**（存在时用于展示，不存在不失败，D3）。
- `DPTestResultParser`：输入为 LMDB 时改用索引生成 `dataname_list`/`datanames_nframe`（名称形如 `<lmdb name>#sys<id>/<frame>` 或 sidecar 名字），不再依赖 `-d` 注释行序；无法建立映射时 **fail-loud**，禁止输出"看似合理"的逐帧归属。
- `uncertain/manager.py` natom 交叉校验与 `data_cleaning` 的 `target_systems` 分支：LMDB 输入时改走索引（不再按目录名 join），否则 fail-loud。

**Verification:** 单测用真实小 LMDB（含多 nloc 组）+ 手工构造的 `-d` 明细，断言逐帧归属与 nloc 组序一致；断言错误输入下 fail-loud；`data_cleaning` 的导出用例保持通过。

#### Task 10: 跨格式帧身份与 mixed 语义

**Files:** `src/dpeva/labeling/integration.py:377-500`、`tests/unit/labeling/*`

**Behavior:**
- `_frame_identity` 改为规范形式（按元素 + 坐标/类型排序后哈希），使 npy 与 lmdb 读回的同一帧得到同一身份；保留 `real_atom_names`/`real_atom_types` 参与身份但不再隐含顺序。给出"行为等价性"证据：在既有 npy 数据上比较规范化前后去重结果（应一致）。
- LMDB 读回缺 `real_atom_types`：在 `_ensure_compatible_type_map` 与写出决策中显式处理（受 D4），并在 integration 报告中记录来源格式与身份口径。

**Verification:** 新增跨格式去重用例（同一帧分别以 npy 与 lmdb 读入，断言判重命中）；既有 integration 单测不回归。

### Phase 4 — 写出侧（PR-4，受 D4）

#### Task 11: LMDB 转换与自校验工具

**Files:** 新增 `src/dpeva/io/lmdb_writer.py`（或 `tools/`）、`tests/unit/io/test_lmdb_writer.py`

**Behavior:**
- 提供 `npy → lmdb` 转换：默认 `dump_systems`（保输入顺序与 `frame_system_ids`），可选 `merge_by_composition=True`（走 `MultiSystems.to`）；`type_map` 显式传入以支持流式。
- 处理两个已知坑：mixed 源注入 `real_atom_names`；payload 内非体系子目录按 `type.raw`/`set.000`/`data.mdb` 判定跳过（同 §1.3 附注）。
- 写出后自校验：帧数、`frame_system_ids` 长度、逐帧 canonical hash（对齐外部 `canon_sym` 方法）；默认 `overwrite=False`，POSIX 原子替换。

**Dependencies:** D4。

**Verification:** 单测（小 fixture）：往返帧数/哈希一致、mixed 源与 provenance 目录两坑各一条回归；`pytest tests/unit -q`。

#### Task 12: `output_format` 契约扩展（仅当 D4 选择纳入）

**Files:** `src/dpeva/constants.py:60-62`、`src/dpeva/config.py:483-503`、`src/dpeva/labeling/manager.py:533-540`、`src/dpeva/labeling/integration.py:141-234`、`examples/recipes/README.md` 与相关 recipe、`docs/guides/configuration.md`、`tests/unit/labeling/*`、`tests/unit/test_config_migration.py`

**Behavior:** 把 `deepmd/lmdb` 加入合法写出格式；写出前校验目标非空/可写；写出后执行 T11 的自校验；文档与 recipes 同步。

**Verification:** 配置校验用例、导出用例、recipes 校验（`examples/recipes/README.md` 声明的一致性）、`docs_pr` 门禁。

### Phase 5 — 证据与能力晋级（PR-5，需 SAI 显式授权）

#### Task 13: 契约测试节点与 CPU 证据

**Files:** `tests/contract/deepmd/test_lmdb_contract.py`、`tests/contract/deepmd/conftest.py`、`tests/contract/deepmd/data/README.md`、`docs/reports/evidence/deepmd-3.2/lmdb-cpu-*/`

**Behavior:** 复用现有 fixture gate 模式：新增 `DPEVA_DEEPMD_LMDB_DATA`（由 qualified 环境提供，或由 `dpdata>=1.1` 在测试内生成小 LMDB）+ 现有 `DPEVA_DEEPMD_PT_MODEL`；至少覆盖 `dp test -s <lmdb>` 与 `dp train`（短步数）两条命令，并记录指标与 `-d` 明细行序断言。产出 CPU evidence JSON。

**Verification:** 契约节点可运行、可收集；`verification_status` 可置 `implemented`。

#### Task 14: SAI qualification 与 manifest 晋级

**Files:** `docs/reports/evidence/deepmd-3.2/lmdb-sai-qualification-*/`、`src/dpeva/compatibility/deepmd-3.2.json`、`docs/reports/2026-09-20-lmdb-format-compatibility.md`、`docs/reports/README.md`

**Behavior:** 在 SAI V100 上执行 `dp test` 的 npy↔lmdb 等价 A/B（目标：四项指标相对差 ≤2e-5，沿用研究侧方法与证据），保留不可变证据；随后按 `CapabilityRecord` 规则晋级 `test + deepmd/lmdb`（以及必要的 `train + deepmd/lmdb`），在同一变更中更新 manifest 与报告。

**Verification:** `python -m pytest tests/unit tests/contract -q`（含 manifest schema 校验）、`python scripts/run_gate.py release` 中与兼容性相关的层；报告与 README 索引更新。

---

## 5. PR 切分与执行顺序

| PR | 内容 | 关键门禁 | 可独立合并 |
|---|---|---|---|
| PR-1 | T1–T4（读侧契约 + 失败语义 + 依赖/doctor + 文档） | `pytest tests/unit -q`、`ruff check`、`run_gate.py docs_pr` | 是 |
| PR-2 | T5–T6（能力矩阵 `data_format` + preflight + 追踪登记） | 同上 + manifest schema 测试 | 是（依赖 PR-1 的探测原语） |
| PR-3 | T7–T10（train 透传、feature 前置校验、`dp test` 帧归属、跨格式身份） | 同上 + 相关单测 | 是 |
| PR-4 | T11–T12（写出侧；受 D4） | 同上 | 是（可延后） |
| PR-5 | T13–T14（证据与晋级；需 SAI 授权） | 契约测试 + SAI 作业证据 | 需授权后执行 |

前置条件：当前工作区有未提交改动（`src/dpeva/workflows/labeling.py` 的 `custom_headers`、`tests/unit/workflows/test_labeling_workflow.py`、两处 README），应先落干净再开 LMDB 分支，保证 PR 可审。

---

## 6. 测试与 fixture 策略

- **探测/路由单测**不需要真实 LMDB：用空目录 + 仅含 `data.mdb` 占位文件的目录即可覆盖 `is_lmdb_path`/`detect_dataset_kind`。
- **真实 LMDB 读契约**由测试内生成（`dpdata>=1.1` + `lmdb` guard，`pytest.importorskip("lmdb")`），不提交 `.mdb` 二进制；生成规模控制在 3 个体系 / ≤10 帧。
- **失败语义**必须有专门的"零体系 + 未知格式"用例（覆盖原 `:127` 路径与 `tests/unit/io/test_dataset.py:91-94` 的旧断言）。
- **不采用**只 mock 的"伪 LMDB"测试来声称兼容性：`_load_single_path` 截断（F3）这类缺陷只会在真实 dpdata 下暴露，因此至少要有一条真实读路径断言"帧数 = 元数据帧数"。
- **端到端与 GPU/Slurm 结论**只能由 SAI qualification 提供；CPU 证据不得外推。

---

## 7. 风险与不可达区域

| 风险 | 影响 | 缓解 |
|---|---|---|
| 上游 `dp eval-desc`/`dp embed` 不支持 LMDB（3.2 GA 实测） | feature 的 CLI exporter 路径**无法**做到"原生兼容" | T8 前置 fail-closed + 文档给出替代路径；如需原生支持应向上游提 issue 并登记 `blocked-upstream` |
| 体系身份不可恢复 | system_balanced 指标、`target_systems`、逐体系导出语义改变 | D3 明确承诺等级；T9 帧身份索引；缺失映射时 fail-loud/显式降级 |
| 原子顺序被读侧重排 | 跨格式去重、逐原子对齐、`-d` 明细配对 | T10 身份规范化；测试中固定"排序规范化后可比较"的口径 |
| `max_frames` 默认上限 10 万 | 大 LMDB 读取直接报错 | T1 显式 `max_frames=None` + 文档记录内存代价；训练侧仍由 deepmd 流式读取 |
| dpdata 版本混淆（`lmdb` 别名） | 1.0.2 下同名不同义 | Global Constraints：只用 canonical 名 + T3 doctor lane |
| 预置环境（`--no-deps`）未升级 dpdata | 运行期才发现不可读 | T3 doctor + T2 fail-loud 报错信息含版本指引 |

---

## 8. 完成定义（DoD）与验收命令

1. `pytest tests/unit -q`（在 dpdata≥1.1 的 GA lane 中）全绿；新增 LMDB 用例覆盖探测、路由、失败语义、帧归属、跨格式身份。
2. 真实 LMDB 读契约：帧数与 `__metadata__.nframes` 一致；与 npy 副本逐帧等价（按元素排序规范化后 max\|Δ\|=0）。
3. `dpeva doctor --json` 能区分"可读 npy / 可读 LMDB"，并报告 dpdata 版本与 `lmdb`/`msgpack` 来源。
4. LMDB 输入在 `feature --feature-exporter eval_desc|embed`、`train` 容器多数据源等不可支持组合上**在提交作业前**失败，错误信息给出替代路径。
5. 能力矩阵含 `deepmd/lmdb` 记录，状态与证据一致；`python scripts/check_traceability.py` 与 `run_gate.py docs_pr` 通过。
6. 需要 SAI 授权时：A/B（npy↔lmdb）四项指标相对差 ≤2e-5，证据归档到 `docs/reports/evidence/`，报告与 `docs/reports/README.md` 索引更新。
7. `docs/reports/2026-09-19-lmdb-read-support.md` 状态按实现结果更新（或在完成报告中引用其 closure）。

---

## 9. 执行记录（2026-09-20，PR-1 范围，工作区未提交）

已按 §4 Phase 1 落地"能支持的先支持、不能支持的前置拒绝"：

| 任务 | 状态 | 落点 |
|---|---|---|
| T1 探测原语 + LMDB 路由 | 已完成 | `src/dpeva/io/dataset.py`：`is_lmdb_path`（对齐 deepmd `is_lmdb`）、`detect_dataset_kind`、`normalize_format`、`_load_lmdb_systems`（只走 `MultiSystems(fmt="deepmd/lmdb", max_frames=None)`）、帧数完整性校验（对比 `__metadata__.nframes`）、`_load_single_path` 显式拒绝 LMDB |
| T2 fail-loud 与豁免 | 已完成 | 新增 `DatasetLoadError` 与 `load_systems(..., on_empty="error"|"warn")`；豁免点：`workflows/labeling.py`（多池扫描）、`io/collection.py::count_frames`、`uncertain/manager.py`、`io/dataproc.py` 回退查询 |
| T3 依赖边界与 doctor | 已完成 | `pyproject.toml` → `dpdata>=1.1`；`run/doctor.py` 新增信息性检查 `dpdata.lmdb`；`tests/unit/test_dependency_contracts.py` 新增下限断言 |
| T4 文档联动 | 已完成 | `docs/reference/upstream-software.md`（§2 dpdata + §2.1 支持矩阵 + §7 版本列）、`docs/guides/installation.md`、`docs/guides/configuration.md`（§3.0 数据集格式边界）、`docs/guides/troubleshooting.md`（§5.3）、`docs/guides/cli.md`（doctor 说明） |
| T7 训练路径透传 | 已完成 | `training/managers.py::resolve_data_path`：LMDB 保持单字符串；容器内含 LMDB 时在展开前报错 |
| T8 feature 前置拒绝 | 已完成（拒绝分支） | `feature/managers.py::submit_cli_job`：LMDB + `eval_desc`/`embed` 在提交前抛错并给出 npy 副本 / 进程内生成两条替代路径；in-process exporter 选项仍待 D3 |
| T9 `dp test` 帧归属 | 部分（拒绝分支） | `io/dataproc.py`：`-d` 明细出现 LMDB 数据源标记（`.lmdb` / `[nloc=`）时拒绝输出逐 system 统计；帧索引映射实现属 Phase 3 |
| T10 跨格式身份 | 未开始 | Phase 3 |
| T5/T6 能力矩阵与追踪 | 部分（登记） | `docs/governance/traceability/capability-evidence.json` 新增 `io.dataset.lmdb`；manifest 的 `data_format` 维度与 preflight 派生（T5）仍待 D2 |
| T11–T14 写出侧与证据 | 未开始 | 受 D4 与 SAI 授权 |

### 9.1 验证证据（2026-09-20）

| 检查 | 环境 | 结果 |
|---|---|---|
| `pytest tests/unit -q`（dpdata 1.0.2 默认 lane） | `dpeva-dpa4` | 972 passed, 3 skipped（LMDB round-trip 按 guard 跳过） |
| `pytest tests/unit tests/contract/deepmd/test_fixture_gate.py -q`（dpdata 1.1.0） | `ft2dp-post` | 976 passed, 5 skipped |
| `scripts/run_gate.py unit`（含 `--cov-fail-under=80`） | `dpeva-dpa4` | 978 passed, 3 skipped；coverage 83.54% |
| `ruff check src tests scripts` | `dpeva-dpa4` | All checks passed |
| `scripts/run_gate.py audit` | `ft2dp-post` | ✅ Audit PASSED |
| `scripts/check_traceability.py` | `ft2dp-post` | ✅ passed（11 条目） |
| `make -C docs html SPHINXOPTS="-W --keep-going"` | `dpeva-dpa4` | build succeeded |
| `run_gate.py docs_artifacts` / `make -C docs linkcheck -W` | `dpeva-dpa4` | ✅ 通过 |
| `scripts/doc_check.py`（= `docs_audit`） | `ft2dp-post` | ❌ 仍失败，原因是工作区三个**既有未跟踪** plan 文件缺 YAML front matter（`2026-07-10-*`、`2026-07-11-*`、`2026-09-02-*`），与本计划改动无关；`docs_pr` profile 因此无法整体变绿 |

### 9.2 执行中的裁决与偏离

- **Ruling（LMDB 分组命名）**：LMDB 读回的分组没有名称，`load_systems` 为其合成 `"<lmdb 目录名>[<组成式>]"` 作为 `target_name`，用于日志与图表标签；不声称这是原始体系名。代价：若下游把它当成真实体系名做匹配，会得到组成分组语义——已在文档与本记录中说明。
- **Ruling（拒绝而非静默降级）**：`target_systems`、`-d` 明细解析、`eval-desc`/`embed`、容器多 LMDB 四类场景一律前置报错，而不是给出"看起来合理"的结果。代价：原先能被静默容忍的用法会显式失败（例如用 LMDB 跑 Clean），这正是本次要消除的失败模式。
- **偏离**：计划 T9 原定实现帧索引映射；本轮先落"识别并拒绝"，因为把 nloc 组映射回体系还需同时回答 D3 的承诺等级与 sidecar 缺失策略。

### 9.3 落地、CI 与收口判定（2026-09-20）

**落地提交（已在 main）**

| commit | 内容 |
|---|---|
| `bf1b58d` | `feat(io): read deepmd/lmdb datasets and fail loud on unreadable inputs`（T1–T4、T7、T8 拒绝分支、T9 拒绝分支、T6 登记） |
| `ac58940` | `docs: reference upstream LMDB issue for eval-desc/embed`（回填上游 issue 编号） |

**CI 结果（推送后核对）**

| 检查 | 结果 | 归因 |
|---|---|---|
| `Python Quality / unit-tests` | ✅ | 本轮改动通过 CI 单测与覆盖率门 |
| `Python Quality / audit` | ✅ | — |
| `Python Quality / explore-extra-smoke` | ✅ | — |
| `Python Quality / lint` | ❌ 既有 | dev extra 只声明 `ruff>=0.1.0`，CI 使用 ruff 0.16.8；**基线 commit `8cf5c51` 上已有 1071 条**报错，本地 ruff 0.15.15 全绿 |
| `Python Quality / integration-tests` | ❌ 既有 | `test_multidatapool_e2e[local]` 在 `8cf5c51` 上同样失败：dev extra 不含 torch，`dp eval-desc` 无法启动 |
| `Docs Build & Check` | ❌ 既有 | 8 条 `myst.xref_missing`，全部来自 5 个本轮未改动文件（指向 `../../scripts/gates.toml`）；main 自 2026-09-06 起红，本地因 doctree 缓存未暴露 |
| `Documentation Governance` | ❌ 既有 | `docs_audit` + freshness（多篇文档 >90 天），2026-09-14 周更已红 |
| `DeepMD 3.2 CPU Contract` / `Deploy Docs` | ❌ 既有 | 需受保护 fixture 的 qualification lane / 跟随 docs 构建失败 |

**收口判定**：本轮交付范围（读侧契约、失败语义、可支持路径透传、不可支持组合前置拒绝、依赖边界与 doctor、文档与追踪登记）已完成并推送，CI 中直接检验本改动的三项检查全绿。计划自身保持 `proposed`：Phase 2–5 与决策门 D2–D4 未实施，表内其余红灯为既有问题，不由本轮引入、也不在本计划范围内。
