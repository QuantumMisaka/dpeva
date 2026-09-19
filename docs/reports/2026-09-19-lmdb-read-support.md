---
title: LMDB 读取支持（数据侧格式扩展）
status: proposed
audience: Developers / Scientific Owner
last-updated: 2026-09-19
owner: QuantumMisaka
---

# LMDB 读取支持（数据侧格式扩展）

## Scope

- **目标**：让 `dpeva.io.dataset.load_systems` 及其调用方（label extract / integration / analysis）把 `deepmd/lmdb` 当作合法输入格式。
- **范围外**：不改训练配置语义、不改默认写出格式（仍是 `deepmd/npy` / `npy/mixed`）、不引入新依赖（只收紧已有依赖下限）。
- **动机**：外部工作流已把训练/验证数据做成 **npy(+mixed) 与 lmdb 双副本**（LMDB 的优势是单文件、便于迁移），
  且 deepmd-kit 3.2.x 已原生支持 LMDB 训练与评测；dpeva 是链路上唯一还不认 LMDB 的一环，且失败方式是**静默返回空**。

## Evidence（2026-09-19 实测，可复核）

环境：`dpeva-dpa4-320-pr6022`（deepmd-kit `3.2.1.dev0+g687b5107`、dpdata `1.0.2`、torch `2.11.0+cu126`），SAI 登录节点；样例数据集 `bulk_ref_v1`（16 dirs / 20 frames）。

| # | 命令 / 场景 | 结果 |
|---|---|---|
| 1 | `dpdata.LabeledSystem("<set>.lmdb", fmt="lmdb")` | `KeyError: 'system_info'` |
| 2 | `dpdata.MultiSystems.from_file("<set>.lmdb", fmt="lmdb")` | 同上 |
| 3 | `dpeva.io.dataset.load_systems("<set>.lmdb")` | **返回 `[]`（静默，无异常）** |
| 4 | `dpeva.io.dataset.load_systems("<set npy 目录>")` | 16 systems（对照，正常） |
| 5 | dpdata `1.1.0` 环境（独立 venv）`MultiSystems.from_file(lmdb, fmt="lmdb")` | 正常；npy↔lmdb 逐帧 canonical hash 往返 **0 missing**（8 个验证集合 + 3 个训练集） |
| 6 | deepmd-kit `3.2.1.dev0` `dp test` / `dp train` 直接吃 LMDB | 正常；`dp test` npy vs lmdb 相对差 ≤4e-7（另一集合 0），200 步训练墙钟差 ~1.5%、loss 差 ~1% |

代码位置：

- `src/dpeva/io/dataset.py:46`、`:105`：`formats_to_try = ["deepmd/npy/mixed", "deepmd/npy"]`（`fmt="auto"` 时的候选格式，硬编码）；
- `src/dpeva/io/dataset.py:120-122`：MultiSystems 失败后的"扫描子目录"fallback（对 LMDB 无意义）；
- `src/dpeva/io/dataset.py:127`：一条都没加载到时 `return []`（**建议改为报错**）；
- `pyproject.toml:32`：`dpdata>=1.0.0`（LMDB 读写需要 `dpdata>=1.1`；`pyproject.toml` 未反映该能力边界）。

## Proposal（最小改动）

**P1 依赖边界**
- `pyproject.toml` 收紧为 `dpdata>=1.1`（或保留下限，但在运行时对 LMDB 请求给出明确报错：`LMDB 需要 dpdata>=1.1，当前 x.y.z`）；
- 在 `docs/reference/upstream-software.md` 的依赖边界表注明 "LMDB read: dpdata>=1.1"。

**P2 代码（预计 ~30 行）**
1. **LMDB 分支**：当 `path` 是目录且存在 `data.mdb` 时，走 `dpdata.MultiSystems.from_file(path, fmt="lmdb")` 并直接返回其 systems（LMDB 天然是多体系容器），**不进入**子目录扫描 fallback；
2. **候选格式**：`fmt == "auto"` 时把 `"lmdb"` 加入候选（放在 npy/mixed 之后，避免改变既有行为）；
3. **fail-loud**：`load_systems` 在"零 system 加载成功"时抛 `ValueError`（信息含路径、尝试过的格式、dpdata 版本），取代 `:127` 的静默 `return []`；调用方无需改动即可获得可诊断的报错；
4. **体系边界（可选但推荐）**：LMDB 的 `__metadata__` 条目携带 `frame_nlocs` / `frame_system_ids`（dpdata 写入时保留），或消费同目录 sidecar `<set>.frame_index.json`（外部工作流已为每个 LMDB 生成）即可恢复原始体系划分，供 system_balanced 统计使用。

**P3（可选，写出侧）**：移植 npy→lmdb 转换（外部参考实现 `tools/npy2lmdb.py`）。两个已知坑：
- mixed 源（`real_atom_types.npy`）需同时注入 `real_atom_names`（取自 `type_map.raw`），否则 writer 拒绝；
- payload 目录里可能有非体系子目录（如 `provenance/`），必须按 `type.raw`/`set.000` 判定并跳过，否则整棵子树会被当作体系读取。

## Acceptance Criteria

1. 同一数据集：`load_systems(<lmdb>)` 与 `load_systems(<npy>)` 的**体系数与帧数一致**，且逐帧数组一致（可用 canonical frame hash 比对，见 Evidence #5 的方法）；
2. 对同一模型：npy 与 lmdb 两侧 `dp test` 四项指标相对差 ≤2e-5（float32 累加量级；外部实测为 ≤2.2e-5，多数集合 ≤1e-6）；
3. 新增失败用例：未知/不支持的格式必须抛错（覆盖现 `:127` 的静默路径）；
4. 新增单测：`tests/` 内放一个 ~100 KB 的小 LMDB fixture（或测试内以 dpdata≥1.1 现场生成），断言加载帧数与 `type_map`；
5. 既有测试套件不回归（`pytest tests -q`）。

## Risks / Boundary

- **dpdata 下限提升**：1.0.2 → 1.1 可能影响其它数据路径；建议先跑完整测试套件，若需保守，可把 LMDB 支持实现为"版本检测 + 明确报错"而不动下限。
- **体系边界**：LMDB 本身不保存原始 system 归属，system_balanced 类统计需 P2.4 的 sidecar/metadata；本报告不改变现有指标口径默认值。
- **不做**：不把 LMDB 设为默认格式、不改训练/评测数值语义、不在本报告内引入新的数据转换依赖。

## Evidence Pointers（仓库外，FT2DP 验证工作区）

- 双副本策略、等效性实测与消费者支持矩阵：`/home/james/work/ft2dp-dpeva/validation-set/lmdb/FORMAT_POLICY_AND_AB_20260919.md`
  （SAI 侧同路径 `$R/validation-sets/lmdb/`；作业 `1404446`（A/B）、`1403020`（训练 smoke））；
- `dp test` 在 LMDB 上的输出读法与 `-d` 明细语义：`validation-set/lmdb/READING_DP_TEST_OUTPUT.md`；
- 转换器与分析工具：`tools/npy2lmdb.py`、`analysis/lmdb_build_frame_index.py`、`analysis/lmdb_detail_metrics.py`。

## Related Plan

- 开发计划（本报告为需求来源；计划在其基础上扩展了体系身份、`dp test` 解析侧、能力矩阵与写出侧范围）：
  [docs/superpowers/plans/2026-09-20-lmdb-format-compatibility.md](../superpowers/plans/2026-09-20-lmdb-format-compatibility.md)（status: proposed）

## Issue 摘要（可直接贴到 GitHub issue）

> **Title**: `io.dataset`: support `deepmd/lmdb` as an input format (and stop failing silently)
>
> `load_systems()` cannot read LMDB today: the candidate format list is hardcoded to `["deepmd/npy/mixed", "deepmd/npy"]`
> (`src/dpeva/io/dataset.py:46,105`) and, when nothing loads, it returns `[]` silently (`:127`) — with dpdata 1.0.2 the
> underlying call raises `KeyError: 'system_info'`. LMDB read/write needs `dpdata>=1.1` (`pyproject.toml:32` pins `>=1.0.0`).
> Meanwhile deepmd-kit 3.2.x reads LMDB natively for `dp train`/`dp test`, and external validation/training data already
> ships npy+lmdb dual copies (single-file LMDB for migration). Proposal: (1) add an LMDB branch
> (`data.mdb` present → `dpdata.MultiSystems.from_file(path, fmt="lmdb")`), (2) add `"lmdb"` to the auto candidates,
> (3) raise a descriptive `ValueError` instead of returning `[]`, (4) optionally consume the `<set>.frame_index.json`
> sidecar / `__metadata__` to restore per-system grouping. Acceptance: frame-count and per-frame equality vs the npy copy,
> ≤2e-5 relative `dp test` parity, a fail-loud unit test, no regressions.
