# DP-EVA Labeling 模块功能点开发与迁移规划文档

* **版本**: 1.0.0
* **状态**: 规划中 (Planning)
* **日期**: 2026-01-31
* **关联模块**: `utils/fp`

---

## 1. 概述 (Overview)

当前 `dpeva/utils/fp` 目录下的脚本群 (`FeCHO_split_treat-ftv3.py`, `reprepare.py`, `subjob_dist.py`) 构成了事实上的 "Labeling" (数据标注/DFT计算准备) 模块。虽然《功能迁移完整性确认表》曾将其标记为“未纳入”，但为了实现 DP-EVA 从“主动学习筛选”到“闭环迭代”的完整能力，有必要将这些经过实战验证的逻辑标准化、模块化，并沉淀到核心库 `src/dpeva` 中。

本相关文档旨在拆解现有脚本逻辑，设计可复用、可扩展的 `dpeva.labeling` 模块架构。

---

## 2. 现有脚本逻辑拆解 (Deconstruction)

### 2.1 `FeCHO_split_treat-ftv3.py` (ABACUS Input Generator)
*   **核心功能**: 将筛选出的 `dpdata` (DeepMD/NPY 格式) 批量转换为 ABACUS 计算任务 (`INPUT`, `STRU`, `KPT`)。
*   **关键逻辑**:
    1.  **结构预处理**: `wrap`, `center`, 坐标平移（确保结构位于 Box 中心）。
    2.  **维度识别 (`judge_vaccum`)**: 基于真空层厚度自动判定结构类型 (Cluster, Cubic Cluster, Layer, String, Bulk)。
    3.  **参数适配**:
        *   **K-Points**: 根据结构维度和晶格常数自动计算 K 点 (Cluster 自动设为 Gamma 点)。
        *   **Dipole Correction**: 针对层状结构 (Layer) 自动开启偶极修正。
        *   **Magnetic Moments**: 根据元素类型 (Fe, O, etc.) 自动设置初始磁矩。
    4.  **文件生成**: 调用 `ase.io.abacus` 生成标准输入文件。
*   **局限性**:
    *   硬编码了大量 DFT 参数 (`basic_input`) 和 路径 (`lib_dir`).
    *   磁矩设置逻辑与特定元素 (FeCHO) 强耦合。
    *   仅支持 ABACUS，难以扩展到 VASP/CP2K。

### 2.2 `reprepare.py` (Input Modifier)
*   **核心功能**: 对已生成的 ABACUS 任务目录进行批量参数调整（如修改 KPT、更换赝势、调整 Smearing）。
*   **关键逻辑**:
    *   遍历目录，读取 `STRU`。
    *   重新执行维度识别与 K 点计算逻辑 (逻辑与 Generator 高度重复)。
    *   覆盖写入 `INPUT`, `KPT`。
*   **局限性**: 代码重复率高，缺乏统一的配置管理。

### 2.3 `subjob_dist.py` (Job Packer)
*   **核心功能**: 将大量扁平化的任务目录打包分层 (e.g., `N_50_0/task_0..49`)，以避免文件系统压力。
*   **局限性**: 逻辑简单但通用性强，目前通过硬编码 `mv` 命令实现，缺乏原子性保障。

---

## 3. 目标架构设计 (Target Architecture)

建议新增顶级模块 `dpeva.labeling`，并扩展 `dpeva.submission`。

```text
src/dpeva/
├── labeling/
│   ├── __init__.py
│   ├── abacus.py           # ABACUS 专用生成器 (实现 Generator 接口)
│   ├── vasp.py             # (预留) VASP 生成器
│   ├── utils.py            # 通用几何工具 (真空层检测、K点估算)
│   └── base.py             # 抽象基类 (BaseGenerator)
└── submission/
    └── packer.py           # 任务打包工具 (原 subjob_dist.py)
```

### 3.1 核心类设计

#### 3.1.1 `dpeva.labeling.utils` (通用工具库)
将物理无关的几何算法下沉：
*   `detect_dimensionality(atoms, vacuum_tol=6.0) -> Tuple[bool, bool, bool]`
*   `estimate_kpoints(atoms, kspacing=0.14, dim_mask=[0,0,0]) -> List[int]`
*   `standardize_cell(atoms) -> Atoms` (Center, Wrap, Swap Axes)

#### 3.1.2 `dpeva.labeling.abacus.AbacusGenerator`
继承自 `BaseGenerator`，负责配置管理与文件生成。

```python
class AbacusGenerator:
    def __init__(self, config: dict):
        self.dft_params = config.get("dft_params", {})
        self.pp_map = config.get("pp_map", {})
        self.orb_map = config.get("orb_map", {})
        self.magmom_map = config.get("magmom_map", {"Fe": 5, "default": 0})
        
    def generate(self, systems: List[Atoms], output_dir: str):
        # 封装原脚本的循环逻辑
        pass
        
    def _configure_system_specifics(self, atoms: Atoms) -> dict:
        # 封装维度识别、K点计算、Dipole修正逻辑
        pass
```

#### 3.1.3 `dpeva.submission.JobPacker`
通用任务打包器。

```python
class JobPacker:
    def __init__(self, pack_size=50, prefix="Group_"):
        self.pack_size = pack_size
        
    def pack(self, root_dir: str):
        # 扫描目录，移动文件夹，生成索引文件
        pass
```

---

## 4. 功能点详细定义 (Functional Specifications)

### 功能点 A: 智能结构维度识别与 K 点设置
*   **输入**: `ase.Atoms` 对象, `kspacing` (float), `vacuum_tol` (float).
*   **输出**: `kpoints` (List[int]), `structure_type` (Enum: Cluster/Layer/Bulk).
*   **逻辑**:
    1.  计算三个方向的真空层厚度。
    2.  若某方向真空层 > `vacuum_tol`，则该方向 K 点设为 1，否则设为 `ceil(2 * pi / (cell_len * kspacing))`。
    3.  若 3 方向均为真空 -> Cluster (Gamma Only)。
*   **性能要求**: 单结构处理时间 < 10ms。

### 功能点 B: 自动磁矩设置 (Auto-Magmom)
*   **输入**: `ase.Atoms`, `magmom_config` (Dict[Element, float]).
*   **逻辑**:
    1.  遍历原子，查表获取初始磁矩。
    2.  支持共线自旋 (Collinear) 设置。
*   **改进**: 支持从 Config 文件读取，而非硬编码。

### 功能点 C: ABACUS 输入生成
*   **输入**: `dft_params` (Dict), `pp/orb` 路径.
*   **输出**: `INPUT`, `STRU`, `KPT` 文件.
*   **逻辑**: 集成 `ase.io.abacus`，但在写入前注入上述自动推导的参数。

---

## 5. 迁移与开发计划 (Migration Plan)

### 阶段 1: 基础设施 (Priority: High)
1.  **提取 `utils.py`**: 将 `FeCHO_split_treat-ftv3.py` 中的 `judge_vaccum`, `set_kpoints`, `set_magmom_for_Atoms` 提取为独立函数，并编写单元测试。
2.  **实现 `JobPacker`**: 将 `subjob_dist.py` 封装为类，增加错误处理（如重名冲突检测）。

### 阶段 2: 核心生成器 (Priority: Medium)
1.  **实现 `AbacusGenerator`**: 重写主逻辑，剥离硬编码参数到 `config` 字典。
2.  **集成测试**: 使用现有 `FeCHO` 数据集验证生成的 `INPUT` 文件与原脚本输出是否一致 (Diff Test)。

### 阶段 3: 命令行入口 (Priority: Low)
1.  **新建 `runner/dpeva_labeling/run_gen_abacus.py`**: 提供 CLI 接口，接受 JSON 配置文件。
2.  **替换**: 正式废弃 `utils/fp` 下的旧脚本。

---

## 6. 风险与降级策略 (Risks)

| 风险点 | 描述 | 应对策略 |
| :--- | :--- | :--- |
| **ASE-ABACUS 依赖** | 项目强依赖 `ase-abacus` 提供的 `ase.io.abacus` 接口（特定 GitLab 源），而非标准 ASE 库。 | 在文档中明确安装指南 (`git clone https://gitlab.com/1041176461/ase-abacus.git`)，或在 `pyproject.toml` 中指定依赖源。 |
| **参数爆炸** | DFT 参数极其复杂，Config 文件可能过于庞大 | 提供 `template` 机制，允许用户仅覆盖差异参数 (`recursive_update`)。 |
| **路径硬编码** | 赝势库路径各机器不同 | 强制要求在 Config 中显式指定 `pseudo_dir`，或通过环境变量 `ABACUS_PP_PATH` 注入。 |

---

## 7. 测试用例设计 (Test Design)

### 单元测试 (Unit Tests)
*   `test_detect_dimensionality`: 构造人工晶胞 (Bulk, Slab, Cluster)，验证 `vaccum_status` 判定准确性。
*   `test_kpoint_generation`: 给定不同晶格常数，验证 K 点计算是否符合 `kspacing` 准则。
*   `test_magmom_setting`: 验证 Fe 原子是否被正确赋予磁矩。

### 回归测试 (Regression Tests)
*   **输入**: 标准 `dpdata` 样本。
*   **基准**: 使用原 `FeCHO_split_treat-ftv3.py` 生成的 `INPUT/KPT/STRU`。
*   **验证**: 新 `AbacusGenerator` 的输出文件内容应与基准完全一致（忽略空格/注释）。

---

## 8. 结论

将 Labeling 模块标准化是 DP-EVA 迈向通用 AI for Science 平台的关键一步。通过本次重构，我们将彻底消除硬编码脚本带来的维护负担，为未来支持更多 DFT 引擎（如 VASP, CP2K）打下坚实基础。
