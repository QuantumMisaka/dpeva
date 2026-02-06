# 技术决策报告：DeepMD-kit 依赖管理方案评估

**日期**: 2026-02-04
**状态**: 已评审 (Approved)
**决策**: 🔴 **拒绝采用 Git Submodule**，推荐采用 **外部环境依赖 + 版本约束** 模式。

---

## 1. 现状分析：DeepMD-kit 调用场景与耦合度

通过对 `dpeva` 代码库的全面审计，我们梳理了项目对 `deepmd-kit` 的依赖情况：

### 1.1 调用方式
项目通过两种方式与 `deepmd-kit` 交互，呈现 **松耦合** 特征：

1.  **CLI 命令行调用 (主要方式)**
    *   **模块**: `src/dpeva/utils/command.py` (DPCommandBuilder)
    *   **场景**:
        *   `dp train`: 模型训练 (Trainer)
        *   `dp freeze`: 模型导出 (Trainer)
        *   `dp test`: 模型推理与精度验证 (Inference)
        *   `dp eval-desc`: 描述符计算 (Feature/Collection)
    *   **特征**: 依赖系统 `PATH` 中的 `dp` 可执行文件，通过 `subprocess` 或作业调度系统（Slurm）调用。不依赖 Python 源码。

2.  **Python API 调用 (可选方式)**
    *   **模块**: `src/dpeva/feature/generator.py`
    *   **场景**: `DescriptorGenerator` 的 `mode="python"`。
    *   **代码**:
        ```python
        try:
            from deepmd.infer.deep_pot import DeepPot
        except ImportError:
            _DEEPMD_AVAILABLE = False
        ```
    *   **特征**: 仅用于本地直接计算描述符。若环境未安装 Python 包，代码会自动降级或禁用该功能，不影响核心 CLI 流程。

### 1.2 定制化需求
*   **当前代码**: 未发现对 `deepmd-kit` 核心算法（如 `DescriptSeA`, `FittingNet`）的源码级修改或 Patch。
*   **需求性质**: `dpeva` 定位为 **上层工作流编排系统 (Orchestrator)**，而非 `deepmd-kit` 的二次开发或插件。

---

## 2. 方案评估：Git Submodule vs 现有模式

### 2.1 采用 Git Submodule 的权衡
若将 `deepmd-kit` 作为 submodule 引入：

*   **优点**:
    *   **绝对版本锁定**: 可精确锁定到某次 Commit，确保所有开发者使用完全一致的代码快照。
    *   **源码级调试**: 方便在开发 `dpeva` 时直接跳转调试 `deepmd-kit` 内部代码。
    *   **私有定制**: 若未来需修改 `deepmd-kit` C++ 核心且无法合并回上游，submodule 是管理 Fork 的好方法。

*   **缺点 (阻碍性因素)**:
    *   **构建复杂性极高**: `deepmd-kit` 包含大量 C++ 代码，依赖 TensorFlow/PyTorch C++ 库。要求所有 `dpeva` 用户在安装时本地编译 `deepmd-kit` 是不现实的（通常通过 Conda/Pip 安装预编译包）。
    *   **仓库体积膨胀**: `deepmd-kit` 仓库较大，增加 `git clone` 时间。
    *   **用户体验差**: 普通用户只想 `pip install dpeva`，不希望处理子模块同步和编译错误。
    *   **环境冲突**: Submodule 强制指定版本可能与用户环境中已安装的 `deepmd-kit` (如系统级 Conda 环境) 冲突。

### 2.2 官方发布版本 (Pip/Conda) 的权衡
*   **优点**:
    *   **安装便捷**: `pip install deepmd-kit` 即可获取预编译 Wheel 包。
    *   **解耦**: `dpeva` 只关注如何调用，不关注如何构建。
    *   **兼容性**: 允许用户根据硬件（CUDA 版本）自由选择匹配的 DeepMD 版本。

---

## 3. 决策结论

基于上述分析，我们 **强烈不建议** 使用 Git Submodule。

**核心理由**:
1.  **构建成本过高**: `dpeva` 是 Python 纯代码项目，引入需编译的 C++ 子模块会破坏项目的轻量级特性。
2.  **定位不符**: `dpeva` 是 `deepmd-kit` 的 **使用者** 而非 **扩展者**。
3.  **版本稳定性**: 通过 `pyproject.toml` 的版本约束足以满足稳定性需求。

---

## 4. 实施方案：替代依赖管理策略

既然拒绝 Submodule，我们需要更规范地管理外部依赖，防止“环境不一致”导致的问题。

### 4.1 依赖声明规范化
目前 `deepmd-kit` 未在 `pyproject.toml` 中声明。建议根据使用程度分级声明：

**修改 `pyproject.toml` (建议)**:

```toml
[project]
name = "dpeva"
# ...
dependencies = [
    "dpdata>=0.2.13",
    "numpy",
    "pandas",
    "pydantic>=2.0"
]

[project.optional-dependencies]
# 将 deepmd-kit 列为可选依赖，因为核心功能可以通过远程 Slurm 调用（此时本地不需要安装 deepmd）
# 或者用户可能使用 conda 安装的 deepmd
local = [
    "deepmd-kit>=2.2.0" 
]
```

### 4.2 运行时版本检查 (Runtime Version Check)
在 `dpeva` 启动时增加版本检查逻辑，确保环境中的 `dp` 命令版本符合要求。

**建议新增模块**: `src/dpeva/utils/env_check.py`

```python
import subprocess
from packaging import version

MIN_DEEPMD_VERSION = "2.0.0"

def check_deepmd_version():
    try:
        # 检查 CLI 版本
        out = subprocess.check_output(["dp", "--version"], text=True).strip()
        # 解析输出 "DeePMD-kit v2.2.9"
        v_str = out.split()[-1].lstrip("v")
        if version.parse(v_str) < version.parse(MIN_DEEPMD_VERSION):
            print(f"WARNING: DeepMD-kit version {v_str} is older than {MIN_DEEPMD_VERSION}.")
    except FileNotFoundError:
        print("WARNING: 'dp' command not found in PATH. Ensure DeepMD-kit is loaded.")
```

### 4.3 CI/CD 策略
在 CI 流程中，直接使用 `pip` 或 `conda` 安装官方发布的 `deepmd-kit` 进行集成测试，而非从源码构建。

```yaml
# GitHub Actions 示例
steps:
  - uses: actions/checkout@v3
  - name: Install Dependencies
    run: |
      pip install .[local]
      pip install deepmd-kit  # 使用官方预编译包
```

### 4.4 文档说明
在 `README.md` 和安装文档中明确说明：
> `dpeva` 依赖 `deepmd-kit`。推荐使用 Conda 安装：
> `conda install deepmd-kit=*=*cpu -c deepmodeling` (开发环境)
> 生产环境请根据 CUDA 版本选择对应的 GPU 包。

---

## 5. 总结
本项目不需要 Submodule。通过 **规范化 `pyproject.toml` 依赖声明** + **运行时版本检查** + **DPCommandBuilder 统一接口**，即可实现对 `deepmd-kit` 的高效管理和稳定调用，同时保持项目的轻量化和易用性。
