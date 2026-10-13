---
title: Document
status: active
audience: Developers
last-updated: 2026-10-13
owner: Workflow Owner
---

# 安装与环境准备（Installation）

- Status: active
- Audience: Users / Developers
- Last-Updated: 2026-10-13

## 1. 目的与范围

本页说明 DP-EVA 的安装方式、Python 依赖、以及运行工作流所需的外部依赖。

## 2. 相关方

- 使用者：在本地或集群环境安装并运行工作流
- 开发者：以可编辑模式安装并进行开发/测试
- 平台维护：提供 DeepMD、ABACUS 与 Slurm 环境

## 3. Python 环境要求

- Python：`>=3.10`
- 包名：`dpeva`

依赖定义以 [pyproject.toml](https://github.com/QuantumMisaka/dpeva/blob/main/pyproject.toml) 为准。

## 4. 从零准备 Local PyTorch 环境

下面以支持 CPU 的 PyTorch 后端为例。先获取源码，再在仓库根目录创建独立 Python 环境；已有检出的用户直接进入仓库即可：

```bash
git clone https://github.com/QuantumMisaka/dpeva.git
cd dpeva
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[deepmd]'
dpeva doctor
dp --version
```

DeepMD-kit 的上游[安装指南](https://docs.deepmodeling.com/projects/deepmd/en/stable/install/easy-install.html)
将 PyTorch 的 CPU 安装分为「先安装 CPU Torch，再安装 DeepMD-kit」；此处
`dpeva[deepmd]` 安装会解析后者的 3.2 运行时依赖。若要用 NVIDIA GPU，先按
上游指南选择与驱动匹配的 PyTorch/CUDA 安装方式，再安装 `dpeva[deepmd]` 并在
实际计算节点检查 `torch.cuda.is_available()`。默认依赖和 `deepmd` extra 本身只约束
DeepMD-kit 版本，不负责选择 PyTorch 的 CPU/CUDA wheel。

示例输入随源码仓库提供。安装后按 {doc}`Quickstart </guides/quickstart>` 使用
`examples/recipes/quickstart/` 中的相对路径配置运行四阶段 Local 链路。

## 5. 其它安装方式与依赖边界

### 5.1 Core：默认安装（含 DeepMD 兼容包络）

默认安装在核心依赖中提供有界的 `deepmd-kit>=3.1.2,<3.3`，用于保留现有
3.1.2 运行包络。这个依赖边界是安装兼容性承诺，不是新的科学能力验证；能力
矩阵和 3.2 证据仍由显式 lane 单独管理。在项目根目录执行：

核心依赖中的 `dpdata>=1.1` 是数据侧能力边界：读取 `deepmd/lmdb` 需要该下限，
`lmdb` 与 `msgpack` 由 dpdata 自身引入。预置环境（`--no-deps`）若仍是
`dpdata<1.1`，`deepmd/lmdb` 输入会被明确拒绝；`dpeva doctor` 的 `dpdata.lmdb`
检查会报告实际可读性。

```bash
python -m pip install -e .
```

验证：

```bash
dpeva --help
```

想使用仓库自带夹具跑最小闭环，请先按第 4 节选择 PyTorch 后端，再阅读
{doc}`Quickstart </guides/quickstart>`。

### 5.2 Dev：开发与测试依赖（可选）

```bash
python -m pip install -e '.[dev]'
```

`dev` extra 只提供测试、格式化和类型检查工具，不额外改变核心 DeepMD 包络或
安装 `atst-tools`。需要明确 3.2 lane 的测试时，显式叠加下一节的 runtime extra。

### 5.3 DeepMD runtime：显式 3.2 运行时 lane

默认安装已经提供 3.1.2 起的 DeepMD 兼容包络。训练、推理或特征工作流若要
显式解析 3.2 lane，在 core 安装上启用 DeepMD extra：

```bash
python -m pip install -e '.[deepmd]'
```

该 extra 的依赖范围是 `deepmd-kit>=3.2,<3.3`。这是用户环境的依赖解析
边界，不表示该范围内的每个版本行为完全等价。

验证运行时能力：

```bash
dpeva doctor
dp --version
```

`doctor` 是显式环境检查。稳定的 3.1.2--3.1.x 运行时会报告 `deepmd` 包络
可用，同时将 3.2 qualification 单独标为未声明；3.2 lane 则单独报告
qualification。旧版本、未来版本和 prerelease 不会绕过边界。

### 5.4 预置环境：由调用方承担全部依赖

如果平台已经预置并验证了完整运行环境，可以显式跳过依赖解析：

```bash
python -m pip install --no-deps -e .
```

`--no-deps` 不是无依赖的默认安装。调用方必须自行提供 `pyproject.toml` 中的
全部依赖（包括 `deepmd-kit`），并保存环境版本与 `dpeva doctor --json` 结果。

### 5.5 Research production：研究生产精确锁定

正式科研结果使用独立环境，并将 DeepMD 精确锁定为 `deepmd-kit==3.2.0`：

```bash
python -m pip install -e '.[deepmd]' 'deepmd-kit==3.2.0'
```

同时保存环境锁文件、`dpeva doctor --json` 和 `dp --version` 输出作为运行
记录。研究生产环境不能只依赖 `>=3.2,<3.3` 范围来声称可复现，也不能把范围内
其他版本未经验证的行为当作 3.2.0 等价物。

### 5.6 Exploration 可选依赖

`dpeva explore` 通过可选 `atst-tools` backend 调用轨迹探索工作流。该依赖不进入核心安装，需要时单独启用：

```bash
python -m pip install -e '.[explore]'
```

验证：

```bash
dpeva explore --help
atst --help
```

说明：

- `dpeva[explore]` 只安装 DP-EVA 的 exploration backend 依赖。
- ABACUS、DeePMD 模型文件、赝势和轨道文件仍由具体 ATST 配置与运行环境提供。

## 6. 预置集群环境

DP-EVA 的 DeepMD 工作流通过 `dp` 命令调用 DeepMD-kit（例如
`dp train/test/eval-desc`）。在 Slurm 环境中，建议通过
`submission.env_setup` 显式加载已锁定的 DeepMD 环境，不要依赖交互式 shell。
SAI 的 `dpeva-dpa4` 是项目已预置的测试环境；
{doc}`SAI 构建记录 </guides/developer/deepmd-kit-sai-build>` 以该环境和未入库的
`test/deepmd-kit` 源码检出为前提，记录的是**已有环境的升级与验证**，不是新账号
从零建环境的完整步骤。新账号先按本页第 4 节创建 Local 环境；集群 GPU 环境的
驱动、模块和队列配置应由集群维护者提供。

参考：

- {doc}`Slurm 指南 </guides/slurm>`
- {doc}`SAI 快速上手 </guides/sai-quickstart>`
- {doc}`SAI DeepMD 构建记录 </guides/developer/deepmd-kit-sai-build>`
- {doc}`DeepMD 依赖决策 </architecture/decisions/2026-02-04-deepmd-dependency>`
- {doc}`上游软件 </reference/upstream-software>`

## 7. 下一步

- 最短路径跑通：{doc}`Quickstart </guides/quickstart>`
- CLI 使用方式：{doc}`CLI 指南 </guides/cli>`
- 配置与路径解析：{doc}`配置指南 </guides/configuration>`
