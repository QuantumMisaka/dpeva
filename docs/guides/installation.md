---
title: Document
status: active
audience: Developers
last-updated: 2026-07-01
owner: Workflow Owner
---

# 安装与环境准备（Installation）

- Status: active
- Audience: Users / Developers
- Last-Updated: 2026-07-01

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

## 4. 安装方式

### 4.1 Core：基础安装（不含 DeepMD）

核心安装适用于不调用 DeepMD 的数据、标注、分析及文档工作。它不安装
`deepmd-kit`，因此没有 `dp` 命令也可以导入 `dpeva` 并查看 CLI 帮助。
在项目根目录执行：

```bash
python -m pip install -e .
```

验证：

```bash
dpeva --help
```

### 4.2 Dev：开发与测试依赖（可选）

```bash
python -m pip install -e '.[dev]'
```

`dev` extra 只提供测试、格式化和类型检查工具，不隐式安装 DeepMD 或
`atst-tools`。需要 DeepMD 的测试时，显式叠加下一节的 runtime extra。

### 4.3 DeepMD runtime：用户运行时依赖（可选）

训练、推理或特征工作流需要 `dp` 时，在 core 安装上显式启用 DeepMD：

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

`doctor` 是显式环境检查；缺少 `dp` 时 core 安装与 `import dpeva` 仍应可用，
需要 DeepMD 的具体工作流才会在执行阶段报告缺失能力。

### 4.4 Research production：研究生产精确锁定

正式科研结果使用独立环境，并将 DeepMD 精确锁定为 `deepmd-kit==3.2.0`：

```bash
python -m pip install -e '.[deepmd]' 'deepmd-kit==3.2.0'
```

同时保存环境锁文件、`dpeva doctor --json` 和 `dp --version` 输出作为运行
记录。研究生产环境不能只依赖 `>=3.2,<3.3` 范围来声称可复现，也不能把范围内
其他版本未经验证的行为当作 3.2.0 等价物。

### 4.5 Exploration 可选依赖

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

## 5. 外部环境说明

DP-EVA 的 DeepMD 工作流通过 `dp` 命令调用 DeepMD-kit（例如
`dp train/test/eval-desc`）。在 Slurm 环境中，建议通过
`submission.env_setup` 显式加载已锁定的 DeepMD 环境，不要依赖交互式 shell。

参考：

- /docs/guides/slurm.md
- /docs/guides/developer/deepmd-kit-sai-build.md
- /docs/architecture/decisions/2026-02-04-deepmd-dependency.md
- /docs/reference/upstream-software.md

## 6. 下一步

- 最短路径跑通：/docs/guides/quickstart.md
- CLI 使用方式：/docs/guides/cli.md
- 配置与路径解析：/docs/guides/configuration.md
