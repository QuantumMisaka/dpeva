# 安装与环境准备（Installation）

- Status: active
- Audience: Users / Developers
- Last-Updated: 2026-02-19

## 1. 目的与范围

本页说明 DP-EVA 的安装方式、Python 依赖、以及运行工作流所需的外部依赖（DeepMD-kit）。

## 2. 相关方

- 使用者：在本地或集群环境安装并运行工作流
- 开发者：以可编辑模式安装并进行开发/测试
- 平台维护：提供 DeepMD 与 Slurm 环境

## 3. Python 环境要求

- Python：`>=3.8`
- 包名：`dpeva`

依赖定义以 [pyproject.toml](/pyproject.toml) 为准。

## 4. 安装方式

### 4.1 可编辑安装（推荐用于开发/使用）

在项目根目录执行：

```bash
python -m pip install -e .
```

验证：

```bash
dpeva --help
```

### 4.2 开发依赖（可选）

```bash
python -m pip install -e '.[dev]'
```

## 5. 外部依赖：DeepMD-kit

DP-EVA 的多数 Workflow 依赖 DeepMD-kit 的 `dp` 命令（例如 `dp train/test/eval-desc`）。

要求：

- `dp` 命令可在 `PATH` 中找到

验证：

```bash
dp --version
```

说明：

- 若 `dp` 不可用，导入 `dpeva` 时会给出警告提示，但并不阻止安装。
- 在 Slurm 环境中，建议通过 `submission.env_setup` 显式加载 DeepMD 环境（不要依赖交互式 shell）。

参考：

- /docs/guides/slurm.md
- /docs/architecture/decisions/2026-02-04-deepmd-dependency.md

## 6. 下一步

- 最短路径跑通：/docs/guides/quickstart.md
- CLI 使用方式：/docs/guides/cli.md
- 配置与路径解析：/docs/guides/configuration.md
