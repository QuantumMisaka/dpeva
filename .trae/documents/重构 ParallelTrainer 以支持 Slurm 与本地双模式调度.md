# 重构任务调度系统：统一的本地/Slurm模板化方案

为了解决 Slurm 配置多样性问题，并同时支持本地与集群环境，我设计了一个基于 Python 原生 `string.Template` 的轻量级任务调度模块。该模块遵循 "Explicit & Simple" 原则，无需引入额外依赖。

## 1. 核心架构设计

新增子模块 `src/dpeva/submission`，包含以下组件：

### A. 模板引擎 (`templates.py`)
- **`TemplateEngine` 类**：
  - 封装 `string.Template`，支持基于 `${VAR}` 语法的安全替换。
  - 提供 `from_file()` (加载用户自定义模板) 和 `from_default()` (加载内置模板) 两种工厂方法。
- **内置模板**：
  - `DEFAULT_SLURM_TEMPLATE`: 包含标准的 `#SBATCH` 头部。
  - `DEFAULT_LOCAL_TEMPLATE`: 简单的 Bash 脚本头。
- **配置数据类 (`JobConfig`)**：
  - 定义统一的配置字段 (`job_name`, `command`, `partition`, `nodes` 等)。
  - 确保所有必要参数都有默认值，且易于扩展。

### B. 作业管理器 (`manager.py`)
- **`JobManager` 类**：
  - **初始化**: 接收 `mode` ("local"/"slurm") 和可选的 `custom_template_path`。
  - **`generate_script(config, output_path)`**: 渲染模板并保存为可执行文件。
  - **`submit(script_path)`**: 
    - Slurm 模式：调用 `sbatch script.sh`。
    - Local 模式：调用 `bash script.sh` (支持 `subprocess` 捕获输出)。

## 2. 详细实施计划

### 步骤 1: 实现 Submission 模块
- 创建 `src/dpeva/submission/` 目录。
- 实现 `templates.py` (模板类与配置类)。
- 实现 `manager.py` (生成与提交逻辑)。

### 步骤 2: 重构 ParallelTrainer
- 修改 `src/dpeva/training/trainer.py`：
  - 移除原有的硬编码 `train.sh` 生成逻辑。
  - 引入 `JobManager`。
  - 在 `setup_workdirs` 中调用 `manager.generate_script` 生成脚本。
  - 在 `train` 方法中调用 `manager.submit` 提交任务。
  - **关键点**: 
    - 本地模式保留原有的 `blocking` 等待逻辑（通过 `subprocess.Popen` 管理）。
    - Slurm 模式改为非阻塞提交（提交即完成，打印 Job ID）。

### 步骤 3: 更新工作流配置
- 更新 `TrainingWorkflow` (`src/dpeva/workflows/train.py`)：
  - 支持从 `config` 中读取 `submission` 相关参数（如 `template_path`, `slurm_partition` 等）。
  - 将这些参数传递给 `ParallelTrainer`。

## 3. 用户使用场景

### 场景 A: 默认本地运行 (无需配置)
```python
# 默认使用内置 Local 模板
config = {"mode": "local"} 
```

### 场景 B: 自定义 Slurm 模板
用户只需提供一个包含 `${job_name}`, `${command}` 等占位符的文本文件：
```bash
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -p gpu_partition
#SBATCH --gres=gpu:1
module load deepmd/2.0
${command}
```
然后在配置中指定路径即可：
```python
config = {
    "mode": "slurm",
    "template_path": "/path/to/my_slurm.tpl",
    "slurm_config": {"partition": "gpu_partition"} # 覆盖默认值
}
```

此方案完美兼顾了开箱即用的便捷性和高度定制的灵活性。
