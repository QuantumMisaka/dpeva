# CollectionWorkflow 代码审查与修复报告

## 1. 审查概述
本次审查重点针对 `dpeva/workflows/collect.py` 模块，旨在解决用户反馈的日志杂乱、默认值隐患以及 Slurm 后端潜在的冗余问题。我们对代码进行了深度清理和重构，并增加了严格的参数校验逻辑。

## 2. 问题分析与修复

### 2.1 日志杂乱与默认值问题
**问题描述**: 
用户指出日志中存在大量无意义的 "Dealing with..." 信息，且在 Auto 模式下会出现基于默认值 (lo=0.12) 的计算日志，造成误导。同时，Auto 模式的日志未能清晰展示参数边界 (Bounds) 的使用情况。

**根本原因**:
1. `__init__` 方法中存在防御性编程逻辑，当用户未指定 `lo` 时强行赋予 `0.12` 默认值。
2. 参数校验函数 `_validate_and_fill_trust_params` 被过早调用，导致在 Auto 模式真正计算前就输出了基于默认值的 "Calculated hi..." 日志。
3. 日志语句冗余，缺乏对关键配置（如 Auto-UQ Bounds）的显式输出。

**修复方案**:
1. **移除默认值**: 彻底删除了 `__init__` 中对 `lo=0.12` 的默认赋值。
2. **重构参数校验**: 
    * 引入 `uq_trust_mode` 的严格分支处理。
    * **Manual 模式**: 强制要求必须提供 `lo` 参数，否则抛出 `ValueError`。
    * **Auto 模式**: 允许 `lo` 为 `None`（将在运行时计算），不再进行提前的数值填充。
3. **优化日志输出**:
    * 删除了 5 行冗余的 "Dealing with..." 日志。
    * 在 Auto 模式计算前，显式打印当前使用的 Bounds (`uq_auto_bounds`) 和计算参数 (`ratio`, `width`)。
    * 明确了 "Auto-calculated" 和 "Final Trust Range" 的日志层级。

### 2.2 代码逻辑与冗余
**问题描述**: 
代码中存在历史遗留的 `_submit_to_slurm_legacy` 方法，该方法会生成临时的 `collect_config_frozen.json` 和 `run_collect_slurm.py` 封装脚本，不符合 v2.4+ 版本推崇的 "Self-Invocation"（自举）模式。

**修复方案**:
1. **删除遗留代码**: 彻底删除了 `_submit_to_slurm_legacy` 方法。
2. **强制规范**: 在 Slurm 模式下，强制要求必须传入 `config_path`（即必须通过配置文件启动），否则直接报错。这确保了所有 Slurm 任务都使用干净的自举模式提交，不再产生中间临时文件。

## 3. 单元测试验证
为了确保改动的正确性，我们新增了单元测试文件 `tests/unit/workflows/test_collect_logging_fix.py`，覆盖了以下场景：

| 测试用例 | 预期行为 | 结果 |
| :--- | :--- | :--- |
| `test_manual_mode_missing_lo` | 抛出 ValueError (Explicit Config原则) | ✅ 通过 |
| `test_manual_mode_valid` | 正常初始化，参数正确传递 | ✅ 通过 |
| `test_auto_mode_valid` | 初始化成功，`lo` 为 `None` (无默认值污染) | ✅ 通过 |
| `test_default_mode_fallback` | 未指定模式时默认为 Manual 并校验参数 | ✅ 通过 |
| `test_slurm_missing_config_path` | Slurm 模式下缺失 config_path 报错 | ✅ 通过 |

## 4. 结论
经过本次审查与修复，`CollectionWorkflow` 的逻辑更加严谨且符合 "The Zen of Python"：
* **Explicit**: 拒绝了隐式的 `0.12` 默认值，强制用户或算法显式指定。
* **Simple**: 移除了复杂的遗留提交逻辑，统一了 Slurm 提交路径。
* **Readable**: 日志信息现在准确反映了程序的运行逻辑，消除了误导性信息。

代码现已准备好合并到主分支。
