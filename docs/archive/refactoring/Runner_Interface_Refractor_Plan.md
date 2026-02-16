### DP-EVA 工作流启动方案调研与演进报告

> **[DEPRECATED]** This report is for historical reference only.

本报告基于对 `dpeva` 当前架构（Runner 模式）的深度审计与未来演进需求的分析，系统对比了“Python 脚本启动（现有方案）”与“统一命令行 CLI（拟议方案）”两种技术路线。

---

#### 1. 执行摘要 (Executive Summary)

**推荐结论**：采用 **“CLI 为主，Python API 为辅”** 的双轨演进策略。

*   **普通用户/标准化场景**：统一使用 `dpeva` CLI 入口。这提供了简洁、一致的用户体验（`dpeva train`, `dpeva infer`），便于文档化、教程编写以及与 Slurm/K8s 等调度系统的集成。
*   **专业用户/复杂编排场景**：保留并增强 Python API (`dpeva.workflows.*`)。对于需要动态参数生成、条件控制流、循环迭代等复杂逻辑的高级用户，直接编写 Python 脚本调用核心类库是更灵活的选择。
*   **Runner 脚本的定位转变**：现有的 `runner/` 目录将从“官方推荐入口”转型为“API 使用范例（Recipes）”，并在未来迁移至 `examples/` 目录，不再承担核心工具职责。

这一策略既实现了工程化的规范性，又保留了科研探索所需的灵活性。

---

#### 2. 方案对比矩阵 (Comparison Matrix)

评分说明：★★★★★ (优秀/原生支持) | ★★★☆☆ (中等/需额外工作) | ★☆☆☆☆ (较差/困难)

| 维度 | 方案一：Runner 脚本 (现状) | 方案二：统一 CLI (推荐) | 核心差异点 |
| :--- | :--- | :--- | :--- |
| **1. 长期演进 (链式调用)** | ★★★★★ | ★★★☆☆ | Python 脚本天然支持逻辑编排；CLI 需依赖外部 Shell 或 DAG 工具串联。 |
| **2. 维护性与扩展性** | ★★☆☆☆ | ★★★★★ | Runner 模式入口分散，代码重复；CLI 模式入口统一，模块化注册，维护成本低。 |
| **3. 用户体验 (一致性)** | ★★☆☆☆ | ★★★★★ | CLI 提供统一的 `dpeva <cmd>` 界面和帮助文档；Runner 需用户记忆脚本路径。 |
| **4. 配置表达能力** | ★★★★★ | ★★★☆☆ | Python 可动态计算参数（如根据时间生成种子）；CLI 依赖静态配置文件。 |
| **5. 环境与依赖管理** | ★★★☆☆ | ★★★★★ | CLI 更符合 Docker/K8s 的 Entrypoint 设计规范，易于打包发布。 |
| **6. 测试与 CI/CD** | ★★★☆☆ | ★★★★★ | CLI 易于进行黑盒集成测试；Runner 脚本通常缺乏独立测试覆盖。 |
| **7. 安全与合规** | ★★★☆☆ | ★★★★☆ | CLI 强制分离配置与代码，减少 Hardcode 风险；便于统一审计日志。 |
| **8. 性能与资源** | ★★★★☆ | ★★★★☆ | 两者启动开销差异微乎其微；CLI 可统一进行资源检查（Pre-flight check）。 |

---

#### 3. 深度维度分析

##### 3.1 用户体验与一致性
*   **现状痛点**: 用户需记忆 `python dpeva/runner/dpeva_train/run_train.py config.json` 这样冗长的命令，且不同脚本的参数处理可能存在细微差异。
*   **CLI 优势**: `dpeva train config.json` 简洁明了，支持 Tab 自动补全，提供统一的 `--help` 文档，降低了认知负荷。
*   **角色适配**: 
    *   **初级用户**: 仅需掌握 CLI 和标准 JSON 配置。
    *   **高级用户**: 当 CLI 无法满足需求时（如参数需动态生成），可无缝切换到 Python API，参考 Recipes 编写自定义脚本。

##### 3.2 可维护性与工程化
*   **代码复用**: CLI 模式将配置加载、路径解析、日志初始化、异常处理等公共逻辑集中在 `dpeva.cli` 模块，消除了 Runner 脚本中的大量重复代码。
*   **扩展成本**: 新增 Workflow 只需编写核心类并在 CLI 中注册一行子命令，无需复制粘贴整个 Runner 脚本。
*   **部署友好**: 在 Docker 容器中，`ENTRYPOINT ["dpeva"]` 比指定特定的 Python 脚本更通用。

##### 3.3 复杂场景支持
*   虽然 CLI 在处理复杂逻辑（如循环、条件分支）时不如 Python 脚本灵活，但这正是保留 Python API 的意义所在。
*   对于绝大多数标准任务（训练、推理、生成特征），CLI +静态配置完全足够且更加健壮。

---

#### 4. CLI 实现架构规划

将在 `src/dpeva/cli.py` 中实现基于 `argparse` 的子命令架构：

```python
# src/dpeva/cli.py (Prototype)
import argparse
import sys
import logging
from dpeva.utils.config import resolve_config_paths
# 延迟导入 Workflow 类以加快 CLI 启动速度
# from dpeva.workflows import ... 

def setup_global_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - DPEVA - %(levelname)s - %(message)s')

def run_workflow(workflow_class, config_path):
    # 通用加载逻辑
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    config = resolve_config_paths(config, config_path)
    
    workflow = workflow_class(config)
    workflow.run()

def handle_train(args):
    from dpeva.workflows import TrainingWorkflow
    run_workflow(TrainingWorkflow, args.config)

def handle_infer(args):
    from dpeva.workflows import InferenceWorkflow
    run_workflow(InferenceWorkflow, args.config)

# ... 其他 handlers (feature, collect, analysis) ...

def main():
    setup_global_logging()
    parser = argparse.ArgumentParser(prog="dpeva", description="DP-EVA: Deep Potential Evolution Accelerator")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available Workflows")

    # Training Sub-command
    p_train = subparsers.add_parser("train", help="Run Training Workflow")
    p_train.add_argument("config", help="Path to configuration JSON")
    p_train.set_defaults(func=handle_train)

    # Inference Sub-command
    p_infer = subparsers.add_parser("infer", help="Run Inference Workflow")
    p_infer.add_argument("config", help="Path to configuration JSON")
    p_infer.set_defaults(func=handle_infer)
    
    # Feature, Collect, Analysis ...

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

#### 5. 迁移与演进路线图

为确保平滑过渡，建议分三个阶段实施：

**阶段一：CLI 引入与共存 (已完成)**
1.  实现 `src/dpeva/cli.py` 并注册 `dpeva` 命令。
2.  在 `pyproject.toml` 中添加 `[project.scripts]` 入口。
3.  保持现有 `runner/` 脚本不变，确保老用户习惯不受影响。

**阶段二：Runner 转型与引导 (已完成)**
1.  修改 `runner/` 下的脚本，使其内部逻辑简化为直接调用 `dpeva.cli` 中的处理函数（或 Workflow 类），消除代码重复。
2.  在 Runner 脚本执行时打印 `FutureWarning`，提示用户未来建议使用 CLI。
3.  更新文档，将 CLI 提升为首选使用方式。

**阶段三：架构清理 (已完成)**
1.  将 `runner/` 目录重命名为 `examples/recipes/`。
2.  移除其中的“脚本”属性（如 `if __name__ == "__main__":` 块），将其转化为纯粹的 Python API 调用示例代码。
3.  正式确立 CLI 为标准入口，Python API 为高级开发接口。


每个阶段开发完毕之后，你都需要审查一遍代码，确保项目代码遵循如下项目规范
1. 所有用户可调变量遵循统一入口，内部变量变量定义清晰明确，且不存在冗余变量。
2. 所有单元测试用例均可正常通过
3. 本项目开发文档得到同步更新，确保其反映项目最新状态。

---

#### 6. 风险评估

*   **风险**: 用户习惯阻力。
    *   **对策**: 长期保留 Python 脚本运行方式（作为 Example），不强制废弃，仅在文档中改变推荐优先级。
*   **风险**: CLI 参数解析灵活性不足。
    *   **对策**: 保持 JSON 配置文件的结构灵活性，未来可考虑支持 YAML 或在 JSON 中引入简单的变量替换。

#### 7. 结论

**立即启动 CLI 改造是提升 DP-EVA 工程化水平的必经之路。** 推荐采用“CLI + Python API”双轨制，既满足了普通用户对简洁、统一工具链的诉求，又保留了专业用户所需的编程灵活性。
