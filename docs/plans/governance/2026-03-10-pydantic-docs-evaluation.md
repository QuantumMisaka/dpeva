
# 评估报告：Sphinx 与 Pydantic 文档化现状

## 1. 现状调研
- **核心代码**: `src/dpeva/config.py` 使用 Pydantic V2 定义了 8 个配置模型（FeatureConfig, TrainingConfig, InferenceConfig, LabelingConfig, AnalysisConfig, CollectionConfig, SubmissionConfig, BaseWorkflowConfig）。
- **字段描述**: 每个字段都使用了 `Field(description="...")` 进行注释，这是自动生成文档的基础。
- **Sphinx 配置**: `docs/source/conf.py` 已经集成了 `sphinxcontrib.autodoc_pydantic` 扩展。
- **文档入口**: `docs/source/api/config.rst` 使用了 `.. automodule:: dpeva.config` 指令。

## 2. 发现的问题
尽管基础架构已经搭建，但目前的文档生成存在以下不足：

1.  **展示粒度不足**: `automodule` 指令虽然能生成文档，但默认情况下可能不会详细展开每个 Config 类的所有字段，或者展示样式不够友好（例如没有表格化展示，或者缺乏类型信息的直观呈现）。
2.  **缺乏分模块展示**: 所有配置都堆在一个 `config.rst` 页面中，用户难以快速定位特定 Workflow 的配置。用户通常关心的是“我要跑 Training 任务，需要配什么”，而不是一次性看所有配置。
3.  **与 Example 脱节**: 虽然代码中有字段描述，但缺乏具体的 JSON 示例片段，用户难以将 Pydantic 字段映射到实际的 JSON 配置文件结构中。
4.  **手动维护的 Reference**: 存在手动维护的 `docs/source/reference/config_schema.md`（已被标记为 Deprecated），这说明之前的自动文档体验不够好，导致开发者倾向于手写。

## 3. 改进方案评估
为了解决上述问题，我们需要优化 Sphinx 的配置和 RST 结构，而不是重写 Pydantic 模型。

- **可行性**: 高。Pydantic V2 与 `sphinxcontrib-autodoc_pydantic` 配合良好，可以通过调整 Sphinx 指令参数来优化输出（例如隐藏 JSON Schema，显示字段默认值，强制显示所有成员）。
- **统一入口**: 用户可调变量的统一入口就在 `src/dpeva/config.py`。我们应该强化这里作为 SSOT (Single Source of Truth)。

## 4. 下一步计划 (Design Plan)
1.  **重构 API 文档结构**: 不再使用单一的 `config.rst`，而是按 Workflow 拆分或在页面内通过 `.. autopydantic_model::` 指令单独渲染每个 Config 类。
2.  **优化渲染设置**: 在 `conf.py` 或指令中配置 `model_show_json=False`, `field_list_validators=False`, `field_show_default=True` 等，使文档更易读。
3.  **废弃手动文档**: 彻底移除 `docs/source/reference/config_schema.md`，并在其位置引用新的自动生成页面。
