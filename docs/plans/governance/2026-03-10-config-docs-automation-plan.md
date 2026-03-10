---
title: 变量说明文档自动化配置方案
status: active
audience: Developers
last-updated: 2026-03-10
---

# 变量说明文档自动化配置方案

## 1. 目标
通过 `sphinxcontrib-autodoc_pydantic` 扩展，将 `src/dpeva/config.py` 中的 Pydantic 模型自动渲染为清晰、易读的 HTML 文档，使其成为项目配置参数的 Single Source of Truth (SSOT)，替代手动维护的 Markdown 文件。

## 2. 核心设计原则
1.  **自动化**: 只要代码变动，文档构建时自动更新。
2.  **可视化**: 隐藏复杂的 JSON Schema，突出字段名称、类型、默认值和描述。
3.  **模块化**: 按功能（Workflow）组织配置说明，而非简单的类列表。
4.  **关联性**: 在文档中提供跳转链接，关联到具体的 Example 文件。

## 3. 具体实施步骤

### 3.1 调整 Sphinx 全局配置 (`docs/source/conf.py`)
优化 `autodoc_pydantic` 的默认行为，使其更适合阅读：
- `autodoc_pydantic_model_show_json = False` (默认不展开 JSON，避免页面过长)
- `autodoc_pydantic_settings_show_json = False`
- `autodoc_pydantic_field_list_validators = False` (默认不列出校验器，除非必要)
- `autodoc_pydantic_field_show_constraints = False` (简化约束显示)
- `autodoc_pydantic_model_show_config_summary = False` (不显示 ConfigDict 摘要)
- `autodoc_pydantic_field_show_default = True` (显示默认值)
- `autodoc_pydantic_field_show_required = True` (标记必填项)

### 3.2 重构 `docs/source/api/config.rst`
将其拆分为更细粒度的结构，或者使用 `.. autopydantic_model::` 指令手动编排顺序，并添加说明文字。

**新结构建议**:
```rst
Configuration Reference
=======================

.. toctree::
   :maxdepth: 2
   :caption: By Workflow

   config/common
   config/feature
   config/training
   config/inference
   config/collection
   config/labeling
   config/analysis
```
或者在一个文件中使用多个 Section。

### 3.3 编写各模块 RST 文件
例如 `docs/source/api/config/labeling.rst`:
```rst
Labeling Configuration
======================

用于第一性原理计算（Labeling Workflow）的配置参数。

.. autopydantic_model:: dpeva.config.LabelingConfig
   :model-show-json: False
   :field-list-validators: False
   :members:
   :undoc-members:
   :exclude-members: model_config
```

### 3.4 替换旧引用
修改 `docs/source/reference/index.rst`，指向新的 API 文档位置，并移除 `config_schema`。

## 4. 集成测试方案
编写一个简单的 Sphinx 构建测试脚本，检查生成的 HTML 是否包含关键字段（如 `dft_params`, `integration_enabled` 等）。

## 5. 部署指南
更新 `docs/guides/developer-guide.md`，说明如何更新配置文档（即：修改代码 docstring 即可）。
