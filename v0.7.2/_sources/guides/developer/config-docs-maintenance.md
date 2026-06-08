---
title: 变量说明文档维护指南
status: active
audience: Developers
last-updated: 2026-03-10
---

# 变量说明文档维护指南

本文档指导开发者如何维护 `src/dpeva/config.py` 中的 Pydantic 配置模型，以确保 `Configuration Reference` 自动文档保持最新和准确。

## 1. 自动生成原理

我们的文档系统通过 `sphinxcontrib-autodoc_pydantic` 扩展，直接从 `src/dpeva/config.py` 读取 Pydantic 模型定义生成 HTML。

- **源文件**: `src/dpeva/config.py` (Single Source of Truth)
- **文档文件**: `docs/source/api/config/*.rst` (负责编排顺序)
- **生成产物**: `docs/build/html/api/config.html`

## 2. 如何添加/修改配置项

只需在 Python 代码中修改，文档会自动同步。

### 2.1 添加新字段

在 `config.py` 的对应 Config 类中添加字段，并**必须**包含 `description` 参数。

```python
class LabelingConfig(BaseWorkflowConfig):
    # ...
    new_param: int = Field(
        default=10, 
        description="这是一个新参数的说明。请务必详细描述其作用。"
    )
```

### 2.2 修改字段说明

直接修改 `description` 字符串。支持简单的 reStructuredText 语法（如 `` `code` ``）。

### 2.3 废弃字段

不要直接删除字段（为了兼容性），应标记为 `Deprecated`。

```python
    old_param: Optional[str] = Field(
        None, 
        description="[DEPRECATED] 该参数已废弃，请使用 `new_param`。"
    )
```

## 3. 文档构建与预览

在本地预览文档效果：

```bash
cd docs
make html
# 打开 docs/build/html/api/config.html 查看
```

## 4. 性能基准

- **构建时间**: 增量构建 < 2s，全量构建 < 10s。
- **依赖**: `sphinx`, `sphinx-book-theme`, `autodoc-pydantic`, `myst-parser`。

## 5. 常见问题

- **文档未更新**: 请确保运行了 `make html`，且 `config.py` 已保存。
- **格式错误**: `description` 中如果包含复杂 RST 语法（如列表），请确保缩进正确，或使用三引号字符串。
