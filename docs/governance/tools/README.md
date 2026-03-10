---
title: Tools（治理工具）
status: active
audience: Maintainers / Developers
last-updated: 2026-03-10
owner: Docs Owner
---

# Tools（治理工具）

本目录描述文档治理自动化能力与脚本职责边界，便于人类与 AI 统一执行门禁。

## 1. 核心脚本与流程

- 治理检查：`scripts/doc_check.py`
- 新鲜度检查：`scripts/check_docs_freshness.py`
- 严格构建：`make -C docs html SPHINXOPTS="-W --keep-going"`
- 链接检查：`make -C docs linkcheck SPHINXOPTS="-W --keep-going -D linkcheck_ignore='https://github.com/QuantumMisaka/dpeva/.*'"`

## 2. CI 对应关系

- 构建门禁：`.github/workflows/docs-check.yml`
- 治理门禁：`.github/workflows/doc-lint.yml`
