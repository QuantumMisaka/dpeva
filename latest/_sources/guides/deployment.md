---
title: GitHub Actions 部署指南 (Deployment Guide)
status: active
audience: Maintainers / DevOps
last-updated: 2026-09-06
owner: Trae AI Agent
---

# GitHub Actions 文档部署指南

本指南详细说明了如何使用 GitHub Actions 将 DP-EVA 的 Sphinx 文档自动构建并发布到 GitHub Pages。

## 1. 部署架构

我们采用 **GitHub Pages + GitHub Actions** 的自动化部署方案。

*   **触发条件**:
    *   `push` 到 `main` 分支: 部署为 `latest` 版本。
    *   `push` 到 `v*` 标签 (Tag): 部署为对应版本号 (如 `v0.6.0`)。
*   **构建环境**: `ubuntu-latest` + `python 3.x`。
*   **发布目标**: `gh-pages` 分支 (作为静态站点源)。

## 2. 前置配置 (Repository Settings)

在 GitHub 仓库设置中，需完成以下配置：

1.  进入 **Settings > Pages**。
2.  在 **Build and deployment** 部分：
    *   **Source**: 选择 `Deploy from a branch`。
    *   **Branch**: 选择 `gh-pages` 分支，目录选择 `/ (root)`。
3.  确保 `GITHUB_TOKEN` 拥有写入权限 (Settings > Actions > General > Workflow permissions > Read and write permissions)。

## 3. 工作流文件详解 (`.github/workflows/docs-deploy.yml`)

### 3.1 权限与前置门禁

工作流默认权限是 `contents: read`。main push、`v*` tag 和手动触发都必须先通过
`release-validation`；该 prerequisite 只调用 gate manifest 的共享 `release` profile，
不复制其命令。只有声明 `needs: release-validation` 的 `deploy` job 获得
`contents: write`，因此未经检查的触发不会写入 Pages 分支。

### 3.2 关键步骤说明

1.  **Release Validation**: 在只读权限下安装 release 所需依赖，并执行共享 `release` profile。
2.  **Checkout**: deploy job 使用 `fetch-depth: 0` 获取完整版本信息。
3.  **Build**: deploy job 通过 manifest 的 `docs_build` gate 生成静态文件。
4.  **Deploy**: 仅该依赖 job 使用 `peaceiris/actions-gh-pages`。
    *   `destination_dir`: 动态设置为 `latest` (对应 main 分支) 或 `vX.Y.Z` (对应 Tag)。
    *   `keep_files: true`: 增量发布，确保推送新版本时不会删除旧版本目录。

## 4. 多版本管理策略

当前采用**目录隔离**策略：

*   根目录 (`/`): 可放置一个重定向页 (`index.html`)，自动跳转到 `/latest/`。
*   `/latest/`: 始终对应 `main` 分支的最新构建。
*   `/v0.6.0/`, `/v0.5.0/`: 对应发布的 Tag 版本。

### 4.1 根目录重定向 (index.html)

建议在 `gh-pages` 分支根目录手动或通过 Action 维护一个 `index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url=latest/">
    <title>Redirecting...</title>
</head>
<body>
    <p>Redirecting to <a href="latest/">latest documentation</a>...</p>
</body>
</html>
```

## 5. 本地验证

在提交 Action 前，建议在本地模拟构建：

```bash
# 1. 清理旧构建
cd docs && make clean

# 2. 修复链接
python scripts/fix_links.py

# 3. 构建
make html

# 4. 预览 (需 Python)
python -m http.server --directory build/html 8000
# 访问 http://localhost:8000
```

## 6. 故障排查

*   **构建失败 (ImportError)**: 检查 `pip install -e .` 是否成功，确保 `src` 目录被正确加入 `PYTHONPATH`。
*   **样式丢失**: 检查 `_static` 目录是否被正确上传，通常由 `.nojekyll` 文件缺失导致 (Sphinx 主题常包含下划线开头的目录，GitHub Pages 默认会忽略它们)。
    *   *注*: `peaceiris/actions-gh-pages` 会自动创建 `.nojekyll`。

## 7. 后续优化

*   集成 `sphinx-multiversion` 插件，在侧边栏自动生成版本切换下拉菜单。
*   添加 PR 预览功能 (Deploy Preview)，利用 Netlify 或 GitHub Actions Artifacts。
