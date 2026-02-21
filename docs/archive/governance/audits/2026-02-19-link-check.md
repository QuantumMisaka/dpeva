# 链接检测报告（Docs Link Check）

- Status: archived
- Audience: Maintainers / Developers
- Last-Updated: 2026-02-19

## 1. 检测范围与规则

- 范围：`/docs/**/*.md`
- 允许的链接类型：
  - 项目内：以 `/` 开头的根相对路径（例如 `/docs/README.md`、`/src/dpeva/cli.py`）
  - 外部链接：`http(s)://...`
  - 页面内锚点：`#...`
- 禁止的链接类型：
  - 文件系统绝对路径（例如 `file://...`、`/abs/path`）
  - 目录相对路径（例如 `../..`、`./foo`）

说明：由于运行环境未安装 Node.js，未使用 `markdown-link-check`；本仓库采用等价的静态链接解析与存在性校验脚本完成全量检测。

## 2. 检测结果

- Markdown 文件数：50
- 项目内链接检查数：128
- 断链数：0
- 外部链接数：0

## 3. 修复记录（本轮）

- 统一将项目内 Markdown 链接标准化为根相对路径（以 `/` 开头）
- 清理并替换历史 `file:///...` 绝对路径链接
- 删除旧入口文档，避免多入口导致内容分叉

