---
title: Labeling 历史统计修复
status: active
audience: Developers
last-updated: 2026-03-14
owner: Labeling Maintainers
---

# Labeling 历史统计修复

当历史 `labeling` 工作目录存在 `task_meta.json` 缺失或损坏时，可使用离线修复脚本重建分支统计，不触发任何重新计算任务。

## 1. 脚本入口

- `scripts/repair_labeling_stats.py`

## 2. 单目录修复

```bash
python3 scripts/repair_labeling_stats.py --workdir /path/to/labeling_workdir
```

执行后会在目标目录写入：

- `outputs/labeling_stats_report.repaired.json`

## 3. 批量修复

```bash
python3 scripts/repair_labeling_stats.py \
  --root /path/to/history_runs \
  --pattern "fp*" \
  --strict
```

参数说明：

- `--workdir`：可重复传入多个目录
- `--root`：批量扫描根目录
- `--pattern`：根目录下匹配模式，默认 `*`
- `--strict`：任一目录失败或报告不可信时返回非零退出码

## 4. 输出与校验

脚本会输出每个目录的修复摘要：

- `trusted`：统计一致性是否通过
- `total/conv/fail/clean/filt`：全局指标
- `report`：修复报告文件路径

最终会输出统一 JSON 汇总，其中包含失败目录及原因列表，便于批处理审计。

## 5. 分支映射策略

修复流程按以下优先级识别分支归属：

1. `task_meta.json` 的 `dataset_name/stru_type`
2. 提取阶段保留的 `task_identity.json`
3. 历史 `outputs/task_identity_map.json`
4. 受控兜底 `unknown/unknown`

对于打包目录 `N_50_x`，不会再将其识别为 Dataset/Type。
