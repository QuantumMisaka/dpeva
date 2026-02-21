# 功能-文档双向追踪矩阵（Feature ↔ Docs）

- Status: active
- Audience: Maintainers / Developers
- Last-Updated: 2026-02-19

本矩阵用于把“对外功能点”（CLI/配置/工作流输出约定）映射到文档章节，作为发布前的强制核对清单。

## 1. CLI 命令（对外接口）

| 功能点 | 代码实现 | 文档入口 |
|---|---|---|
| `dpeva train` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva infer` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva feature` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva collect` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva analysis` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |

## 2. 配置模型（Pydantic）

| 功能点 | 代码实现 | 文档入口 |
|---|---|---|
| 路径解析与提交后端 | `/src/dpeva/utils/config.py`、`/src/dpeva/config.py` | `/docs/guides/configuration.md` |
| `SubmissionConfig` | `/src/dpeva/config.py` | `/docs/reference/config-schema.md`、`/docs/reference/validation.md` |
| `TrainingConfig` | `/src/dpeva/config.py` | `/docs/reference/config-schema.md`、`/docs/reference/validation.md` |
| `InferenceConfig` | `/src/dpeva/config.py` | `/docs/reference/config-schema.md`、`/docs/reference/validation.md` |
| `FeatureConfig` | `/src/dpeva/config.py` | `/docs/reference/config-schema.md`、`/docs/reference/validation.md` |
| `CollectionConfig` | `/src/dpeva/config.py` | `/docs/reference/config-schema.md`、`/docs/reference/validation.md` |
| `AnalysisConfig` | `/src/dpeva/config.py` | `/docs/reference/config-schema.md`、`/docs/reference/validation.md` |

## 3. 工作流完成标记与可观测性（对外约定）

| 功能点 | 代码实现 | 文档入口 |
|---|---|---|
| 完成锚点 `DPEVA_TAG: WORKFLOW_FINISHED` | `/src/dpeva/constants.py` | `/docs/guides/slurm.md`、`/docs/guides/troubleshooting.md`、`/docs/guides/testing/integration-slurm.md` |
| Collect 在 Slurm 下可监控完成 | `/src/dpeva/workflows/collect.py` | `/docs/guides/testing/integration-slurm.md` |

## 4. Slurm 集成测试（对外交付）

| 功能点 | 代码实现 | 文档入口 |
|---|---|---|
| Slurm E2E Smoke（Multi DataPool） | `/tests/integration/test_slurm_multidatapool_e2e.py` | `/docs/guides/testing/integration-slurm.md` |
| 编排器与日志监控 | `/tests/integration/slurm_multidatapool/orchestrator.py` | `/docs/guides/testing/integration-slurm.md` |
| 输入裁剪（降本） | `/tests/integration/slurm_multidatapool/data_minimizer.py` | `/docs/guides/testing/integration-slurm.md`、`/docs/guides/testing/integration-slurm-plan.md` |

