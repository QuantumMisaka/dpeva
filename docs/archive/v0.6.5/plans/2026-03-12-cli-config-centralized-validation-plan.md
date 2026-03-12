## CLI config.json 集中校验实施计划（2026-03-12）

### 目标
- 在 `cli.py` 入口为 `train/infer/feature/collect/analysis/label` 统一增加 `config` 文件存在性与合法性前置校验。
- 对用户输入错误实现友好报错，避免进入 workflow 后才抛出 traceback。

### 实施步骤
1. 在 `src/dpeva/cli.py` 新增统一校验函数 `validate_config_path`：
   - 校验路径存在、是文件、可读。
   - 校验后缀建议为 `.json`，异常时给出清晰提示。
   - 对 `prepare/execute/postprocess` 被误填为 `config` 时追加提示“可能需要 `--stage`”。
2. 将六个子命令的 `add_argument("config", ...)` 统一改为 `type=validate_config_path`，实现 argparse 阶段拦截。
3. 在 `load_and_resolve_config` 增加 JSON 解析异常转译：
   - 捕获 `json.JSONDecodeError`，输出包含文件路径与错误位置的可读信息。
4. 在 `main()` 异常出口做错误分层：
   - 用户输入类错误（参数/路径/JSON）仅输出简洁错误，不打印 traceback。
   - 其他内部异常保留 traceback。
5. 增加/更新单测（`tests/unit/test_cli.py`）：
   - 不存在 config 时解析阶段失败。
   - `dpeva label prepare` 给出 `--stage` 引导。
   - 非法 JSON 给出解析错误。
   - 保证既有分发测试继续通过。
6. 同步文档（`docs/guides/cli.md`）：
   - 增加 config 前置校验行为与典型误用示例。

### 验证计划
- `pytest tests/unit/test_cli.py`
- 手工验证：
  - `dpeva train not_exist.json`
  - `dpeva label prepare`
  - `dpeva label valid.json --stage prepare`
- 若修改文档，执行：
  - `python3 scripts/check_docs.py`
  - `python3 scripts/check_docs_freshness.py --days 90`
  - `make -C docs html SPHINXOPTS="-W --keep-going"`

### 完成标准
- 六个子命令统一在 CLI 阶段完成 config 文件校验。
- 非法输入不再进入 workflow 执行。
- 错误提示可读且可操作，单测全绿。
