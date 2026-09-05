# Task 1 report — Python compatibility bridges

## Revision

- Implementation commit: `925ade24f65c7233182fc06b75ee9072529547ee`
  (`fix: preserve v0.8 Python configuration and helper APIs`)
- Worktree: `.worktrees/compat-082`, branch `fix/compat-082`

## Diagnosis and integration decisions

- Public workflow and analysis Pydantic boundaries accepted flat documented
  submission aliases only through `migrate_legacy_config`, preserving strict
  unknown-field rejection, equal-value acceptance, conflict failure, and input
  nonmutation. Exploration-native `backend="atst-tools"` remains untouched.
- `load_and_resolve_config(path)` now preserves the historical `dict` return;
  `load_config_with_metadata(path)` owns `MigrationResult` evidence for infer and
  feature handlers. The recovery script consumes the dict helper explicitly.
- `DPCommandBuilder` restores `_backend`, `set_backend`, and historical
  positional command signatures. New stateless callers can use keyword-only
  `backend`; workflow managers continue to use injected `DeepMDAdapter`.
- `check_deepmd_version()` remains explicit and deprecated, returning the doctor
  result and adding an actionable `UserWarning` for non-OK probes.
- Configuration, CLI, and recipe documentation was synchronized.

## Evidence mapping

| Criterion | Evidence |
|---|---|
| RED detects missing behavior | `task-1-red.log`, exit code 1: 6 failed and 6 setup errors against the newly added compatibility cases; raw output retained. |
| Baseline before test additions | `task-1-baseline.log`, exit code 0: original acceptance suite 37 passed. |
| Focused GREEN | `task-1-green-focused.log`, exit code 0: 43 passed, 6 warnings. |
| Exact acceptance command | `task-1-green.log`, exit code 0: 43 passed, 6 warnings. |
| Syntax check | `python -m compileall -q src/dpeva/config.py src/dpeva/config_migration.py src/dpeva/cli.py src/dpeva/utils/command.py src/dpeva/utils/env_check.py scripts/fp11_1344_recover_after_false_finish.py`, exit code 0. |
| Diff hygiene | `git diff --check`, exit code 0 before commit. |

Exact acceptance command:

```text
python -m pytest tests/unit/test_config_migration.py tests/unit/test_cli.py tests/unit/utils/test_backend_config.py tests/unit/utils/test_env_check.py tests/unit/utils/test_config_paths.py tests/unit/scripts/test_fp11_1344_recover.py -q
```

## Remaining concern

The broader manager check had 46 passed and one failure in the pre-existing
`tests/unit/feature/test_execution_manager.py` case that calls the facade with
the removed explicit-backend positional form (`DPCommandBuilder.embed("pt", model=...)`).
This is outside the brief's owned files and exact acceptance command; the scoped
facade now intentionally follows the historical API and manager production code
uses `DeepMDAdapter` directly.
