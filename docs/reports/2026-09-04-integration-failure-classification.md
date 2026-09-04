# Integration Failure Classification (2026-09-04)

## Scope

This report records the integration baseline for the reliable execution kernel
work and separates test-fixture drift from the inference false-success signal.

## Baseline

Command:

```text
conda run -n dpeva-dpa4 pytest tests/integration -q
```

The unmodified baseline exited with status `1` and reported:

```text
3 failed, 6 passed, 7 skipped in 162.11s (0:02:42)
```

The failed tests were:

- `tests/integration/test_e2e_cycle.py::test_e2e_cycle_label_integration_analysis`
- `tests/integration/test_slurm_multidatapool_e2e.py::test_multidatapool_e2e[local]`
- `tests/integration/test_v080_atst_acceptance.py::test_v080_explore_cli_acceptance_writes_manifest`

| Failure | Classification | Evidence | Disposition |
|---|---|---|---|
| label integration analysis | fixture/test-infrastructure drift | `LabelingWorkflow._run_extract_impl()` now unpacks three values from `extract_results()`, while the test configures obsolete `process_results.return_value` | update the mock to `extract_results.return_value = ([], [], [])` |
| local multidatapool inference | product false-success plus environment prerequisite mismatch | `dp test` raises while loading `libcuda.so`; generated `run_test.sh` continues to echo `DPEVA_TAG: WORKFLOW_FINISHED`, so `JobManager.submit()` returns success and expected `results.*.out` files are absent | retain as R1 RED evidence; fix fail-closed execution and artifact checks |
| explore CLI acceptance | fixture interpreter artifact | the fake executable uses `#!/usr/bin/env python` while the test deliberately limits `PATH` to the fake bin plus `/usr/bin` | render the fake shebang from `sys.executable` |

## Fixture repairs

- The labeling fixture now configures the current three-result
  `extract_results()` contract.
- The fake ATST executable now uses the active test interpreter via
  `#!{sys.executable}`, making it independent of a `python` alias in the
  restricted test `PATH`.

## Post-repair verification

```text
conda run -n dpeva-dpa4 pytest tests/integration/test_e2e_cycle.py -q
1 passed in 3.53s

conda run -n dpeva-dpa4 pytest tests/integration/test_v080_atst_acceptance.py -q
3 passed in 3.36s

conda run -n dpeva-dpa4 pytest tests/integration -q
1 failed, 8 passed, 7 skipped in 147.83s (0:02:27)
```

The remaining failure is only
`tests/integration/test_slurm_multidatapool_e2e.py::test_multidatapool_e2e[local]`.
Its generated `test.log` reports `RuntimeError: failed to compute neighbors:
Failed to load libcuda.so`, while `run_test.sh` still emits
`DPEVA_TAG: WORKFLOW_FINISHED` after the failing command and no
`results.*.out` artifacts are produced. This is intentionally preserved for
the execution fail-closed fix in Task 3.
