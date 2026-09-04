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

## Phase 0 checkpoint evidence (Task 5)

The focused R1/R2 negative behavior suite was run in the declared environment:

```text
conda run -n dpeva-dpa4 pytest tests/unit/run/test_status.py tests/unit/submission/test_guards.py tests/unit/workflows/test_workflow_completion_marker.py tests/unit/training/test_training_managers.py tests/unit/feature/test_execution_manager.py tests/unit/inference/test_inference_execution_manager.py -q
48 passed in 1.44s
```

The full integration suite was then run as required:

```text
conda run -n dpeva-dpa4 pytest tests/integration -q
1 failed, 8 passed, 7 skipped in 9.12s
```

Before the capability guard, the only failure was
`tests/integration/test_slurm_multidatapool_e2e.py::test_multidatapool_e2e[local]`.
The first actionable failure was in the generated script before `dp --pt
eval-desc` could run:

```text
/tmp/pytest-of-james/pytest-2214/test_multidatapool_e2e_local_0/work/desc_pool/run_evaldesc.sh: line 7: /opt/envs/deepmd3.1.2.env: No such file or directory
```

The exact local capability evidence was:

```text
python /home/james/apps/miniforge3/envs/dpeva-dpa4/bin/python
deepmd_spec ModuleSpec(.../site-packages/deepmd/...)
deepmd_version 3.2.0b1.dev67+g73de44b1f
dp_test /home/james/apps/miniforge3/envs/dpeva-dpa4/bin/dp
required_env=False
cuda_library=libcuda.so.1
ctypes.CDLL('libcuda.so') -> OSError: libcuda.so: cannot open shared object file: No such file or directory
```

`nvidia-smi` can see the host GPU, but that does not establish that the
DeepMD process can load the required CUDA library. The integration test now
performs this exact environment-file and DeepMD/libcuda probe before creating
the work directory or submitting/starting any job, and skips only when that
capability is unavailable. The prior Task 3 negative regression remains
classified above: a failing `dp test` must not be followed by a completion
marker or accepted artifacts.

The guard itself was focused-checked:

```text
conda run -n dpeva-dpa4 pytest tests/integration/test_slurm_multidatapool_e2e.py -q
2 skipped in 0.37s
SKIPPED: local DeepMD integration capability unavailable: required environment file is missing (/opt/envs/deepmd3.1.2.env)
SKIPPED: Set DPEVA_RUN_SLURM_ITEST=1 to enable Slurm integration tests
```

After the guard, the complete integration suite accounted for all 16 collected
tests with no failures:

```text
conda run -n dpeva-dpa4 pytest tests/integration -q
8 passed, 8 skipped in 3.46s
```

The eight skips were the existing five GPU-only DeepMD cases, one missing-data
labeling reproduction, one Slurm backend case without `DPEVA_RUN_SLURM_ITEST=1`,
and the local multidatapool case with the capability-based reason:

```text
local DeepMD integration capability unavailable: required environment file is missing (/opt/envs/deepmd3.1.2.env)
```

The local skip is emitted before `_source_data_root`, work directory creation,
or any workflow job creation. The repository-level checks also passed:

```text
conda run -n dpeva-dpa4 ruff check src tests scripts
All checks passed!

conda run -n dpeva-dpa4 pytest tests/unit -q
536 passed in 19.62s

git diff --check
(no output; exit 0)
```

All R1/R2 negative tests and the classified integration suite are therefore
green. Plan B is unblocked; Plans C–E remain subject to their subsequent
controller gates.

Phase 0 checkpoint: GO — R1/R2 negative tests and the classified integration suite pass.
