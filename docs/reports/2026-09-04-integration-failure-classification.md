---
title: Integration Failure Classification
status: active
audience: Developers / AI Agents
last-updated: 2026-09-04
owner: Project Maintainer
---

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

## Phase 0 checkpoint evidence (Task 5; initial checkpoint superseded)

The focused R1/R2 negative behavior suite was run in the declared environment:

```text
conda run -n dpeva-dpa4 pytest tests/unit/run/test_status.py tests/unit/submission/test_guards.py tests/unit/workflows/test_workflow_completion_marker.py tests/unit/training/test_training_managers.py tests/unit/feature/test_execution_manager.py tests/unit/inference/test_inference_execution_manager.py -q
48 passed in 1.65s
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

The rejected guard itself was focused-checked:

```text
conda run -n dpeva-dpa4 pytest tests/integration/test_slurm_multidatapool_e2e.py -q
2 skipped in 0.37s
SKIPPED: local DeepMD integration capability unavailable: required environment file is missing (/opt/envs/deepmd3.1.2.env)
SKIPPED: Set DPEVA_RUN_SLURM_ITEST=1 to enable Slurm integration tests
```

The initial capability guard was rejected in review because a historical
`/opt/envs/deepmd3.1.2.env` path and a `libcuda.so` filename probe are not a
portable runtime capability. That guard was removed. The local fixture now
derives its setup from the current `sys.executable` environment by prepending
that interpreter's `bin` directory to `PATH`, so the generated `dp` command
uses the same runtime as the test runner.

The complete integration suite was rerun without a local capability skip:

```text
conda run -n dpeva-dpa4 pytest tests/integration -q
1 failed, 9 passed, 7 skipped in 126.82s (0:02:06)
```

The seven skips were the existing five GPU-only DeepMD cases, one missing-data
labeling reproduction, and one Slurm backend case without
`DPEVA_RUN_SLURM_ITEST=1`. The local multidatapool case executed and failed in
inference; its generated `run_test.sh` returned non-zero and its `test.log`
recorded:

```text
RuntimeError: failed to compute neighbors: Failed to load libcuda.so. Try appending the directory containing this library to your $LD_LIBRARY_PATH environment variable.
```

This is a real executed workflow failure, not a skip. The guarded command
returned exit status 1 before producing `results.*.out`, and no completion
marker was accepted. Therefore the integration suite is not green and the
repository-level checks do not establish a Phase 0 GO.

The portable setup behavior was first exposed by this RED test:

```text
conda run -n dpeva-dpa4 pytest tests/integration/test_slurm_multidatapool_e2e.py::test_local_runtime_setup_is_derived_from_current_interpreter -q
F                                                                        [100%]
E       assert []
1 failed in 0.44s
```

After implementation, the same test was GREEN:

```text
conda run -n dpeva-dpa4 pytest tests/integration/test_slurm_multidatapool_e2e.py::test_local_runtime_setup_is_derived_from_current_interpreter -q
.                                                                        [100%]
1 passed in 0.37s
```

The repository-level checks were run after the fixture change and passed:

```text
conda run -n dpeva-dpa4 ruff check src tests scripts
All checks passed!

conda run -n dpeva-dpa4 pytest tests/unit -q
536 passed in 21.04s

git diff --check
(no output; exit 0)
```

The R1/R2 negative tests and repository checks are green, but the required
local integration chain is not. Per the controller ruling, a true executed
DeepMD failure cannot be converted to a capability skip. Plans B–E remain
blocked.

Phase 0 checkpoint: STOP

Failing command:

```text
conda run -n dpeva-dpa4 pytest tests/integration -q
```

First actionable failure:

```text
RuntimeError: failed to compute neighbors: Failed to load libcuda.so. Try appending the directory containing this library to your $LD_LIBRARY_PATH environment variable.
```

## Review fix round 2: current-runtime CUDA driver setup

The fix-round 1 STOP was resolved using new environment evidence. The local
harness now derives an optional driver directory from
`Path(shutil.which("nvidia-smi")).resolve().parent` and appends a quoted
`LD_LIBRARY_PATH` export only when that same directory contains
`libcuda.so`. No driver path is hard-coded, no library-name load probe is used,
and the setup is never a skip or success criterion. The local configs continue
to use the current `sys.executable` bin directory for `dp`.

The directed tests first exposed the missing setup (RED):

```text
conda run -n dpeva-dpa4 pytest tests/integration/test_slurm_multidatapool_e2e.py -k 'runtime_setup and driver' -q
F.                                                                       [100%]
1 failed, 1 passed, 3 deselected in 0.42s
```

After the implementation both driver-presence branches passed (GREEN):

```text
conda run -n dpeva-dpa4 pytest tests/integration/test_slurm_multidatapool_e2e.py -k 'runtime_setup and driver' -q
..                                                                       [100%]
2 passed, 3 deselected in 0.41s
```

Without any `DPEVA_TEST_ENV_SETUP` override, the exact local E2E passed:

```text
env -u DPEVA_TEST_ENV_SETUP conda run -n dpeva-dpa4 pytest tests/integration/test_slurm_multidatapool_e2e.py::test_multidatapool_e2e[local] -q
.                                                                        [100%]
1 passed in 197.48s (0:03:17)
```

The complete integration suite also passed with no external setup override:

```text
env -u DPEVA_TEST_ENV_SETUP conda run -n dpeva-dpa4 pytest tests/integration -q
.....ssssss....s...                                                      [100%]
12 passed, 7 skipped in 190.25s (0:03:10)
```

All 19 collected tests are accounted for. The seven skips are the existing
five GPU-only DeepMD cases, the opt-in labeling reproduction without data, and
the Slurm case without `DPEVA_RUN_SLURM_ITEST=1`; the local multidatapool E2E
is an executed pass.

R1/R2 focused suite:

```text
conda run -n dpeva-dpa4 pytest tests/unit/run/test_status.py tests/unit/submission/test_guards.py tests/unit/workflows/test_workflow_completion_marker.py tests/unit/training/test_training_managers.py tests/unit/feature/test_execution_manager.py tests/unit/inference/test_inference_execution_manager.py -q
................................................                         [100%]
48 passed in 1.74s
```

Repository checks:

```text
conda run -n dpeva-dpa4 ruff check src tests scripts
All checks passed!

conda run -n dpeva-dpa4 pytest tests/unit -q
536 passed in 20.71s

git diff --check
(no output; exit 0)
```

The prior STOP is superseded: the same-runtime local chain now executes and
passes, all integration artifacts and completion markers are checked by the
test, and all negative/repository checks are green.

Phase 0 checkpoint: GO — R1/R2 negative tests and the classified integration suite pass.
