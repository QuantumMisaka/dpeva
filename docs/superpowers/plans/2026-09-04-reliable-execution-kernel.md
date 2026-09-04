---
title: Reliable Execution Kernel Implementation Plan
status: proposed
audience: Developers / AI Agents
last-updated: 2026-09-04
owner: Workflow Owner
---

# Reliable Execution Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local and Slurm-generated jobs fail closed, expose one canonical run-state machine, and turn the three observed integration failures into classified, reproducible evidence.

**Spec:** `docs/superpowers/specs/2026-09-04-project-governance-and-deepmd-3-2-design.html` (`#requirements` R1–R2, `#errors`, `#rollout` §15.1 and §15.9)

**Architecture:** Add a pure state-transition module and a small shell-command guard beneath the existing workflow managers. Preserve current workflow layouts and completion marker for compatibility, but emit it only after the command and artifact checks succeed. Repair test-fixture drift separately from product defects so Phase 0 evidence remains honest.

**Tech Stack:** Python 3.10+, Pydantic-independent dataclasses/enums, Bash, pytest, existing `JobManager` and workflow managers.

## Global Constraints

- This is the hard prerequisite for Plans B–E; do not start their implementation before the Phase 0 exit check passes.
- Keep `DPEVA_TAG: WORKFLOW_FINISHED` as a compatibility output, never as the source of truth (`#principles`, `#errors`).
- Do not add a scheduler framework, polling service, database, or workflow-wide refactor (`#goals`, §15.9).
- Treat the observed DeepMD CUDA failure as an environment prerequisite failure plus a product false-success defect; do not encode that specific CUDA message into product logic.
- Preserve user work and use recoverable deletion discipline.

---

### Task 1: Record and repair the integration baseline

**Files:**
- Create: `docs/reports/2026-09-04-integration-failure-classification.md`
- Modify: `tests/integration/test_e2e_cycle.py`
- Modify: `tests/integration/test_v080_atst_acceptance.py`

**Test strategy:**
- Behavior boundary: the suite distinguishes stale mocks and interpreter lookup artifacts from the inference false-success product defect.
- Existing suite to extend: `tests/integration/test_e2e_cycle.py`, `tests/integration/test_v080_atst_acceptance.py`, `tests/integration/test_slurm_multidatapool_e2e.py`.
- New test file justification: none; the failures already have owning integration modules.
- Temporary probes: `/tmp/pytest-of-james/**/test.log` may be inspected but must not be copied into the repository.

**Interfaces:**
- Consumes: `LabelingManager.extract_results() -> tuple[list[Path], list[Path], list[Path]]`, the active interpreter at `sys.executable`, and the existing integration command.
- Produces: a committed classification report and a baseline where only the known inference false-success remains RED before Task 3.

- [ ] **Step 1: Capture the current three-failure baseline**

Run: `conda run -n dpeva-dpa4 pytest tests/integration -q`

Expected: exit `1`; exactly these tests fail and the remaining result is `6 passed, 7 skipped`:

```text
tests/integration/test_e2e_cycle.py::test_e2e_cycle_label_integration_analysis
tests/integration/test_slurm_multidatapool_e2e.py::test_multidatapool_e2e[local]
tests/integration/test_v080_atst_acceptance.py::test_v080_explore_cli_acceptance_writes_manifest
```

- [ ] **Step 2: Write the classification report**

Create the report with this evidence table and the exact baseline command/output summary:

```markdown
| Failure | Classification | Evidence | Disposition |
|---|---|---|---|
| label integration analysis | fixture/test-infrastructure drift | `LabelingWorkflow._run_extract_impl()` now unpacks three values from `extract_results()`, while the test configures obsolete `process_results.return_value` | update the mock to `extract_results.return_value = ([], [], [])` |
| local multidatapool inference | product false-success plus environment prerequisite mismatch | `dp test` raises while loading `libcuda.so`; generated `run_test.sh` continues to echo `DPEVA_TAG: WORKFLOW_FINISHED`, so `JobManager.submit()` returns success and expected `results.*.out` files are absent | retain as R1 RED evidence; fix fail-closed execution and artifact checks |
| explore CLI acceptance | fixture interpreter artifact | the fake executable uses `#!/usr/bin/env python` while the test deliberately limits `PATH` to the fake bin plus `/usr/bin` | render the fake shebang from `sys.executable` |
```

- [ ] **Step 3: Repair the stale labeling mock**

Replace the obsolete mock setup with:

```python
label_manager.prepare_tasks.return_value = [job_bundle]
label_manager.extract_results.return_value = ([], [], [])
```

Run: `conda run -n dpeva-dpa4 pytest tests/integration/test_e2e_cycle.py -q`

Expected: `1 passed`.

- [ ] **Step 4: Make the fake ATST executable interpreter-explicit**

Replace the fake executable body with:

```python
fake_atst.write_text(
    f"#!{sys.executable}\n"
    "from pathlib import Path\n"
    "Path('result.extxyz').write_text('1\\nProperties=species:S:1:pos:R:3\\nH 0 0 0\\n', encoding='utf-8')\n",
    encoding="utf-8",
)
```

Run: `conda run -n dpeva-dpa4 pytest tests/integration/test_v080_atst_acceptance.py -q`

Expected: all tests in the module pass.

- [ ] **Step 5: Confirm the remaining RED signal is the product defect**

Run: `conda run -n dpeva-dpa4 pytest tests/integration -q`

Expected: only `test_multidatapool_e2e[local]` fails; its generated `test.log` shows an external-command failure and its generated script still prints the completion marker.

- [ ] **Step 6: Commit the classification and fixture repairs**

```bash
git add docs/reports/2026-09-04-integration-failure-classification.md tests/integration/test_e2e_cycle.py tests/integration/test_v080_atst_acceptance.py
git commit -m "test: classify integration baseline failures"
```

Expected: one commit containing only the report and the two fixture corrections.

### Task 2: Add the canonical state-transition kernel

**Files:**
- Create: `src/dpeva/run/__init__.py`
- Create: `src/dpeva/run/status.py`
- Create: `tests/unit/run/test_status.py`

**Test strategy:**
- Behavior boundary: only the seven states and declared transitions in SPEC `#errors` are accepted; resume/recovery remain event names, not states.
- Existing suite to extend: none; no current suite owns a reusable run-state boundary.
- New test file justification: `tests/unit/run/test_status.py` establishes the independently runnable state-machine contract later consumed by Run Manifest.
- Temporary probes: none.

**Interfaces:**
- Consumes: no repository runtime services.
- Produces: `RunState`, `RunEventKind`, `InvalidStateTransition`, and `transition(current: RunState, target: RunState, event: RunEventKind | None = None) -> RunState`.

- [ ] **Step 1: Write the failing state tests**

```python
import pytest

from dpeva.run.status import (
    InvalidStateTransition,
    RunEventKind,
    RunState,
    transition,
)


def test_canonical_state_values_are_closed() -> None:
    assert {state.value for state in RunState} == {
        "created", "validated", "submitted", "running", "partial", "failed", "finished"
    }


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunState.CREATED, RunState.VALIDATED),
        (RunState.VALIDATED, RunState.RUNNING),
        (RunState.VALIDATED, RunState.SUBMITTED),
        (RunState.SUBMITTED, RunState.RUNNING),
        (RunState.RUNNING, RunState.PARTIAL),
        (RunState.RUNNING, RunState.FAILED),
        (RunState.RUNNING, RunState.FINISHED),
        (RunState.PARTIAL, RunState.FAILED),
    ],
)
def test_normal_transitions(current: RunState, target: RunState) -> None:
    assert transition(current, target) is target


def test_recovery_requires_an_explicit_event() -> None:
    with pytest.raises(InvalidStateTransition):
        transition(RunState.FAILED, RunState.RUNNING)
    assert transition(RunState.FAILED, RunState.RUNNING, RunEventKind.RECOVERY) is RunState.RUNNING


def test_finished_is_terminal() -> None:
    with pytest.raises(InvalidStateTransition):
        transition(RunState.FINISHED, RunState.RUNNING, RunEventKind.RESUME)
```

- [ ] **Step 2: Run the new tests to verify RED**

Run: `pytest tests/unit/run/test_status.py -q`

Expected: collection fails because `dpeva.run.status` does not exist.

- [ ] **Step 3: Implement the pure transition module**

```python
from __future__ import annotations

from enum import Enum


class RunState(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    RUNNING = "running"
    PARTIAL = "partial"
    FAILED = "failed"
    FINISHED = "finished"


class RunEventKind(str, Enum):
    RESUME = "resume"
    RECOVERY = "recovery"


class InvalidStateTransition(ValueError):
    pass


_NORMAL_TRANSITIONS = {
    RunState.CREATED: {RunState.VALIDATED, RunState.FAILED},
    RunState.VALIDATED: {RunState.SUBMITTED, RunState.RUNNING, RunState.FAILED},
    RunState.SUBMITTED: {RunState.RUNNING, RunState.FAILED},
    RunState.RUNNING: {RunState.PARTIAL, RunState.FAILED, RunState.FINISHED},
    RunState.PARTIAL: {RunState.FAILED},
    RunState.FAILED: set(),
    RunState.FINISHED: set(),
}


def transition(
    current: RunState,
    target: RunState,
    event: RunEventKind | None = None,
) -> RunState:
    if target in _NORMAL_TRANSITIONS[current]:
        return target
    if current in {RunState.FAILED, RunState.PARTIAL} and target is RunState.RUNNING:
        if event in {RunEventKind.RESUME, RunEventKind.RECOVERY}:
            return target
    raise InvalidStateTransition(f"illegal run-state transition: {current.value} -> {target.value}")
```

Export the four public symbols from `src/dpeva/run/__init__.py`.

- [ ] **Step 4: Run the state tests to verify GREEN**

Run: `pytest tests/unit/run/test_status.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the state kernel**

```bash
git add src/dpeva/run/__init__.py src/dpeva/run/status.py tests/unit/run/test_status.py
git commit -m "feat: add canonical run state machine"
```

### Task 3: Make generated jobs fail closed and artifact-aware

**Files:**
- Create: `src/dpeva/submission/guards.py`
- Create: `tests/unit/submission/test_guards.py`
- Modify: `src/dpeva/submission/templates.py`
- Modify: `src/dpeva/training/managers.py`
- Modify: `src/dpeva/inference/managers.py`
- Modify: `src/dpeva/feature/managers.py`
- Modify: `tests/unit/workflows/test_workflow_completion_marker.py`
- Modify: `tests/unit/feature/test_execution_manager.py`
- Modify: `tests/unit/inference/test_inference_execution_manager.py`
- Modify: `tests/unit/training/test_training_managers.py`

**Test strategy:**
- Behavior boundary: a failing command or missing expected artifact returns non-zero and never prints the finished marker.
- Existing suite to extend: workflow completion-marker and manager execution suites.
- New test file justification: `test_guards.py` owns the shared command/validation rendering boundary used by three managers.
- Temporary probes: none.

**Interfaces:**
- Consumes: `WORKFLOW_FINISHED_TAG` and shell commands already quoted by `DPCommandBuilder`.
- Produces: `guarded_command(command: str, artifact_checks: list[str]) -> str` and fail-closed default templates.

- [ ] **Step 1: Write failing shell-behavior tests**

```python
import subprocess

from dpeva.submission.guards import guarded_command


def test_failed_command_does_not_emit_finished(tmp_path) -> None:
    script = tmp_path / "fail.sh"
    script.write_text("#!/bin/bash\nset -Eeuo pipefail\n" + guarded_command("false", []))
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "DPEVA_TAG: WORKFLOW_FINISHED" not in result.stdout


def test_missing_artifact_does_not_emit_finished(tmp_path) -> None:
    check = f"test -s {tmp_path / 'missing.out'}"
    result = subprocess.run(
        ["bash", "-c", "set -Eeuo pipefail\n" + guarded_command("true", [check])],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "DPEVA_TAG: WORKFLOW_FINISHED" not in result.stdout
```

Run: `pytest tests/unit/submission/test_guards.py -q`

Expected: collection fails because `dpeva.submission.guards` does not exist.

- [ ] **Step 2: Implement the shared guard and strict templates**

```python
from dpeva.constants import WORKFLOW_FINISHED_TAG


def guarded_command(command: str, artifact_checks: list[str]) -> str:
    lines = [command, *artifact_checks, f'printf "%s\\n" "{WORKFLOW_FINISHED_TAG}"']
    return "\n".join(lines) + "\n"
```

Add `set -Eeuo pipefail` immediately after `#!/bin/bash` in both `DEFAULT_LOCAL_TEMPLATE` and `DEFAULT_SLURM_TEMPLATE`. Do not alter custom templates; document that a custom template must provide equivalent fail-closed semantics.

- [ ] **Step 3: Replace unconditional marker concatenation with concrete checks**

Use `guarded_command()` in each manager with these checks:

```python
# training/managers.py, after dp_train_cmd and dp_freeze_cmd are assembled
environment_lines = [
    f"export OMP_NUM_THREADS={omp_threads}",
    f"export DP_INTER_OP_PARALLELISM_THREADS={max(1, omp_threads // 2)}",
    f"export DP_INTRA_OP_PARALLELISM_THREADS={omp_threads}",
]
cmd = guarded_command(
    command="\n".join([*environment_lines, dp_train_cmd, dp_freeze_cmd]),
    artifact_checks=["test -s model.ckpt.pt", "test -s lcurve.out"],
)

# inference/managers.py
cmd = guarded_command(
    command=dp_test_cmd,
    artifact_checks=[f"compgen -G {shlex.quote(results_prefix + '.*.out')} >/dev/null"],
)

# feature/managers.py
checks = (
    [f"test -s {shlex.quote(output_hdf5)}"]
    if feature_exporter == "embed"
    else [f"find {shlex.quote(abs_output_dir)} -type f -name '*.npy' -print -quit | grep -q ."]
)
cmd = guarded_command(command=cmd, artifact_checks=checks)
```

Import `shlex` where needed. Update completion-marker unit tests to assert ordering (`command`, artifact check, marker) rather than mere marker presence.

- [ ] **Step 4: Run focused manager and guard tests**

Run: `pytest tests/unit/submission/test_guards.py tests/unit/workflows/test_workflow_completion_marker.py tests/unit/feature/test_execution_manager.py tests/unit/inference/test_inference_execution_manager.py tests/unit/training/test_training_managers.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit fail-closed script generation**

```bash
git add src/dpeva/submission/guards.py src/dpeva/submission/templates.py src/dpeva/training/managers.py src/dpeva/inference/managers.py src/dpeva/feature/managers.py tests/unit/submission/test_guards.py tests/unit/workflows/test_workflow_completion_marker.py tests/unit/feature/test_execution_manager.py tests/unit/inference/test_inference_execution_manager.py tests/unit/training/test_training_managers.py
git commit -m "fix: make generated jobs fail closed"
```

### Task 4: Propagate local child and recursive feature failures

**Files:**
- Modify: `src/dpeva/training/managers.py`
- Modify: `src/dpeva/feature/managers.py`
- Modify: `tests/unit/training/test_training_managers.py`
- Modify: `tests/unit/feature/test_execution_manager.py`

**Test strategy:**
- Behavior boundary: any failed training child or feature leaf makes the owning local workflow fail; partial output remains for diagnosis.
- Existing suite to extend: manager unit suites.
- New test file justification: none.
- Temporary probes: none.

**Interfaces:**
- Consumes: `multiprocessing.Process.exitcode` and `_compute_feature()` exceptions.
- Produces: `WorkflowError` with failed task indices or failed leaf paths.

- [ ] **Step 1: Add failing training exit-code tests**

```python
from dpeva.utils.exceptions import WorkflowError


@patch("dpeva.training.managers.multiprocessing.Process")
def test_submit_jobs_rejects_failed_child(mock_process, manager) -> None:
    first = MagicMock(exitcode=0)
    second = MagicMock(exitcode=7)
    mock_process.side_effect = [first, second]
    with pytest.raises(WorkflowError, match=r"training tasks failed: \[1\]"):
        manager.submit_jobs(["a.sh", "b.sh"], ["a", "b"], blocking=True)
```

Run: `pytest tests/unit/training/test_training_managers.py::TestTrainingExecutionManager::test_submit_jobs_rejects_failed_child -q`

Expected: FAIL because failed child exit codes are ignored.

- [ ] **Step 2: Aggregate child exit codes after every join**

```python
for process in processes:
    process.join()
failed = [index for index, process in enumerate(processes) if process.exitcode != 0]
if failed:
    raise WorkflowError(f"training tasks failed: {failed}")
self.logger.info("All local training tasks completed.")
```

Import `WorkflowError` from `dpeva.utils.exceptions`.

- [ ] **Step 3: Add and implement feature leaf aggregation**

Add a test with two fake leaf systems where `_compute_feature()` raises for one. Assert `run_local_python_recursion()` raises `WorkflowError` containing the failed path and does not log the completion marker.

Implement one shared failure list for the recursive traversal:

```python
failures: list[str] = []

def process_recursive(current_path: str, current_output_dir: str) -> None:
    if io_manager.is_leaf_system(current_path):
        try:
            desc = self._compute_feature(generator, current_path, output_mode, feature_kind)
            out_file = current_output_dir + ".npy"
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.save(out_file, desc)
        except Exception as exc:
            self.logger.error("Failed to process %s: %s", current_path, exc)
            failures.append(current_path)
        return
    for child in sorted(os.listdir(current_path)):
        child_path = os.path.join(current_path, child)
        if os.path.isdir(child_path):
            process_recursive(child_path, os.path.join(current_output_dir, child))

# after traversal
if failures:
    raise WorkflowError(f"feature generation failed for {len(failures)} system(s): {failures}")
```

- [ ] **Step 4: Run focused failure-propagation tests**

Run: `pytest tests/unit/training/test_training_managers.py tests/unit/feature/test_execution_manager.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit local failure propagation**

```bash
git add src/dpeva/training/managers.py src/dpeva/feature/managers.py tests/unit/training/test_training_managers.py tests/unit/feature/test_execution_manager.py
git commit -m "fix: propagate local workflow failures"
```

### Task 5: Prove the Phase 0 exit and set the stop/go checkpoint

**Files:**
- Modify: `docs/reports/2026-09-04-integration-failure-classification.md`
- Modify: `docs/guides/cli.md`
- Modify: `docs/governance/traceability/workflow-contract-test-matrix.md`

**Test strategy:**
- Behavior boundary: documentation no longer advises consumers to treat the marker alone as success, and the full integration suite has no unexplained failure.
- Existing suite to extend: full unit and integration suites.
- New test file justification: none.
- Temporary probes: none.

**Interfaces:**
- Consumes: Tasks 1–4 behavior and SPEC Phase 0 exit signal.
- Produces: a Phase 0 decision record: `GO` only when R1/R2 evidence is green; otherwise `STOP` with the failing command.

- [ ] **Step 1: Update the completion contract documentation**

State in both documents that the marker means “command and artifact checks completed” for newly guarded paths, but downstream orchestration must also require process/job success; Slurm submission alone remains `submitted`.

Insert this normative paragraph in the completion/status sections of both files:

```markdown
`WORKFLOW_FINISHED` is written only after the guarded command returns zero and all declared artifacts pass validation. Consumers MUST require both a successful process/job state and the marker; the marker alone is not proof of success. `sbatch` returning a JobID establishes only `submitted`, not `finished`.
```

- [ ] **Step 2: Run the focused negative behavior suite**

Run: `pytest tests/unit/run/test_status.py tests/unit/submission/test_guards.py tests/unit/workflows/test_workflow_completion_marker.py tests/unit/training/test_training_managers.py tests/unit/feature/test_execution_manager.py tests/unit/inference/test_inference_execution_manager.py -q`

Expected: all tests pass, including failed-command, missing-artifact, child-failure, and illegal-transition cases.

- [ ] **Step 3: Run the integration suite in the declared environment**

Run: `conda run -n dpeva-dpa4 pytest tests/integration -q`

Expected: no failures; pass and skip totals account for all 16 collected tests, and no failure is hidden by a completion marker. If the local DeepMD fixture still lacks a usable runtime capability, mark that test with a capability-based skip before job creation and record the exact doctor evidence—do not restore false success.

- [ ] **Step 4: Run the repository-level checks**

Run: `ruff check src tests scripts && pytest tests/unit -q && git diff --check`

Expected: all commands exit `0`.

- [ ] **Step 5: Record the checkpoint and commit**

Append `Phase 0 checkpoint: GO — R1/R2 negative tests and the classified integration suite pass.` when all exit checks pass. Otherwise append `Phase 0 checkpoint: STOP` followed by the literal failing command and its first actionable failure; Plans B–E remain blocked.

```bash
git add docs/reports/2026-09-04-integration-failure-classification.md docs/guides/cli.md docs/governance/traceability/workflow-contract-test-matrix.md
git commit -m "docs: record reliable execution checkpoint"
```

Expected: Phase 0 has an evidence-backed `GO` or `STOP`; only `GO` unblocks Plan B.
