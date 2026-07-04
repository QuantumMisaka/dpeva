---
title: Slurm Array Backend Refactor Implementation Plan
status: active
audience: Developers / Operators / AI Agents
last-updated: 2026-07-04
owner: Workflow Owner
---

# Slurm Array Backend Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor DP-EVA Slurm submission support so homogeneous chunked workloads use Slurm job arrays, starting with `labeling` execution bundles, while preserving existing single-job behavior for training, inference, feature, collect, and analysis unless explicitly opted in later.

**Architecture:** Add generic array rendering, manifest execution, job-id parsing, and Slurm status polling under `src/dpeva/submission/`. Then route `LabelingWorkflow` Slurm execution through one array per attempt. Keep `JobManager.submit()` and current workflow contracts backward compatible. Monitoring moves from ad hoc `squeue` polling in `workflows/labeling.py` into reusable submission-layer methods.

**Tech Stack:** Python 3.12, Pydantic v2, dataclasses, Slurm `sbatch --array`, `squeue --array`, `sacct -P`, pytest with mocked subprocess calls, SAI Slurm conventions.

---

## Research Confirmation

SAI staff's suggestion is correct for the current labeling execution pattern.

The Slurm upstream documentation defines job arrays as a mechanism for collections of similar batch jobs, all sharing the same initial options. Array indices are submitted with `--array`; each element receives `SLURM_ARRAY_TASK_ID`; `%` limits simultaneous running array tasks; and array stdout/stderr can use `%A` and `%a` for array job id and task id. This matches DP-EVA labeling bundles: each packed `N_<tasks_per_job>_<idx>` directory runs the same ABACUS runner with the same partition/QoS/resources, differing only by working directory.

Relevant sources:

- SchedMD Job Array Support: https://slurm.schedmd.com/job_array.html
- SchedMD `sbatch --array`: https://slurm.schedmd.com/sbatch.html
- SchedMD `squeue --array`, `%F`, `%K`: https://slurm.schedmd.com/squeue.html
- SchedMD `sacct -P`, `JobID`, `State`, `ExitCode`: https://slurm.schedmd.com/sacct.html
- Princeton Research Computing Slurm guide, Job Arrays section: https://researchcomputing.princeton.edu/support/knowledge-base/slurm

SAI-specific constraints from `sai-user-guide` that must stay true:

- Do not add memory fields or CPU fields by default for SAI jobs. The backend currently only renders `cpus_per_task` when explicitly configured; preserve that behavior.
- GPU QoS limits still apply to each array element. Use `--array=0-N%M` to bound concurrently running elements under SAI QoS limits such as `rush-1o2gpu`, `rush-gpu`, or `flood-1o2gpu`.
- Keep resource fields explicit and user controlled through `submission.slurm_config`.

## Slurm Array Adoption Matrix

Use array now:

- `LabelingWorkflow.run_execute`: yes. Current code submits one Slurm job per packed bundle via `_submit_job_dirs()` and then polls all job IDs. These are homogeneous ABACUS bundle jobs and are the primary target of the refactor.

Add backend support now, but do not switch workflow behavior:

- `TrainingExecutionManager.submit_jobs`: array-compatible in principle because each model replica is one similar GPU job, but the number of replicas is usually small, each job is long-running, existing docs expect per-model `train.slurm`/`train.out`, and Slurm blocking wait is not currently implemented. Keep one job per model for now.
- `InferenceExecutionManager.submit_jobs`: array-compatible in principle because each model test job has the same resources, but model-specific directories/logs are user-facing and the job count is normally small. Keep current behavior for now.

Do not use array in this refactor:

- `FeatureExecutionManager.submit_cli_job`: one Slurm job currently loops sub-pools for `eval_desc`/`embed`. This can become an opt-in array mode later for very large multi-pool feature jobs, but changing it now would alter logs and partial-failure behavior.
- `FeatureExecutionManager.submit_python_slurm_job`: one recursive Python worker; not an array workload.
- `CollectionWorkflow._submit_to_slurm`: one CPU self-submission worker; not an array workload.
- `AnalysisWorkflow._submit_to_slurm`: one CPU self-submission worker; not an array workload.
- `exploration`: delegated to external backend; no current JobManager-managed chunk submission.

## Files

Create:

- `src/dpeva/submission/array.py`
- `tests/unit/submission/test_slurm_array.py`

Modify:

- `src/dpeva/submission/templates.py`
- `src/dpeva/submission/manager.py`
- `src/dpeva/submission/__init__.py`
- `src/dpeva/workflows/labeling.py`
- `tests/unit/submission/test_job_manager.py`
- `tests/unit/workflows/test_labeling_workflow.py`
- `docs/guides/slurm.md`
- `docs/guides/developer-guide.md`

---

### Task 1: Render Slurm Array Directives in JobConfig

**Files:**
- Modify: `src/dpeva/submission/templates.py`
- Modify: `tests/unit/submission/test_job_manager.py`

- [ ] **Step 1: Add failing tests for Slurm array rendering**

Add these tests to `tests/unit/submission/test_job_manager.py`:

```python
def test_generate_script_slurm_array_with_limit(tmp_path):
    config = JobConfig(
        job_name="array_job",
        command="echo $SLURM_ARRAY_TASK_ID",
        partition="debug",
        array="0-7",
        array_task_limit=3,
        output_log="logs/%A_%a.out",
        error_log="logs/%A_%a.err",
    )
    manager = JobManager(mode="slurm")
    output_path = tmp_path / "array.slurm"

    manager.generate_script(config, str(output_path))

    content = output_path.read_text()
    assert "#SBATCH --array=0-7%3" in content
    assert "#SBATCH -o logs/%A_%a.out" in content
    assert "#SBATCH -e logs/%A_%a.err" in content
    assert "echo $SLURM_ARRAY_TASK_ID" in content


def test_generate_script_slurm_array_without_limit(tmp_path):
    config = JobConfig(
        job_name="array_job",
        command="hostname",
        array="1,3,5",
    )
    manager = JobManager(mode="slurm")
    output_path = tmp_path / "array.slurm"

    manager.generate_script(config, str(output_path))

    assert "#SBATCH --array=1,3,5" in output_path.read_text()
```

- [ ] **Step 2: Extend `JobConfig` with optional array fields**

In `src/dpeva/submission/templates.py`, add fields to `JobConfig` after `custom_headers`:

```python
    array: Optional[str] = None
    array_task_limit: Optional[int] = None
```

Add this private method to `JobConfig`:

```python
    def _array_expression(self) -> Optional[str]:
        if not self.array:
            return None
        if self.array_task_limit is None:
            return self.array
        if self.array_task_limit < 1:
            raise ValueError("array_task_limit must be >= 1")
        return f"{self.array}%{self.array_task_limit}"
```

Then in `to_dict()`, append the array directive after `error_log` and before GPU fields:

```python
        array_expr = self._array_expression()
        if array_expr:
            optional_params.append(f"#SBATCH --array={array_expr}")
```

Use this exact anchor in `src/dpeva/submission/templates.py`: insert between the existing `if self.error_log:` block and the existing `if self.gpus_per_node:` block.

```python
        if self.error_log:
            optional_params.append(f"#SBATCH -e {self.error_log}")

        array_expr = self._array_expression()
        if array_expr:
            optional_params.append(f"#SBATCH --array={array_expr}")

        if self.gpus_per_node:
            optional_params.append(f"#SBATCH --gpus-per-node={self.gpus_per_node}")
```

- [ ] **Step 3: Verify existing non-array rendering stays unchanged**

Run:

```bash
pytest tests/unit/submission/test_job_manager.py -q
```

Expected: all existing tests pass, and the generated non-array Slurm scripts do not include `#SBATCH --array`.

---

### Task 2: Add Generic Array Manifest and Worker

**Files:**
- Create: `src/dpeva/submission/array.py`
- Modify: `src/dpeva/submission/__init__.py`
- Create: `tests/unit/submission/test_slurm_array.py`

- [ ] **Step 1: Add manifest and worker tests**

Create `tests/unit/submission/test_slurm_array.py`:

```python
import pytest

from dpeva.submission.array import (
    ArrayTaskSpec,
    build_array_command,
    load_array_manifest,
    run_array_task,
    write_array_manifest,
)


def test_write_and_load_array_manifest(tmp_path):
    manifest_path = tmp_path / "tasks.json"
    tasks = [
        ArrayTaskSpec(index=0, name="bundle0", working_dir=tmp_path / "a", argv=["python", "-V"]),
        ArrayTaskSpec(index=1, name="bundle1", working_dir=tmp_path / "b", argv=["python", "-V"]),
    ]

    write_array_manifest(tasks, manifest_path)
    loaded = load_array_manifest(manifest_path)

    assert [task.index for task in loaded] == [0, 1]
    assert loaded[0].name == "bundle0"
    assert loaded[1].working_dir == tmp_path / "b"


def test_write_array_manifest_rejects_non_contiguous_indices(tmp_path):
    with pytest.raises(ValueError, match="contiguous"):
        write_array_manifest(
            [
                ArrayTaskSpec(index=0, name="a", working_dir=tmp_path, argv=["true"]),
                ArrayTaskSpec(index=2, name="b", working_dir=tmp_path, argv=["true"]),
            ],
            tmp_path / "tasks.json",
        )


def test_run_array_task_uses_slurm_array_task_id(tmp_path, monkeypatch):
    manifest_path = tmp_path / "tasks.json"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    marker = work_dir / "ran.txt"
    script = work_dir / "worker.py"
    script.write_text("from pathlib import Path\nPath('ran.txt').write_text('ok')\n")

    write_array_manifest(
        [ArrayTaskSpec(index=0, name="bundle0", working_dir=work_dir, argv=["python", str(script)])],
        manifest_path,
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    run_array_task(manifest_path)

    assert marker.read_text() == "ok"


def test_run_array_task_raises_on_missing_task(tmp_path, monkeypatch):
    manifest_path = tmp_path / "tasks.json"
    write_array_manifest(
        [ArrayTaskSpec(index=0, name="bundle0", working_dir=tmp_path, argv=["true"])],
        manifest_path,
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")

    with pytest.raises(IndexError, match="No array task"):
        run_array_task(manifest_path)


def test_build_array_command_uses_module_worker(tmp_path):
    command = build_array_command(tmp_path / "tasks.json")
    assert " -m dpeva.submission.array " in command
    assert str(tmp_path / "tasks.json") in command
```

- [ ] **Step 2: Implement `src/dpeva/submission/array.py`**

```python
"""Manifest-based Slurm array worker utilities for DP-EVA submission.

Typical caller: JobManager.submit_array() writes a manifest and submits a
single sbatch script. Slurm executes this module with
`python -m dpeva.submission.array <manifest.json>` in each array element.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union


@dataclass(frozen=True)
class ArrayTaskSpec:
    index: int
    name: str
    working_dir: Path
    argv: Sequence[str]

    def to_json_dict(self) -> dict:
        if self.index < 0:
            raise ValueError("array task index must be >= 0")
        if not self.argv:
            raise ValueError("array task argv must not be empty")
        return {
            "index": self.index,
            "name": self.name,
            "working_dir": str(Path(self.working_dir).resolve()),
            "argv": [str(item) for item in self.argv],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "ArrayTaskSpec":
        return cls(
            index=int(data["index"]),
            name=str(data["name"]),
            working_dir=Path(data["working_dir"]),
            argv=[str(item) for item in data["argv"]],
        )


def _validate_contiguous(tasks: List[ArrayTaskSpec]) -> None:
    expected = list(range(len(tasks)))
    actual = [task.index for task in tasks]
    if actual != expected:
        raise ValueError(f"array task indices must be contiguous from 0; got {actual}")


def write_array_manifest(tasks: Iterable[ArrayTaskSpec], manifest_path: Union[str, Path]) -> Path:
    manifest = Path(manifest_path)
    task_list = list(tasks)
    _validate_contiguous(task_list)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = [task.to_json_dict() for task in task_list]
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest


def load_array_manifest(manifest_path: Union[str, Path]) -> List[ArrayTaskSpec]:
    payload = json.loads(Path(manifest_path).read_text())
    tasks = [ArrayTaskSpec.from_json_dict(item) for item in payload]
    _validate_contiguous(tasks)
    return tasks


def get_array_task(tasks: List[ArrayTaskSpec], task_id: Optional[int] = None) -> ArrayTaskSpec:
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise RuntimeError("SLURM_ARRAY_TASK_ID is not set")
        task_id = int(raw)
    for task in tasks:
        if task.index == task_id:
            return task
    raise IndexError(f"No array task with index {task_id}")


def run_array_task(manifest_path: Union[str, Path], task_id: Optional[int] = None) -> None:
    tasks = load_array_manifest(manifest_path)
    task = get_array_task(tasks, task_id=task_id)
    work_dir = Path(task.working_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"Array task working directory not found: {work_dir}")
    print(f"DPEVA array task {task.index}: {task.name}")
    print(f"Working directory: {work_dir}")
    print("Command: " + " ".join(shlex.quote(arg) for arg in task.argv))
    subprocess.run(list(task.argv), cwd=str(work_dir), check=True)


def build_array_command(manifest_path: Union[str, Path]) -> str:
    manifest = shlex.quote(str(Path(manifest_path).resolve()))
    return f"{shlex.quote(sys.executable)} -m dpeva.submission.array {manifest}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("Usage: python -m dpeva.submission.array <manifest.json>", file=sys.stderr)
        return 2
    run_array_task(args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Export array helpers**

In `src/dpeva/submission/__init__.py`, export:

```python
from .array import ArrayTaskSpec
```

and add `"ArrayTaskSpec"` to `__all__`.

---

### Task 3: Add JobManager Array Submission and Slurm Status Parsing

**Files:**
- Modify: `src/dpeva/submission/manager.py`
- Create/modify: `tests/unit/submission/test_slurm_array.py`

- [ ] **Step 1: Add parser and submit-array tests**

Append tests to `tests/unit/submission/test_slurm_array.py`:

```python
from unittest.mock import patch

from dpeva.submission.manager import JobManager
from dpeva.submission.templates import JobConfig


def test_parse_sbatch_job_id():
    assert JobManager.parse_sbatch_job_id("Submitted batch job 12345") == "12345"
    assert JobManager.parse_sbatch_job_id("12345") == "12345"


def test_submit_array_writes_manifest_and_single_script(tmp_path):
    manager = JobManager(mode="slurm")
    tasks = [
        ArrayTaskSpec(index=0, name="a", working_dir=tmp_path / "a", argv=["python", "-V"]),
        ArrayTaskSpec(index=1, name="b", working_dir=tmp_path / "b", argv=["python", "-V"]),
    ]
    job_config = JobConfig(job_name="label_array", command="")
    manifest_path = tmp_path / "array" / "tasks.json"
    script_path = tmp_path / "array" / "submit.slurm"

    with patch.object(manager, "submit", return_value="Submitted batch job 24680") as mock_submit:
        job_id = manager.submit_array(
            tasks=tasks,
            job_config=job_config,
            manifest_path=manifest_path,
            script_path=script_path,
            working_dir=tmp_path,
            array_task_limit=1,
        )

    assert job_id == "24680"
    assert manifest_path.exists()
    assert script_path.exists()
    content = script_path.read_text()
    assert "#SBATCH --array=0-1%1" in content
    assert "python" in content
    assert "-m dpeva.submission.array" in content
    mock_submit.assert_called_once_with(str(script_path), working_dir=str(tmp_path))


def test_parse_sacct_records_ignores_batch_steps():
    rows = "\n".join(
        [
            "24680|COMPLETED|0:0",
            "24680_0|COMPLETED|0:0",
            "24680_0.batch|COMPLETED|0:0",
            "24680_1|FAILED|1:0",
        ]
    )

    records = JobManager.parse_sacct_records(rows)

    assert [record.job_id for record in records] == ["24680", "24680_0", "24680_1"]
    assert records[2].array_task_id == 1
    assert records[2].state == "FAILED"


def test_summarize_array_records_reports_failed_tasks():
    records = JobManager.parse_sacct_records(
        "24680_0|COMPLETED|0:0\n24680_1|FAILED|1:0\n"
    )

    summary = JobManager.summarize_array_records("24680", records, expected_count=2)

    assert summary.completed == 1
    assert summary.failed == 1
    assert summary.failed_task_ids == [1]
    assert not summary.ok


@patch("dpeva.submission.manager.subprocess.run")
def test_query_active_slurm_ids_preserves_array_element_suffixes(mock_run):
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "24680_0\n24680_1\n"
    mock_run.return_value.stderr = ""
    manager = JobManager(mode="slurm")

    active_ids = manager.query_active_slurm_ids(["Submitted batch job 24680"])

    assert active_ids == ["24680_0", "24680_1"]
```

- [ ] **Step 2: Add lightweight status dataclasses**

At the top of `src/dpeva/submission/manager.py`:

```python
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional

from .array import ArrayTaskSpec, build_array_command, write_array_manifest
```

Add below imports:

```python
SUCCESS_STATES = {"COMPLETED"}


@dataclass(frozen=True)
class SlurmJobRecord:
    job_id: str
    state: str
    exit_code: str
    array_task_id: Optional[int] = None


@dataclass(frozen=True)
class SlurmArraySummary:
    array_job_id: str
    expected_count: int
    completed: int
    failed: int
    missing_task_ids: List[int]
    failed_task_ids: List[int]

    @property
    def ok(self) -> bool:
        return self.failed == 0 and not self.missing_task_ids and self.completed == self.expected_count
```

- [ ] **Step 3: Implement array methods on `JobManager`**

Add methods to `JobManager`:

```python
    @staticmethod
    def parse_sbatch_job_id(output: str) -> str:
        import re

        match = re.search(r"\b(\d+)\b", output)
        if not match:
            raise ValueError(f"Could not parse Slurm job id from output: {output!r}")
        return match.group(1)

    def submit_array(
        self,
        tasks: Iterable[ArrayTaskSpec],
        job_config: JobConfig,
        manifest_path: str,
        script_path: str,
        working_dir: str = ".",
        array_task_limit: Optional[int] = None,
    ) -> str:
        task_list = list(tasks)
        if not task_list:
            raise ValueError("Cannot submit an empty Slurm array")
        manifest = write_array_manifest(task_list, manifest_path)
        job_config.command = build_array_command(manifest)
        job_config.array = f"0-{len(task_list) - 1}"
        if array_task_limit is not None:
            job_config.array_task_limit = array_task_limit
        if job_config.output_log is None:
            job_config.output_log = "slurm-%A_%a.out"
        if job_config.error_log is None:
            job_config.error_log = "slurm-%A_%a.err"
        self.generate_script(job_config, script_path)
        return self.parse_sbatch_job_id(self.submit(script_path, working_dir=str(working_dir)))

    @staticmethod
    def parse_sacct_records(stdout: str) -> List[SlurmJobRecord]:
        records: List[SlurmJobRecord] = []
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("|")
            if len(fields) < 3:
                continue
            job_id, state, exit_code = fields[:3]
            if "." in job_id:
                continue
            array_task_id = None
            if "_" in job_id:
                suffix = job_id.rsplit("_", 1)[1]
                if suffix.isdigit():
                    array_task_id = int(suffix)
            records.append(SlurmJobRecord(job_id=job_id, state=state, exit_code=exit_code, array_task_id=array_task_id))
        return records

    @staticmethod
    def summarize_array_records(array_job_id: str, records: List[SlurmJobRecord], expected_count: int) -> SlurmArraySummary:
        task_records = [record for record in records if record.array_task_id is not None]
        seen = {record.array_task_id for record in task_records if record.array_task_id is not None}
        missing = [idx for idx in range(expected_count) if idx not in seen]
        failed_task_ids = [
            int(record.array_task_id)
            for record in task_records
            if record.array_task_id is not None
            and (record.state not in SUCCESS_STATES or record.exit_code != "0:0")
        ]
        completed = sum(
            1
            for record in task_records
            if record.state in SUCCESS_STATES and record.exit_code == "0:0"
        )
        return SlurmArraySummary(
            array_job_id=array_job_id,
            expected_count=expected_count,
            completed=completed,
            failed=len(failed_task_ids),
            missing_task_ids=missing,
            failed_task_ids=failed_task_ids,
        )
```

- [ ] **Step 4: Add active-job and accounting query helpers**

Add:

```python
    def query_active_slurm_ids(self, job_ids: Iterable[str]) -> List[str]:
        """Return active Slurm job records for base job IDs.

        Input IDs are normalized to base Slurm job IDs for querying. Returned
        IDs preserve Slurm's output, including array element suffixes such as
        "24680_0".
        """
        clean_ids = [self.parse_sbatch_job_id(job_id) for job_id in job_ids]
        if not clean_ids:
            return []
        result = subprocess.run(
            ["squeue", "--array", "--job", ",".join(clean_ids), "--noheader", "--format=%i"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "squeue failed")
        return result.stdout.strip().split()

    def query_sacct_records(self, job_id: str) -> List[SlurmJobRecord]:
        clean_id = self.parse_sbatch_job_id(job_id)
        result = subprocess.run(
            ["sacct", "-j", clean_id, "-P", "-n", "-o", "JobID,State,ExitCode"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sacct failed")
        return self.parse_sacct_records(result.stdout)
```

Do not add a blocking wait loop here yet. `LabelingWorkflow` will call these helpers so the wait policy remains workflow-controlled.

---

### Task 4: Route Labeling Slurm Execution Through One Array per Attempt

**Files:**
- Modify: `src/dpeva/workflows/labeling.py`
- Modify: `tests/unit/workflows/test_labeling_workflow.py`

- [ ] **Step 1: Add labeling array tests**

Append tests:

```python
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_submit_job_dirs_slurm_uses_single_array_by_default(self, MockManager, tmp_path):
        config = LabelingConfig(
            work_dir=str(tmp_path),
            input_data_path=str(tmp_path / "data"),
            submission={
                "backend": "slurm",
                "slurm_config": {"partition": "4V100", "qos": "rush-gpu", "array_max_parallel": 2},
            },
            dft_params={},
            attempt_params=[],
            pp_dir="/tmp/pp",
            orb_dir="/tmp/orb",
        )
        job_a = tmp_path / "inputs" / "N_50_0"
        job_b = tmp_path / "inputs" / "N_50_1"
        job_a.mkdir(parents=True)
        job_b.mkdir(parents=True)
        manager = MockManager.return_value
        manager.generate_runner_script.side_effect = ["print('a')", "print('b')"]
        wf = LabelingWorkflow(config)
        wf.job_manager.submit_array = MagicMock(return_value="12345")

        job_ids = wf._submit_job_dirs([job_a, job_b], attempt=0)

        assert job_ids == ["12345"]
        wf.job_manager.submit_array.assert_called_once()
        tasks = wf.job_manager.submit_array.call_args.kwargs["tasks"]
        assert [task.index for task in tasks] == [0, 1]
        assert [task.working_dir for task in tasks] == [job_a, job_b]
        assert (job_a / "run_batch.py").exists()
        assert (job_b / "run_batch.py").exists()


    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_submit_job_dirs_slurm_can_disable_array(self, MockManager, tmp_path):
        config = LabelingConfig(
            work_dir=str(tmp_path),
            input_data_path=str(tmp_path / "data"),
            submission={
                "backend": "slurm",
                "slurm_config": {"use_job_array": False},
            },
            dft_params={},
            attempt_params=[],
            pp_dir="/tmp/pp",
            orb_dir="/tmp/orb",
        )
        job_a = tmp_path / "inputs" / "N_50_0"
        job_b = tmp_path / "inputs" / "N_50_1"
        job_a.mkdir(parents=True)
        job_b.mkdir(parents=True)
        manager = MockManager.return_value
        manager.generate_runner_script.side_effect = ["print('a')", "print('b')"]
        wf = LabelingWorkflow(config)
        wf.job_manager.submit_python_script = MagicMock(side_effect=["11", "12"])

        job_ids = wf._submit_job_dirs([job_a, job_b], attempt=0)

        assert job_ids == ["11", "12"]
        assert wf.job_manager.submit_python_script.call_count == 2


    @patch("dpeva.workflows.labeling.logger.error")
    @patch("dpeva.workflows.labeling.LabelingManager")
    def test_submit_job_dirs_slurm_array_failure_falls_back_to_individual(self, MockManager, mock_error, tmp_path):
        config = LabelingConfig(
            work_dir=str(tmp_path),
            input_data_path=str(tmp_path / "data"),
            submission={
                "backend": "slurm",
                "slurm_config": {"array_max_parallel": 2},
            },
            dft_params={},
            attempt_params=[],
            pp_dir="/tmp/pp",
            orb_dir="/tmp/orb",
        )
        job_a = tmp_path / "inputs" / "N_50_0"
        job_b = tmp_path / "inputs" / "N_50_1"
        job_a.mkdir(parents=True)
        job_b.mkdir(parents=True)
        manager = MockManager.return_value
        manager.generate_runner_script.side_effect = [
            "print('a array')",
            "print('b array')",
            "print('a legacy')",
            "print('b legacy')",
        ]
        wf = LabelingWorkflow(config)
        wf.job_manager.submit_array = MagicMock(side_effect=RuntimeError("array submit failed"))
        wf.job_manager.submit_python_script = MagicMock(side_effect=["11", "12"])

        job_ids = wf._submit_job_dirs([job_a, job_b], attempt=0)

        assert job_ids == ["11", "12"]
        wf.job_manager.submit_array.assert_called_once()
        assert wf.job_manager.submit_python_script.call_count == 2
        assert any("Falling back to individual" in call.args[0] for call in mock_error.call_args_list)
```

- [ ] **Step 2: Import array helpers in labeling workflow**

At the top of `src/dpeva/workflows/labeling.py`, add:

```python
from dpeva.submission.array import ArrayTaskSpec
```

- [ ] **Step 3: Split legacy single-job submission into a private helper**

Replace the body of `_submit_job_dirs()` with a dispatcher:

```python
    def _submit_job_dirs(self, active_job_dirs: List[Path], attempt: int) -> List[str]:
        slurm_conf = self.config.submission.slurm_config
        if self.config.submission.backend == "slurm" and slurm_conf.get("use_job_array", True):
            return self._submit_job_dirs_as_array(active_job_dirs, attempt)
        return self._submit_job_dirs_individually(active_job_dirs, attempt)
```

Move the current loop into `_submit_job_dirs_individually()` unchanged except for the method name:

```python
    def _submit_job_dirs_individually(self, active_job_dirs: List[Path], attempt: int) -> List[str]:
        job_ids = []
        for job_dir in active_job_dirs:
            runner_content = self.manager.generate_runner_script(job_dir)
            runner_name = "run_batch.py"
            slurm_conf = self.config.submission.slurm_config
            job_config = JobConfig(
                command="",
                job_name=f"fp_{job_dir.name}_att{attempt}",
                partition=slurm_conf.get("partition"),
                qos=slurm_conf.get("qos"),
                nodes=slurm_conf.get("nodes", 1),
                ntasks=slurm_conf.get("ntasks", 1),
                gpus_per_node=slurm_conf.get("gpus_per_node"),
                cpus_per_task=slurm_conf.get("cpus_per_task"),
                walltime=slurm_conf.get("walltime", "24:00:00"),
                env_setup=self.config.submission.env_setup,
            )
            try:
                job_id = self.job_manager.submit_python_script(
                    runner_content,
                    runner_name,
                    job_config,
                    working_dir=str(job_dir),
                )
                job_ids.append(job_id)
            except Exception as e:
                logger.error(f"Failed to submit job for {job_dir}: {e}")
        return job_ids
```

- [ ] **Step 4: Add array submission helper**

Add:

```python
    def _submit_job_dirs_as_array(self, active_job_dirs: List[Path], attempt: int) -> List[str]:
        if not active_job_dirs:
            return []
        slurm_conf = self.config.submission.slurm_config
        work_dir = Path(self.config.work_dir)
        array_dir = work_dir / "slurm_arrays"
        logs_dir = array_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        tasks: List[ArrayTaskSpec] = []
        for idx, job_dir in enumerate(active_job_dirs):
            runner_content = self.manager.generate_runner_script(job_dir)
            runner_path = job_dir / "run_batch.py"
            runner_path.write_text(runner_content)
            tasks.append(
                ArrayTaskSpec(
                    index=idx,
                    name=job_dir.name,
                    working_dir=job_dir,
                    argv=[sys.executable, "-u", str(runner_path.resolve())],
                )
            )

        job_config = JobConfig(
            command="",
            job_name=slurm_conf.get("job_name", f"fp_label_att{attempt}"),
            partition=slurm_conf.get("partition"),
            qos=slurm_conf.get("qos"),
            nodes=slurm_conf.get("nodes", 1),
            ntasks=slurm_conf.get("ntasks", 1),
            gpus_per_node=slurm_conf.get("gpus_per_node"),
            cpus_per_task=slurm_conf.get("cpus_per_task"),
            walltime=slurm_conf.get("walltime", "24:00:00"),
            output_log=str(logs_dir / f"labeling_att{attempt}-%A_%a.out"),
            error_log=str(logs_dir / f"labeling_att{attempt}-%A_%a.err"),
            env_setup=self.config.submission.env_setup,
        )

        array_limit = slurm_conf.get("array_max_parallel", slurm_conf.get("max_parallel"))
        manifest_path = array_dir / f"labeling_attempt_{attempt}.manifest.json"
        script_path = array_dir / f"labeling_attempt_{attempt}.slurm"

        try:
            job_id = self.job_manager.submit_array(
                tasks=tasks,
                job_config=job_config,
                manifest_path=str(manifest_path),
                script_path=str(script_path),
                working_dir=str(work_dir),
                array_task_limit=array_limit,
            )
            logger.info(f"Submitted labeling Slurm array {job_id} with {len(tasks)} tasks.")
            return [job_id]
        except Exception as e:
            logger.error(f"Failed to submit labeling Slurm array: {e}. Falling back to individual submissions.")
            return self._submit_job_dirs_individually(active_job_dirs, attempt)
```

Do not pass extra `slurm_config` keys via `**kwargs` here, because `JobConfig` does not accept arbitrary keys. The current labeling code already enumerates allowed Slurm fields explicitly; preserve that pattern.

---

### Task 5: Replace Labeling's Ad Hoc Slurm Monitoring with Submission Helpers

**Files:**
- Modify: `src/dpeva/workflows/labeling.py`
- Modify: `tests/unit/workflows/test_labeling_workflow.py`

- [ ] **Step 1: Replace the old monitoring test and add a new query test**

Delete the entire existing test block in `tests/unit/workflows/test_labeling_workflow.py` whose decorators start with:

```python
    @patch("dpeva.workflows.labeling.time.sleep")
    @patch("dpeva.workflows.labeling.subprocess.run", side_effect=RuntimeError("squeue down"))
    @patch("dpeva.workflows.labeling.logger.error")
```

Replace that deleted block with the first test below, preserving the same test name. Then add the second test as a new test. Do not leave the old `subprocess.run` patch in the file.

```python
    @patch("dpeva.workflows.labeling.time.sleep")
    @patch("dpeva.workflows.labeling.logger.error")
    def test_monitor_slurm_jobs_logs_query_failure_and_continues(self, mock_error, mock_sleep, config):
        wf = LabelingWorkflow(config)
        wf.job_manager.query_active_slurm_ids = MagicMock(side_effect=RuntimeError("squeue down"))
        mock_sleep.side_effect = RuntimeError("stop-loop")

        with pytest.raises(RuntimeError, match="stop-loop"):
            wf._monitor_slurm_jobs(["job-123"], interval=1)

        mock_error.assert_called_once()


    @patch("dpeva.workflows.labeling.time.sleep")
    @patch("dpeva.workflows.labeling.logger.info")
    def test_monitor_slurm_jobs_uses_job_manager_query(self, mock_info, mock_sleep, config):
        wf = LabelingWorkflow(config)
        wf.job_manager.query_active_slurm_ids = MagicMock(side_effect=[["123_0"], []])

        wf._monitor_slurm_jobs(["123"], interval=1)

        assert wf.job_manager.query_active_slurm_ids.call_count == 2
        assert mock_sleep.call_count == 1
        assert any("All jobs finished" in call.args[0] for call in mock_info.call_args_list)
```

- [ ] **Step 2: Simplify `_monitor_slurm_jobs()`**

Replace `_monitor_slurm_jobs()` with:

```python
    def _monitor_slurm_jobs(self, job_ids: List[str], interval: int = 60):
        """Monitor Slurm jobs and wait until they leave the queue."""
        if not job_ids:
            return

        logger.info(f"Monitoring {len(job_ids)} Slurm job groups...")
        wait_count = 0
        while True:
            try:
                active_ids = set(self.job_manager.query_active_slurm_ids(job_ids))
                if not active_ids:
                    logger.info("All jobs finished.")
                    break
                if wait_count % 10 == 0:
                    logger.info(
                        f"{len(active_ids)} Slurm job records still active... "
                        f"(waited {wait_count * interval // 60} mins)"
                    )
                wait_count += 1
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Failed to query Slurm queue: {e}")
                time.sleep(interval)
```

Keep this queue-based monitor behavior compatible with existing tests. Do not fail the workflow based only on accounting yet; convergence is still checked by `self.manager.process_results(active_job_dirs)` after jobs finish.

- [ ] **Step 3: Remove unused imports**

After replacing the monitor, remove `subprocess` and `re` imports from `src/dpeva/workflows/labeling.py` if they are no longer used.

---

### Task 6: Document Array Configuration and Workflow Scope

**Files:**
- Modify: `docs/guides/slurm.md`
- Modify: `docs/guides/developer-guide.md`

- [ ] **Step 1: Update Slurm guide**

In `docs/guides/slurm.md`, update both the front matter `last-updated` field and the visible `- Last-Updated:` line to `2026-07-03`. Then add a section after `3.2 常用扩展字段`:

````markdown
### 3.3 Slurm Array 字段

Labeling 的 Slurm 执行阶段默认使用 job array 提交 packed bundles。以下 `slurm_config` 字段仅由 labeling workflow 消费，不是 train/infer/feature 的全局 array 开关：

- `use_job_array`：Labeling 专用开关，默认 `true`。设为 `false` 时回退到旧的逐 bundle 提交。
- `array_max_parallel`：同一 array 中最多同时运行的元素数，渲染为 `#SBATCH --array=0-N%M`。
- `max_parallel`：`array_max_parallel` 的兼容别名。

示例：

```json
{
  "submission": {
    "backend": "slurm",
    "env_setup": ["source scripts/env/dpeva-dpa4.env"],
    "slurm_config": {
      "partition": "4V100",
      "qos": "rush-gpu",
      "nodes": 1,
      "ntasks": 4,
      "gpus_per_node": 4,
      "walltime": "24:00:00",
      "array_max_parallel": 2
    }
  }
}
```

在 SAI 上，`array_max_parallel` 应结合 QoS 的并发 GPU 限制设置。例如每个 array element 使用 4 卡且 QoS 同时可用 16 卡时，可先设为 `4` 或更保守的 `2`。
````

- [ ] **Step 2: Update log naming section**

In `docs/guides/slurm.md`, add labeling array logs:

```markdown
- Labeling：`<work_dir>/slurm_arrays/logs/labeling_att<attempt>-%A_%a.out`
```

- [ ] **Step 3: Update developer guide Slurm architecture**

In `docs/guides/developer-guide.md`, update both the front matter `last-updated` field and the visible `- Last-Updated:` line to `2026-07-03`. Then add these bullets to the Slurm backend section:

```markdown
*   **Job Array Support**: Labeling packed bundles use one Slurm job array per execution attempt. The submission layer provides generic manifest-based array execution (`ArrayTaskSpec`) so future workflows can opt in without duplicating `sbatch --array` handling.
*   **Array Scope Policy**: Use arrays for many homogeneous chunks with identical resources. Keep single-job submission for self-submitting workflows and small user-facing model-replica jobs unless there is an explicit opt-in design.
```

---

### Task 7: Verification

**Files:**
- No code edits unless tests expose defects.

- [ ] **Step 1: Run focused unit tests**

```bash
pytest tests/unit/submission/test_job_manager.py tests/unit/submission/test_slurm_array.py tests/unit/workflows/test_labeling_workflow.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run workflow tests impacted by submission changes**

```bash
pytest tests/unit/workflows/test_train_workflow_init.py tests/unit/training/test_training_managers.py tests/unit/inference/test_inference_execution_manager.py tests/unit/workflows/test_feature_workflow_submission.py tests/unit/feature/test_execution_manager.py tests/unit/workflows/test_collection_workflow_submission.py tests/unit/workflows/test_analysis_workflow.py -q
```

Expected: existing non-labeling Slurm behavior remains unchanged.

- [ ] **Step 3: Run full unit suite if time permits**

```bash
pytest tests/unit -q
```

Expected: all unit tests pass. If unrelated tests fail because the current worktree already has dirty changes, record failures with file/test names and do not revert user changes.

- [ ] **Step 4: Unit dry-run inspection**

Do not submit real jobs by default. Use focused unit tests to verify array script content and labeling routing:

```bash
pytest tests/unit/submission/test_slurm_array.py::test_submit_array_writes_manifest_and_single_script tests/unit/workflows/test_labeling_workflow.py::TestLabelingWorkflow::test_submit_job_dirs_slurm_uses_single_array_by_default -q
```

- [ ] **Step 5: Optional real SAI execute-stage validation**

Run this only when a tiny disposable SAI labeling config is available and real queue submission is acceptable. First prepare packed inputs, then execute only the submission stage:

```bash
dpeva label <config.json> --stage prepare
dpeva label <config.json> --stage execute
grep -E "Submitted labeling Slurm array|Monitoring" <work_dir>/labeling_execute.log
sed -n '1,120p' <work_dir>/slurm_arrays/labeling_attempt_0.slurm
```

Expected behavior:

- `labeling_execute.log` shows the array job was submitted and monitoring entered the queue polling path.
- Contains one `#SBATCH --array=0-N%M` line.
- Contains `%A_%a` in output and error logs.
- Does not add `--mem` or new CPU fields unless explicitly configured.
- Uses the configured `partition`, `qos`, `nodes`, `ntasks`, `gpus_per_node`, and `walltime`.

- [ ] **Step 6: Record integration-test scope in handoff**

If this change is delivered through a PR, include this text in the PR description or implementation handoff:

```markdown
Slurm array support is covered by unit tests with mocked Slurm commands. This PR does not update `tests/integration/test_slurm_multidatapool_e2e.py`; real SAI end-to-end validation is tracked as follow-up. Optional manual validation command: `dpeva label <config.json> --stage execute` after `--stage prepare`, then inspect `<work_dir>/slurm_arrays/labeling_attempt_0.slurm` and `<work_dir>/labeling_execute.log`.
```

---

## Implementation Notes

- Keep backward compatibility: `JobConfig.array` defaults to `None`; non-array workflows render exactly as before.
- Keep labeling rollback path: `submission.slurm_config.use_job_array=false` must preserve the current per-bundle `submit_python_script()` behavior.
- If `submit_array()` fails before Slurm accepts the array job, fall back to `_submit_job_dirs_individually()` so labeling preserves the old partial-submission behavior.
- Keep array tasks contiguous from `0` so the generated `--array=0-N` expression is simple and robust.
- Use manifest `argv` lists and `subprocess.run(..., shell=False)` to avoid adding shell quoting risk to generated per-task commands.
- Keep queue polling non-fatal on transient Slurm query errors, matching existing labeling behavior.
- Do not make `sacct` accounting success required for labeling convergence in this first refactor. DP-EVA's labeling convergence is determined by ABACUS output parsing; Slurm accounting is useful for diagnostics but not the source of truth.

## Self-Review

- The plan targets the staff-reported issue directly: labeling changes from many `sbatch` calls to one `sbatch --array` per attempt.
- It avoids broad behavioral churn in train/infer/feature, where arrays are possible but not necessary for the immediate problem.
- It keeps SAI constraints intact by not introducing memory defaults and by supporting `%` concurrency limits.
- It provides a rollback switch and focused tests for both new array mode and legacy submission mode.
- It centralizes Slurm array rendering and status parsing in `submission`, so future workflow opt-in does not duplicate Slurm command details.
