---
title: Run Contract and Strict Config Implementation Plan
status: proposed
audience: Developers / AI Agents
last-updated: 2026-09-04
owner: Project Maintainer
---

# Run Contract and Strict Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce the smallest versioned run manifest, explicit environment doctor, strict configuration migration, and immutable run identity needed to govern feature and inference first.

**Spec:** `docs/superpowers/specs/2026-09-04-project-governance-and-deepmd-3-2-design.html` (`#architecture` §8.1–8.5, `#contracts` §9.1 and §9.5, `#requirements` R3/R7/R11/R12, `#rollout` §15.2 and §15.9)

**Architecture:** Build a focused `dpeva.run` package on Plan A's pure state machine. The CLI performs explicit legacy migration before strict Pydantic validation, then feature and infer create an immutable `.dpeva/runs/{run-id}/run.json`; other workflows remain unchanged until the pilot proves useful. Doctor uses injected subprocess execution and never runs during import.

**Tech Stack:** Python 3.10+, Pydantic v2, pathlib, JSON, subprocess, pytest, existing CLI/workflows.

## Global Constraints

- Requires Plan A checkpoint `GO` and its `dpeva.run.status` API.
- Pilot only `feature` and `infer`; do not wire train/collect/label/analysis/clean/explore in this plan.
- Schema `1.0` contains only fields consumed by R3/R7/R11/R12; additions require a real consumer and test (`#rollout` §15.9).
- Read old config, write normalized config; never mutate the user's source JSON (`#contracts` §9.5).
- Existing run evidence is immutable by default; `--resume` and `--force` are mutually exclusive and create explicit events.

---

### Task 1: Remove import-time DeepMD probing and add the doctor probe model

**Files:**
- Modify: `src/dpeva/__init__.py`
- Modify: `src/dpeva/constants.py`
- Replace: `src/dpeva/utils/env_check.py`
- Create: `src/dpeva/run/doctor.py`
- Modify: `tests/unit/utils/test_env_check.py`
- Create: `tests/unit/run/test_doctor.py`

**Test strategy:**
- Behavior boundary: importing `dpeva` launches no subprocess; doctor reports available, missing, unparsable, and incompatible DeepMD states as stable JSON data.
- Existing suite to extend: `tests/unit/utils/test_env_check.py`.
- New test file justification: `test_doctor.py` owns the new structured report boundary.
- Temporary probes: none.

**Interfaces:**
- Consumes: `MIN_DEEPMD_VERSION = "3.2.0"`, `MAX_DEEPMD_VERSION = "3.3"`, and an injected command runner compatible with `subprocess.run`.
- Produces: `DoctorCheck`, `DoctorReport`, `probe_deepmd(run=subprocess.run) -> DoctorCheck`, and `build_doctor_report() -> DoctorReport`.

- [ ] **Step 1: Write failing import and doctor tests**

```python
import importlib
import json
import subprocess

import dpeva
from dpeva.run.doctor import build_doctor_report, probe_deepmd


def test_import_dpeva_does_not_probe_external_commands(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("subprocess called")))
    importlib.reload(dpeva)


def test_probe_deepmd_missing_is_structured() -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("dp")
    check = probe_deepmd(run=missing)
    assert check.model_dump() == {
        "name": "deepmd",
        "status": "missing",
        "version": None,
        "detail": "dp executable not found",
    }


def test_doctor_report_is_json_serializable() -> None:
    report = build_doctor_report(checks=[])
    assert json.loads(report.model_dump_json())["schema_version"] == "1.0"
```

Run: `pytest tests/unit/run/test_doctor.py tests/unit/utils/test_env_check.py -q`

Expected: FAIL because the structured doctor API is absent and import still probes `dp`.

- [ ] **Step 2: Define the structured probe**

```python
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence

from packaging.version import Version
from pydantic import BaseModel

from dpeva.constants import MAX_DEEPMD_VERSION, MIN_DEEPMD_VERSION


class DoctorCheck(BaseModel):
    name: str
    status: str
    version: str | None = None
    detail: str


class DoctorReport(BaseModel):
    schema_version: str = "1.0"
    status: str
    checks: list[DoctorCheck]


def probe_deepmd(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> DoctorCheck:
    try:
        result = run(["dp", "--version"], check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return DoctorCheck(name="deepmd", status="missing", detail="dp executable not found")
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    match = re.search(r"v?(\d+(?:\.\d+)+(?:[A-Za-z0-9_.!+\-]*))", output)
    if result.returncode != 0:
        return DoctorCheck(name="deepmd", status="error", detail=output or f"dp exited {result.returncode}")
    if match is None:
        return DoctorCheck(name="deepmd", status="unknown", detail=f"unparsed version: {output}")
    raw = match.group(1)
    parsed = Version(raw)
    compatible = parsed.is_devrelease or Version(MIN_DEEPMD_VERSION) <= parsed < Version(MAX_DEEPMD_VERSION)
    return DoctorCheck(
        name="deepmd",
        status="ok" if compatible else "incompatible",
        version=raw,
        detail=f"required >= {MIN_DEEPMD_VERSION}, < {MAX_DEEPMD_VERSION}",
    )


def build_doctor_report(checks: Sequence[DoctorCheck] | None = None) -> DoctorReport:
    observed = list(checks) if checks is not None else [probe_deepmd()]
    status = "ok" if all(item.status == "ok" for item in observed) else "failed"
    return DoctorReport(status=status, checks=observed)
```

Set `MIN_DEEPMD_VERSION = "3.2.0"` and add `MAX_DEEPMD_VERSION = "3.3"`. Reduce `src/dpeva/__init__.py` to package metadata and `__version__`; remove all environment-check imports and exception handling. Keep `check_deepmd_version()` only as a deprecated wrapper that calls `probe_deepmd()` when explicitly invoked.

- [ ] **Step 3: Run doctor unit tests**

Run: `pytest tests/unit/run/test_doctor.py tests/unit/utils/test_env_check.py -q`

Expected: all tests pass with no import-time warning.

- [ ] **Step 4: Commit explicit probing**

```bash
git add src/dpeva/__init__.py src/dpeva/constants.py src/dpeva/utils/env_check.py src/dpeva/run/doctor.py tests/unit/utils/test_env_check.py tests/unit/run/test_doctor.py
git commit -m "feat: make environment checks explicit"
```

### Task 2: Add `dpeva doctor --json`

**Files:**
- Modify: `src/dpeva/cli.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `docs/guides/cli.md`

**Test strategy:**
- Behavior boundary: human output is default; `--json` emits only valid JSON and exits `0` for `ok`, `1` otherwise.
- Existing suite to extend: `tests/unit/test_cli.py`.
- New test file justification: none.
- Temporary probes: none.

**Interfaces:**
- Consumes: `build_doctor_report() -> DoctorReport` from Task 1.
- Produces: `handle_doctor(args) -> None` and CLI command `dpeva doctor [--json]`.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_doctor_json_exit_zero(monkeypatch, capsys) -> None:
    from dpeva.run.doctor import DoctorReport
    monkeypatch.setattr("dpeva.run.doctor.build_doctor_report", lambda: DoctorReport(status="ok", checks=[]))
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "doctor", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_doctor_json_exit_one(monkeypatch) -> None:
    from dpeva.run.doctor import DoctorCheck, DoctorReport
    report = DoctorReport(status="failed", checks=[DoctorCheck(name="deepmd", status="missing", detail="dp executable not found")])
    monkeypatch.setattr("dpeva.run.doctor.build_doctor_report", lambda: report)
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "doctor", "--json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
```

Run: `pytest tests/unit/test_cli.py -q`

Expected: FAIL because `doctor` is not registered.

- [ ] **Step 2: Implement the handler and parser**

```python
def handle_doctor(args) -> None:
    from dpeva.run.doctor import build_doctor_report
    report = build_doctor_report()
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        for check in report.checks:
            print(f"{check.name}: {check.status} - {check.detail}")
    if report.status != "ok":
        raise SystemExit(1)
```

Register:

```python
p_doctor = subparsers.add_parser("doctor", help="Report runtime capabilities")
p_doctor.add_argument("--json", action="store_true", help="Emit a stable JSON report")
p_doctor.set_defaults(func=handle_doctor)
```

- [ ] **Step 3: Verify CLI behavior**

Run: `pytest tests/unit/test_cli.py -q && conda run -n dpeva-dpa4 python -m dpeva.cli --no-banner doctor --json`

Expected: tests pass; the command emits schema `1.0` JSON. Its exit code reflects the current environment and must match the top-level `status`.

- [ ] **Step 4: Commit doctor CLI**

```bash
git add src/dpeva/cli.py tests/unit/test_cli.py docs/guides/cli.md
git commit -m "feat: add structured doctor command"
```

### Task 3: Introduce explicit legacy migration followed by strict validation

**Files:**
- Create: `src/dpeva/config_migration.py`
- Create: `tests/unit/test_config_migration.py`
- Modify: `src/dpeva/config.py`
- Modify: `src/dpeva/cli.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `docs/guides/configuration.md`
- Modify: `docs/reference/validation.md`

**Test strategy:**
- Behavior boundary: misspelled/unknown fields fail before submission; supported flat submission keys migrate once, are removed from normalized data, and generate warnings.
- Existing suite to extend: CLI and configuration tests.
- New test file justification: migration is a pure independently versioned boundary, distinct from Pydantic model validation.
- Temporary probes: none.

**Interfaces:**
- Consumes: raw JSON mapping and config schema version `1.0`.
- Produces: `MigrationWarning`, `MigrationResult`, and `migrate_legacy_config(raw: dict[str, Any]) -> MigrationResult`.

- [ ] **Step 1: Write migration and strictness tests**

```python
from pydantic import ValidationError

from dpeva.config import InferenceConfig
from dpeva.config_migration import migrate_legacy_config


def test_flat_submission_keys_migrate_and_are_removed() -> None:
    result = migrate_legacy_config({"backend": "slurm", "env_setup": "module load x", "data_path": "data"})
    assert result.normalized["submission"] == {"backend": "slurm", "env_setup": "module load x"}
    assert "backend" not in result.normalized
    assert {item.field for item in result.warnings} == {"backend", "env_setup"}


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="results_prefx"):
        InferenceConfig.model_validate({"data_path": "data", "results_prefx": "wrong"})
```

Run: `pytest tests/unit/test_config_migration.py -q`

Expected: FAIL because migration does not exist and extra fields are ignored.

- [ ] **Step 2: Implement a non-mutating migration result**

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MigrationWarning:
    field: str
    replacement: str
    removal_version: str = "1.0"


@dataclass(frozen=True)
class MigrationResult:
    normalized: dict[str, Any]
    warnings: tuple[MigrationWarning, ...]


_FLAT_SUBMISSION_KEYS = ("backend", "slurm_config", "env_setup", "slurm_array", "slurm_array_task_limit")


def migrate_legacy_config(raw: dict[str, Any]) -> MigrationResult:
    normalized = deepcopy(raw)
    warnings: list[MigrationWarning] = []
    submission = deepcopy(normalized.get("submission", {}))
    for key in _FLAT_SUBMISSION_KEYS:
        if key not in normalized:
            continue
        if key in submission and submission[key] != normalized[key]:
            raise ValueError(f"conflicting legacy field '{key}' and submission.{key}")
        submission[key] = normalized.pop(key)
        warnings.append(MigrationWarning(field=key, replacement=f"submission.{key}"))
    if submission:
        normalized["submission"] = submission
    return MigrationResult(normalized=normalized, warnings=tuple(warnings))
```

- [ ] **Step 3: Make every public config model strict**

Define one base and inherit it from `SubmissionConfig`, `BaseWorkflowConfig`, `ExplorationConfig`, `AnalysisConfig`, `LabelingTaskSelectorConfig`, and `LabelingTaskClassConfig`:

```python
class StrictConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )
```

Remove `extract_flat_submission_config` validators from `BaseWorkflowConfig` and `AnalysisConfig`; migration now owns that behavior.

- [ ] **Step 4: Route all CLI config loads through migration**

Change `load_and_resolve_config()` to return the migration result alongside the original path-resolved mapping:

```python
def load_and_resolve_config(config_path: str) -> MigrationResult:
    raw = load_json_config(config_path)
    migrated = migrate_legacy_config(raw)
    resolved = resolve_config_paths(migrated.normalized, config_path)
    return MigrationResult(normalized=resolved, warnings=migrated.warnings)
```

Each handler passes `.normalized` to its config/workflow and logs each warning in the concrete form `legacy config field backend; use submission.backend; removal target 1.0` (substituting the warning's actual field and replacement). Retain `load_json_config()` as the extracted JSON error-handling logic from the current function.

- [ ] **Step 5: Run config and CLI tests**

Run: `pytest tests/unit/test_config_migration.py tests/unit/test_cli.py tests/unit/utils/test_config_paths.py tests/unit/test_llpr_config.py -q`

Expected: all tests pass; no supported recipe fails validation.

- [ ] **Step 6: Validate every versioned recipe**

Run: `python -m pytest tests/unit -q`

Expected: all unit tests pass. If a repository recipe contains an undocumented extra field, either add the real typed field with a consuming code path or remove the stale recipe field; do not add a catch-all.

- [ ] **Step 7: Commit strict migration**

```bash
git add src/dpeva/config_migration.py src/dpeva/config.py src/dpeva/cli.py tests/unit/test_config_migration.py tests/unit/test_cli.py docs/guides/configuration.md docs/reference/validation.md
git commit -m "feat: migrate legacy config before strict validation"
```

### Task 4: Define the minimal run-manifest schema and atomic recorder

**Files:**
- Create: `src/dpeva/run/models.py`
- Create: `src/dpeva/run/recorder.py`
- Create: `tests/unit/run/test_recorder.py`
- Modify: `src/dpeva/run/__init__.py`

**Test strategy:**
- Behavior boundary: successful and failed attempts produce schema-valid JSON; transitions are validated; writes are atomic.
- Existing suite to extend: Plan A state tests.
- New test file justification: the persistent manifest is a new independently testable boundary.
- Temporary probes: none.

**Interfaces:**
- Consumes: `RunState`, `RunEventKind`, and `transition()` from Plan A.
- Produces: `RunManifest`, `RunEvent`, `FailureRecord`, `ArtifactRecord`, `JobRecord`, and `StatusRecorder`.

- [ ] **Step 1: Write failing recorder tests**

```python
import json

from dpeva.run import RunState
from dpeva.run.models import RunManifest
from dpeva.run.recorder import StatusRecorder


def test_failed_run_is_persisted(tmp_path) -> None:
    path = tmp_path / "run.json"
    recorder = StatusRecorder.create(path=path, run_id="feature-fixed", workflow="feature")
    recorder.transition(RunState.VALIDATED)
    recorder.transition(RunState.RUNNING)
    recorder.fail(category="EXECUTION", message="dp exited 2")
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "EXECUTION"
    assert not (tmp_path / "run.json.tmp").exists()


def test_manifest_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RunManifest.model_validate({"run_id": "x", "workflow": "infer", "status": "created", "mystery": 1})
```

Run: `pytest tests/unit/run/test_recorder.py -q`

Expected: FAIL because the models and recorder are absent.

- [ ] **Step 2: Implement the closed schema**

```python
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dpeva.run.status import RunState


class RunModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunEvent(RunModel):
    state: RunState
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: Literal["transition", "resume", "recovery", "force"] = "transition"
    attempt_id: int = 1


class FailureRecord(RunModel):
    category: Literal["CONFIG", "CAPABILITY", "ENVIRONMENT", "EXECUTION", "ARTIFACT", "DATA_INTEGRITY", "UPSTREAM"]
    message: str


class ArtifactRecord(RunModel):
    kind: str
    path: str
    producer_run: str
    status: Literal["declared", "verified", "missing"]
    checksum: str | None = None


class JobRecord(RunModel):
    name: str
    backend: Literal["local", "slurm"]
    job_id: str | None = None
    status: RunState
    failure: str | None = None


class RunManifest(RunModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    workflow: str
    status: RunState = RunState.CREATED
    source: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    config: dict[str, str] = Field(default_factory=dict)
    inputs: list[dict[str, str]] = Field(default_factory=list)
    jobs: list[JobRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)
    failure: FailureRecord | None = None
```

- [ ] **Step 3: Implement atomic persistence and state updates**

`StatusRecorder` must write `run.json.tmp`, flush and `os.fsync()`, then `os.replace()` it. `transition()` calls the Plan A transition function before appending a `RunEvent`; `fail()` records a `FailureRecord` and enters `failed`; `partial()` records the same failure shape and enters `partial`. No method may delete artifacts.

```python
def _write(self) -> None:
    temporary = self.path.with_suffix(self.path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(self.manifest.model_dump_json(indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, self.path)
```

- [ ] **Step 4: Run recorder tests**

Run: `pytest tests/unit/run/test_status.py tests/unit/run/test_recorder.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit run schema and recorder**

```bash
git add src/dpeva/run/__init__.py src/dpeva/run/models.py src/dpeva/run/recorder.py tests/unit/run/test_recorder.py
git commit -m "feat: add versioned run manifest recorder"
```

### Task 5: Create immutable run contexts with explicit resume/force semantics

**Files:**
- Create: `src/dpeva/run/context.py`
- Create: `tests/unit/run/test_context.py`
- Modify: `src/dpeva/run/__init__.py`

**Test strategy:**
- Behavior boundary: default IDs are unique; a requested existing ID fails; resume requires non-terminal state; force preserves the previous manifest as an attempt snapshot.
- Existing suite to extend: run package tests.
- New test file justification: directory allocation and overwrite policy are distinct from state persistence.
- Temporary probes: none.

**Interfaces:**
- Consumes: `StatusRecorder` and normalized/original configuration paths.
- Produces: `RunOptions` and `RunContext.create(work_dir, workflow, options, original_config, normalized_config) -> RunContext`.

- [ ] **Step 1: Write failing identity tests**

```python
from dpeva.run.context import RunContext, RunOptions


def test_existing_run_id_is_not_overwritten(tmp_path) -> None:
    options = RunOptions(run_id="feature-fixed")
    RunContext.create(tmp_path, "feature", options, {"x": 1}, {"x": 1})
    with pytest.raises(FileExistsError):
        RunContext.create(tmp_path, "feature", options, {"x": 1}, {"x": 1})


def test_force_archives_previous_manifest(tmp_path) -> None:
    options = RunOptions(run_id="feature-fixed")
    RunContext.create(tmp_path, "feature", options, {"x": 1}, {"x": 1})
    forced = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="feature-fixed", force=True, reason="rerun after corrected input"),
        {"x": 2},
        {"x": 2},
    )
    assert (forced.run_dir / "attempts" / "attempt-0001.json").exists()
```

Run: `pytest tests/unit/run/test_context.py -q`

Expected: FAIL because `RunContext` does not exist.

- [ ] **Step 2: Implement run allocation**

```python
@dataclass(frozen=True)
class RunOptions:
    run_id: str | None = None
    resume: bool = False
    force: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.resume and self.force:
            raise ValueError("--resume and --force are mutually exclusive")
        if (self.resume or self.force) and not self.run_id:
            raise ValueError("--resume/--force requires --run-id")
        if self.force and not self.reason:
            raise ValueError("--force requires --reason")
```

`RunContext.create()` stores runs under `work_dir/.dpeva/runs/{run-id}/`, writes `config.original.json` and `config.resolved.json`, and records relative references in the manifest. A generated ID uses `{workflow}-{UTC YYYYmmddTHHMMSSZ}-{six hex characters}`. Resume increments `attempt_id`; force first copies the old manifest to `attempts/attempt-NNNN.json` and records a `force` event. Implement `register_verified_artifacts(kind: str, paths: Sequence[Path]) -> None` to append `ArtifactRecord` entries with `producer_run=self.run_id`, paths relative to the run's work directory, and streaming SHA-256 checksums, then persist through `StatusRecorder`.

- [ ] **Step 3: Run context tests**

Run: `pytest tests/unit/run/test_context.py -q`

Expected: all tests pass.

- [ ] **Step 4: Commit immutable contexts**

```bash
git add src/dpeva/run/__init__.py src/dpeva/run/context.py tests/unit/run/test_context.py
git commit -m "feat: add immutable run contexts"
```

### Task 6: Pilot the run contract in feature and infer

**Files:**
- Modify: `src/dpeva/cli.py`
- Modify: `src/dpeva/workflows/feature.py`
- Modify: `src/dpeva/workflows/infer.py`
- Modify: `src/dpeva/inference/managers.py`
- Modify: `src/dpeva/utils/exceptions.py`
- Create: `src/dpeva/run/artifacts.py`
- Create: `tests/integration/test_run_contract_pilot.py`
- Modify: `tests/unit/workflows/test_feature_workflow_env.py`
- Modify: `tests/unit/workflows/test_infer_workflow_exec.py`
- Modify: `examples/recipes/README.md`
- Modify: `docs/guides/cli.md`
- Modify: `docs/guides/configuration.md`

**Test strategy:**
- Behavior boundary: feature and infer create manifests for success/failure; partial ensemble results are `partial` with CLI exit `1`; repeated IDs obey immutable/resume/force rules.
- Existing suite to extend: feature/infer workflow unit tests.
- New test file justification: the pilot spans CLI, run context, manager execution, artifacts, and exit semantics.
- Temporary probes: none.

**Interfaces:**
- Consumes: `RunOptions`, `RunContext`, migrated config result, and Plan A failure propagation.
- Produces: CLI options `--run-id`, `--resume`, `--force`, `--reason` for feature/infer; `PartialWorkflowError`; `validate_feature_outputs()` and `validate_inference_outputs()`; manifests at `.dpeva/runs/{run-id}/run.json`.

- [ ] **Step 1: Write failing pilot integration tests**

Create tests using fake `JobManager.submit()` results and small files so no real DeepMD process is required:

```python
import json
import subprocess

import pytest

from dpeva.config import FeatureConfig, InferenceConfig
from dpeva.run.context import RunOptions
from dpeva.utils.exceptions import PartialWorkflowError
from dpeva.workflows.feature import FeatureWorkflow
from dpeva.workflows.infer import InferenceWorkflow


def test_feature_failure_writes_failed_manifest(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "type.raw").write_text("0\n")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    output = tmp_path / "desc"

    def fail_submit(*args, **kwargs):
        raise subprocess.CalledProcessError(2, ["bash", "run_evaldesc.sh"])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", fail_submit)
    config = FeatureConfig(data_path=data, model_path=model, savedir=output)
    with pytest.raises(subprocess.CalledProcessError):
        FeatureWorkflow(
            config,
            original_config=config.model_dump(mode="json"),
            run_options=RunOptions(run_id="feature-failure"),
        ).run()
    payload = json.loads((output / ".dpeva/runs/feature-failure/run.json").read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "EXECUTION"


def test_infer_mixed_children_write_partial_manifest(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "type.raw").write_text("0\n")
    work = tmp_path / "work"
    for index in (0, 1):
        model = work / str(index) / "model.ckpt.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model")

    def mixed_submit(self, script_path, working_dir="."):
        directory = Path(working_dir)
        if directory.parts[-2] == "0":
            (directory / "results.e.out").write_text("0 0\n")
            return ""
        raise subprocess.CalledProcessError(4, ["bash", str(script_path)])

    monkeypatch.setattr("dpeva.submission.manager.JobManager.submit", mixed_submit)
    config = InferenceConfig(work_dir=work, data_path=data, task_name="test_val")
    with pytest.raises(PartialWorkflowError):
        InferenceWorkflow(
            config,
            original_config=config.model_dump(mode="json"),
            run_options=RunOptions(run_id="infer-partial"),
        ).run()
    payload = json.loads((work / ".dpeva/runs/infer-partial/run.json").read_text())
    assert payload["status"] == "partial"
    assert [job["status"] for job in payload["jobs"]] == ["finished", "failed"]
```

Import `Path` from `pathlib`; no external executable is permitted in this test file.

Run: `pytest tests/integration/test_run_contract_pilot.py -q`

Expected: FAIL because workflows do not accept a run context.

- [ ] **Step 2: Add shared run options to feature/infer parsers**

```python
def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reason")
```

Construct `RunOptions` in the handlers and pass the original plus normalized configuration to each workflow. Do not add these options to other commands in this pilot.

Change the feature constructor signature to `__init__(self, config: Union[Dict, FeatureConfig], *, original_config: dict[str, Any] | None = None, run_options: RunOptions | None = None)` and the inference signature to `__init__(self, config: Union[Dict, InferenceConfig], config_path: Optional[str] = None, *, original_config: dict[str, Any] | None = None, run_options: RunOptions | None = None)`. After the existing config model is created, assign:

```python
self.original_config = original_config or self.config.model_dump(mode="json")
self.run_options = run_options or RunOptions()
```

- [ ] **Step 3: Add concrete artifact validators and partial inference results**

```python
class ArtifactValidationError(RuntimeError):
    pass


def validate_feature_outputs(output_dir: Path, exporter: str) -> list[Path]:
    pattern = "embedding.hdf5" if exporter == "embed" else "*.npy"
    outputs = sorted(output_dir.rglob(pattern))
    if not outputs or any(path.stat().st_size == 0 for path in outputs):
        raise ArtifactValidationError(f"missing or empty feature artifacts under {output_dir}")
    return outputs


def validate_inference_outputs(output_dir: Path, prefix: str) -> list[Path]:
    outputs = sorted(output_dir.glob(f"{prefix}.*.out"))
    if not outputs or any(path.stat().st_size == 0 for path in outputs):
        raise ArtifactValidationError(f"missing or empty inference artifacts under {output_dir}")
    return outputs
```

Add `PartialWorkflowError(WorkflowError)` in `src/dpeva/utils/exceptions.py`. Change local `InferenceExecutionManager.submit_jobs()` to attempt every model and return `list[JobRecord]`; each record has `name=f"model-{index}"`, `backend="local"`, and status `finished` only after `validate_inference_outputs()` passes. A caught command or artifact exception yields `failed` with `failure=str(exc)`. Slurm returns `submitted` records with parsed JobIDs.

- [ ] **Step 4: Wrap workflow execution with state recording**

For each pilot workflow:

```python
context = RunContext.create(
    work_dir=Path(self.output_dir),
    workflow="feature",
    options=self.run_options,
    original_config=self.original_config,
    normalized_config=self.config.model_dump(mode="json"),
)
context.recorder.transition(RunState.VALIDATED)
try:
    context.recorder.transition(RunState.RUNNING)
    self._run_body()
    outputs = validate_feature_outputs(Path(self.output_dir), self.feature_exporter)
    context.register_verified_artifacts("feature", outputs)
    context.recorder.transition(RunState.FINISHED)
except ArtifactValidationError as exc:
    context.recorder.fail(category="ARTIFACT", message=str(exc))
    raise
except Exception as exc:
    context.recorder.fail(category="EXECUTION", message=str(exc))
    raise
```

Extract the current `run()` body into a private `_run_body()` without altering scientific operations. In inference, attach returned job records and call `recorder.partial(category="EXECUTION", message="one or more inference jobs failed")` when at least one, but not all, local jobs fail; then raise `PartialWorkflowError`. For Slurm, stop at `submitted`; completion recovery is outside this pilot unless the workflow is running locally.

- [ ] **Step 5: Run pilot and existing workflow tests**

Run: `pytest tests/integration/test_run_contract_pilot.py tests/unit/workflows/test_feature_workflow_env.py tests/unit/workflows/test_infer_workflow_exec.py -q`

Expected: all tests pass; failed and partial manifests remain readable.

- [ ] **Step 6: Run Phase 1 exit checks for R3/R7/R11/R12**

Run: `pytest tests/unit/run tests/unit/test_config_migration.py tests/unit/test_cli.py tests/integration/test_run_contract_pilot.py -q`

Expected: all tests pass and cover success/failure/partial, doctor states, typo rejection, migration warning/snapshot, and immutable rerun behavior.

- [ ] **Step 7: Commit the pilot**

```bash
git add src/dpeva/cli.py src/dpeva/workflows/feature.py src/dpeva/workflows/infer.py src/dpeva/inference/managers.py src/dpeva/utils/exceptions.py src/dpeva/run/artifacts.py tests/integration/test_run_contract_pilot.py tests/unit/workflows/test_feature_workflow_env.py tests/unit/workflows/test_infer_workflow_exec.py examples/recipes/README.md docs/guides/cli.md docs/guides/configuration.md
git commit -m "feat: pilot run contracts in feature and infer"
```

### Task 7: Stop/go review before expanding the contract

**Files:**
- Create: `docs/reports/2026-09-04-run-contract-pilot-report.md`

**Test strategy:**
- Behavior boundary: expansion is authorized only when the pilot shows unique diagnostic value without material runtime or schema burden.
- Existing suite to extend: none; this is an evidence checkpoint.
- New test file justification: none.
- Temporary probes: none.

**Interfaces:**
- Consumes: pilot manifests, test durations, failure messages, and developer feedback from Tasks 1–6.
- Produces: an explicit `GO` for Plans C/D or `STOP` with the contract kept feature/infer-only.

- [ ] **Step 1: Measure the pilot**

Run: `pytest tests/unit/run tests/integration/test_run_contract_pilot.py --durations=20 -q`

Expected: all tests pass; capture total duration and the slowest 20 tests.

- [ ] **Step 2: Write the pilot report with four measured questions**

Record:

1. Did each injected failure produce a more actionable category and evidence pointer than the pre-pilot log alone?
2. Did manifest creation add less than 100 ms median overhead in the local fake-command benchmark?
3. Did any schema field remain unread by tests, CLI output, recovery, or downstream plans?
4. Did any existing recipe require a semantic rewrite rather than a documented migration?

The decision is `GO` only when questions 1–2 are yes and questions 3–4 are no. Otherwise record `STOP`, remove unused fields before reconsideration, and keep other workflows unmodified.

Use this report table so the decision is mechanically reviewable:

```markdown
| Check | Measurement/evidence | Pass condition | Result |
|---|---|---|---|
| diagnostic value | failing test + manifest evidence pointer | all injected failures improve diagnosis | PASS/FAIL |
| median overhead | local fake-command benchmark, milliseconds | < 100 ms | PASS/FAIL |
| unused fields | field-to-consumer list | zero | PASS/FAIL |
| migration burden | changed recipes and reason | zero semantic rewrites | PASS/FAIL |

Decision: GO only when all four rows are PASS; otherwise STOP.
```

- [ ] **Step 3: Run repository checks and commit the checkpoint**

Run: `ruff check src tests scripts && pytest tests/unit -q && python3 scripts/doc_check.py && git diff --check`

Expected: all commands exit `0` except any separately documented pre-existing `doc_check.py` failures outside this plan's files.

```bash
git add docs/reports/2026-09-04-run-contract-pilot-report.md
git commit -m "docs: record run contract pilot decision"
```

Expected: Plans C and D remain blocked unless the report says `GO`.
