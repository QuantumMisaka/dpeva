---
title: DeepMD 3.2 Compatibility Lane Implementation Plan
status: proposed
audience: Developers / AI Agents
last-updated: 2026-09-04
owner: Compatibility Owner
---

# DeepMD 3.2 Compatibility Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepMD-kit 3.2 compatibility capability-driven, reproducible in CPU CI, and separately qualified on SAI V100 before any capability is declared supported.

**Spec:** `docs/superpowers/specs/2026-09-04-project-governance-and-deepmd-3-2-design.html` (`#deepmd`, `#testing` §13.2, `#requirements` R6/R8/R9, `#rollout` §15.4 and §15.9)

**Architecture:** Move DeepMD out of the core install into a bounded optional extra, load a packaged JSON capability manifest, and replace class-global command state with immutable `DeepMDAdapter` instances. Keep CPU contract fixtures minimal and move GPU/Slurm truth to one SAI qualification script plus report; promotion edits the manifest only after evidence exists.

**Tech Stack:** Python 3.10+, JSON, importlib.resources, pytest, GitHub Actions, DeepMD-kit 3.2.0, Slurm/SAI V100.

## Global Constraints

- Requires Plan B pilot checkpoint `GO`; Plan C may proceed in parallel.
- Package range is `deepmd-kit>=3.2,<3.3`; CI and production qualification use exact 3.2.0 (`#deepmd` §10.1).
- The only capability states are `supported`, `experimental`, `unsupported`, and `blocked-upstream`.
- `pt-expt` periodic DPA4C `eval-desc` starts experimental; non-PBC remains blocked-upstream; `pt-expt embed` is unsupported.
- CPU CI cannot promote GPU/Slurm claims. SAI evidence cannot imply universal backend correctness.
- Do not add DPA-ADAPT, JAX, TF2, GROMACS, automatic production upgrades, or Phase 3 features.

---

### Task 1: Split and bound the DeepMD dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/unit/test_dependency_contracts.py`
- Modify: `docs/reference/upstream-software.md`
- Modify: `docs/guides/configuration.md`
- Modify: `README.md`

**Test strategy:**
- Behavior boundary: core installation has no DeepMD requirement; `[deepmd]` carries the bounded range; `[dev]` remains usable without `dp`.
- Existing suite to extend: `tests/unit/test_dependency_contracts.py`.
- New test file justification: none.
- Temporary probes: a disposable virtual environment outside the repository; remove it after the install checks.

**Interfaces:**
- Consumes: setuptools optional dependencies.
- Produces: `dpeva[deepmd]` with `deepmd-kit>=3.2,<3.3` and documented exact-lock production guidance.

- [ ] **Step 1: Write failing dependency assertions**

```python
def test_deepmd_is_bounded_optional_dependency() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    core = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]
    assert all(not item.startswith("deepmd-kit") for item in core)
    assert extras["deepmd"] == ["deepmd-kit>=3.2,<3.3"]
```

Run: `pytest tests/unit/test_dependency_contracts.py -q`

Expected: FAIL because DeepMD is still an unbounded core dependency.

- [ ] **Step 2: Move the dependency and document installation modes**

```toml
[project.optional-dependencies]
deepmd = [
    "deepmd-kit>=3.2,<3.3",
]
```

Remove `deepmd-kit` from `[project].dependencies`. Document `pip install dpeva` for non-DeepMD workflows, `pip install 'dpeva[deepmd]'` for user runtime, and exact environment locks for research production.

- [ ] **Step 3: Verify both import modes**

Run: `pytest tests/unit/test_dependency_contracts.py tests/unit/test_cli.py::test_cli_dispatch_train_without_banner -q`

Expected: tests pass without invoking `dp` during import.

- [ ] **Step 4: Commit dependency boundaries**

```bash
git add pyproject.toml tests/unit/test_dependency_contracts.py docs/reference/upstream-software.md docs/guides/configuration.md README.md
git commit -m "build: bound DeepMD as an optional dependency"
```

### Task 2: Add the packaged capability manifest and query API

**Files:**
- Create: `src/dpeva/compatibility/__init__.py`
- Create: `src/dpeva/compatibility/deepmd.py`
- Create: `src/dpeva/compatibility/deepmd-3.2.json`
- Create: `tests/unit/compatibility/test_deepmd_matrix.py`
- Modify: `pyproject.toml`

**Test strategy:**
- Behavior boundary: callers can answer one exact operation/backend/family/artifact/data/environment query and receive a single valid state plus evidence metadata.
- Existing suite to extend: none.
- New test file justification: the versioned compatibility declaration is a new public machine-readable boundary.
- Temporary probes: none.

**Interfaces:**
- Consumes: packaged `deepmd-3.2.json`.
- Produces: `CapabilityKey`, `CapabilityRecord`, `CapabilityMatrix.load_default()`, `CapabilityMatrix.get(key) -> CapabilityRecord`, and `CapabilityMatrix.require(key, allow_experimental=False) -> CapabilityRecord`.

- [ ] **Step 1: Write failing matrix tests**

```python
import pytest

from dpeva.compatibility.deepmd import CapabilityKey, CapabilityMatrix, CapabilityUnavailable


def test_manifest_uses_only_canonical_states() -> None:
    matrix = CapabilityMatrix.load_default()
    assert {record.status for record in matrix.records} <= {
        "supported", "experimental", "unsupported", "blocked-upstream"
    }


def test_non_pbc_pt_expt_is_blocked() -> None:
    key = CapabilityKey(
        operation="eval-desc", backend="pt-expt", model_family="DPA4C",
        artifact="exportable", data_format="deepmd/npy", environment="cpu-non-pbc",
    )
    with pytest.raises(CapabilityUnavailable, match="blocked-upstream"):
        CapabilityMatrix.load_default().require(key)
```

Run: `pytest tests/unit/compatibility/test_deepmd_matrix.py -q`

Expected: collection fails because the compatibility package is absent.

- [ ] **Step 2: Create the exact initial manifest**

Use schema `1.0` with one record per SPEC §10.2 capability. Every record contains:

```json
{
  "key": {
    "operation": "eval-desc",
    "backend": "pt-expt",
    "model_family": "DPA4C",
    "artifact": "exportable",
    "data_format": "deepmd/npy",
    "environment": "cpu-periodic"
  },
  "status": "experimental",
  "version_range": ">=3.2,<3.3",
  "verification_command": "pytest tests/contract/deepmd/test_dpa4c_eval_desc.py -q",
  "evidence_ref": null,
  "upstream_issue": null
}
```

The non-PBC record has `status="blocked-upstream"` and `upstream_issue="https://github.com/deepmodeling/deepmd-kit/issues/6002"`. The `pt-expt/embed` record is `unsupported`.

- [ ] **Step 3: Implement strict loading and exact-key matching**

```python
class CapabilityKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: str
    backend: str
    model_family: str
    artifact: str
    data_format: str
    environment: str


class CapabilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cpu_contract: str
    sai_qualification: str | None = None


class CapabilityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: CapabilityKey
    status: Literal["supported", "experimental", "unsupported", "blocked-upstream"]
    version_range: str
    verification_command: str
    evidence_ref: CapabilityEvidence | None = None
    upstream_issue: str | None = None


class CapabilityUnavailable(RuntimeError):
    pass
```

`CapabilityMatrix.get()` returns the single exact-key record and raises `CapabilityUnavailable` for a missing key. `require()` delegates to `get()`, rejects unsupported/blocked records, and rejects experimental records unless `allow_experimental=True`. Duplicate keys make load fail.

- [ ] **Step 4: Package and test the JSON resource**

Add:

```toml
[tool.setuptools.package-data]
dpeva = ["compatibility/*.json"]
```

Run: `pytest tests/unit/compatibility/test_deepmd_matrix.py -q && python -c "from dpeva.compatibility.deepmd import CapabilityMatrix; print(len(CapabilityMatrix.load_default().records))"`

Expected: tests pass and the printed count matches the manifest record count.

- [ ] **Step 5: Commit the capability API**

```bash
git add src/dpeva/compatibility pyproject.toml tests/unit/compatibility/test_deepmd_matrix.py
git commit -m "feat: add DeepMD capability manifest"
```

### Task 3: Replace class-global command state with immutable adapters

**Files:**
- Create: `src/dpeva/compatibility/adapter.py`
- Create: `tests/unit/compatibility/test_deepmd_adapter.py`
- Modify: `src/dpeva/utils/command.py`
- Modify: `src/dpeva/training/managers.py`
- Modify: `src/dpeva/inference/managers.py`
- Modify: `src/dpeva/feature/managers.py`
- Modify: `tests/unit/utils/test_backend_config.py`
- Modify: manager test files for training/inference/feature.

**Test strategy:**
- Behavior boundary: concurrent adapters cannot leak backend state; each operation checks capability before returning an argv/shell command.
- Existing suite to extend: command builder and manager tests.
- New test file justification: adapter state isolation and capability preflight form a new boundary.
- Temporary probes: none.

**Interfaces:**
- Consumes: `CapabilityMatrix` and existing command quoting behavior.
- Produces: immutable `DeepMDAdapter(backend, matrix, allow_experimental)` methods `train`, `freeze`, `test`, `eval_desc`, and `embed`.

- [ ] **Step 1: Write failing isolation and capability tests**

```python
def test_adapter_backend_state_is_instance_local(matrix) -> None:
    pt = DeepMDAdapter("pt", matrix, allow_experimental=True)
    expt = DeepMDAdapter("pt-expt", matrix, allow_experimental=True)
    assert pt.base_command == ("dp", "--pt")
    assert expt.base_command == ("dp", "--pt-expt")
    assert pt.base_command == ("dp", "--pt")


def test_pt_expt_embed_is_rejected_before_command_build(matrix) -> None:
    adapter = DeepMDAdapter("pt-expt", matrix, allow_experimental=True)
    with pytest.raises(CapabilityUnavailable):
        adapter.embed(model="model.pt", system="data", output="embedding.hdf5", model_family="DPA4C")
```

Run: `pytest tests/unit/compatibility/test_deepmd_adapter.py -q`

Expected: FAIL because `DeepMDAdapter` is absent.

- [ ] **Step 2: Implement an immutable adapter**

Use `@dataclass(frozen=True)` with tuple `base_command`. It owns only backend-specific command construction; capability authorization remains the exact-key query from Task 2 and is called by workflow preflight before the adapter. This separation avoids smuggling model/data/environment schema into command builders. Return shell strings because existing managers and guards consume shell strings.

```python
@dataclass(frozen=True)
class DeepMDAdapter:
    backend: str

    def preflight(self, matrix: CapabilityMatrix, key: CapabilityKey,
                  allow_experimental: bool = False) -> CapabilityRecord:
        if key.backend != self.backend:
            raise CapabilityUnavailable(f"adapter backend {self.backend} does not match {key.backend}")
        return matrix.require(key, allow_experimental=allow_experimental)

    @property
    def base_command(self) -> tuple[str, str]:
        return ("dp", f"--{self.backend}")

    def train(self, input_file: str, finetune_path: str | None = None, init_model_path: str | None = None,
              skip_neighbor_stat: bool = False, log_file: str | None = None) -> str:
        argv = [*self.base_command, "train", input_file]
        if skip_neighbor_stat:
            argv.append("--skip-neighbor-stat")
        if finetune_path:
            argv.extend(["--finetune", finetune_path])
        elif init_model_path:
            argv.extend(["--init-model", init_model_path])
        return _with_log(shlex.join(argv), log_file)

    def freeze(self, output: str | None = None) -> str:
        argv = [*self.base_command, "freeze"]
        if output:
            argv.extend(["-o", output])
        return shlex.join(argv)

    def test(self, model: str, system: str, prefix: str, head: str | None = None,
             log_file: str | None = None) -> str:
        argv = [*self.base_command, "test", "-s", system, "-m", model, "-d", prefix]
        if head:
            argv.extend(["--head", head])
        return _with_log(shlex.join(argv), log_file)

    def eval_desc(self, model: str, system: str, output: str, head: str | None = None,
                  log_file: str | None = None) -> str:
        argv = [*self.base_command, "eval-desc", "-s", system, "-m", model, "-o", output]
        if head:
            argv.extend(["--head", head])
        return _with_log(shlex.join(argv), log_file)

    def embed(self, model: str, system: str, output: str, head: str | None = None,
              dtype: str = "fp32", log_file: str | None = None) -> str:
        argv = [*self.base_command, "embed", "-s", system, "-m", model, "-o", output, "--dtype", dtype]
        if head:
            argv.extend(["--head", head])
        return _with_log(shlex.join(argv), log_file)
```

Define `_with_log(command, log_file)` in the same module; it returns `command` unchanged when no log is requested and otherwise appends ` > {shlex.quote(log_file)} 2>&1`. Preserve the listed signatures exactly.

- [ ] **Step 3: Inject adapters into the three managers**

Manager constructors accept `adapter: DeepMDAdapter | None = None`; when absent they construct one from `dp_backend` and the default matrix. Remove all `DPCommandBuilder.set_backend()` calls. Keep `DPCommandBuilder` as a deprecated stateless compatibility facade for one release; its methods require an explicit backend parameter and hold no class variable.

Apply this constructor pattern in `TrainingExecutionManager`, `InferenceExecutionManager`, and `FeatureExecutionManager`:

```python
def __init__(self, *args, dp_backend: str = "pt", adapter: DeepMDAdapter | None = None, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.adapter = adapter or DeepMDAdapter(backend=dp_backend)
```

Replace each builder call with the corresponding `self.adapter.train(...)`, `self.adapter.freeze(...)`, `self.adapter.test(...)`, `self.adapter.eval_desc(...)`, or `self.adapter.embed(...)` call. No constructor or compatibility facade may write module/class state. Qualification and any workflow that already owns a complete `CapabilityKey` must call `adapter.preflight(...)` before command creation; do not add speculative config fields merely to manufacture a key in legacy paths.

- [ ] **Step 4: Run command and manager tests**

Run: `pytest tests/unit/compatibility/test_deepmd_adapter.py tests/unit/utils/test_backend_config.py tests/unit/training/test_training_managers.py tests/unit/inference/test_inference_execution_manager.py tests/unit/feature/test_execution_manager.py -q`

Expected: all tests pass; no test resets global backend state.

- [ ] **Step 5: Commit the adapter migration**

```bash
git add src/dpeva/compatibility/adapter.py src/dpeva/utils/command.py src/dpeva/training/managers.py src/dpeva/inference/managers.py src/dpeva/feature/managers.py tests/unit/compatibility/test_deepmd_adapter.py tests/unit/utils/test_backend_config.py tests/unit/training/test_training_managers.py tests/unit/inference/test_inference_execution_manager.py tests/unit/feature/test_execution_manager.py
git commit -m "refactor: isolate DeepMD command capabilities"
```

### Task 4: Add minimal real DeepMD 3.2 CPU contract tests

**Files:**
- Create: `tests/contract/deepmd/conftest.py`
- Create: `tests/contract/deepmd/test_cli_contract.py`
- Create: `tests/contract/deepmd/test_dpa4c_eval_desc.py`
- Create: `tests/contract/deepmd/data/README.md`
- Modify: `pyproject.toml`
- Create: `.github/workflows/deepmd-contract.yml`

**Test strategy:**
- Behavior boundary: real 3.2.0 executes PT test/eval-desc/embed and periodic pt-expt DPA4C eval-desc on the smallest licensed fixture; negative cases assert exit/status/artifact semantics.
- Existing suite to extend: none; current unit tests mock command execution.
- New test file justification: real upstream CLI behavior must remain isolated from unit tests and GPU qualification.
- Temporary probes: downloaded/generated model artifacts under pytest temporary directories only.

**Interfaces:**
- Consumes: exact DeepMD-kit 3.2.0, a tiny periodic Fe/C/H/O fixture, and test model references documented in `tests/contract/deepmd/data/README.md`.
- Produces: pytest marker `deepmd_contract` and CI artifact `deepmd-cpu-contract`.

- [ ] **Step 1: Register and gate the contract marker**

```toml
[tool.pytest.ini_options]
markers = [
    "deepmd_contract: real DeepMD 3.2 CPU CLI/API contract",
]
```

`conftest.py` skips only when an explicitly named fixture environment variable is absent; every skip message names the missing variable and owner. It must not skip after a command starts.

Use exactly `DPEVA_DEEPMD_PT_MODEL`, `DPEVA_DEEPMD_DPA4C_MODEL`, and `DPEVA_DEEPMD_PERIODIC_DATA`. Session fixtures resolve each value to an existing path before the first test; the public PT-model fixture may instead run `dp --pt pretrained download DPA-3.2-5M --cache-dir {pytest_cache_dir}`. DPA4C tests skip only when `DPEVA_DEEPMD_DPA4C_MODEL` is absent and therefore cannot promote that capability.

- [ ] **Step 2: Implement real subprocess assertions**

Each test calls `subprocess.run([...], check=False, capture_output=True, text=True)` with argv lists. Assert:

- `dp --version` parses as exactly `3.2.0` in stable CI.
- PT `test` creates numeric output with expected row/column shape.
- PT `eval-desc` creates descriptor arrays with frame count equal to the input fixture.
- PT `embed` creates HDF5 datasets `descriptor`, `atomic_feature`, `structural_feature`, and `atom_types`.
- periodic pt-expt DPA4C `eval-desc` returns zero and its output is consumed by `dpeva.io.collection`.
- pt-expt `embed` is rejected by preflight before subprocess launch.
- a fake zero-exit/no-artifact runner is classified `ARTIFACT`; a non-zero runner is `EXECUTION`.

Use this common assertion helper so zero exit alone never passes:

```python
def run_contract(argv: list[str], required_paths: list[Path]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    missing = [str(path) for path in required_paths if not path.exists()]
    assert missing == [], f"missing contract artifacts: {missing}"
    return result
```

- [ ] **Step 3: Add an exact-lock CI job**

The workflow installs the project without DeepMD, then installs `deepmd-kit==3.2.0` plus development dependencies, runs only `pytest -m deepmd_contract tests/contract/deepmd -q`, and uploads logs/artifact summaries even on failure. Trigger on relevant source/manifest/test paths and weekly schedule; do not run on documentation-only changes.

The command step is exactly:

```yaml
- name: Install exact DeepMD contract environment
  run: |
    python -m pip install --upgrade pip
    python -m pip install -e '.[dev]'
    python -m pip install 'deepmd-kit==3.2.0'
- name: Run DeepMD CPU contract
  run: pytest -m deepmd_contract tests/contract/deepmd -q
```

- [ ] **Step 4: Run the contract locally in the qualified CPU environment**

Run: `conda run -n ft2dp-post pytest -m deepmd_contract tests/contract/deepmd -q`

Expected: all enabled CPU contract tests pass; skips are limited to explicitly unavailable licensed/model fixtures documented in the data README.

- [ ] **Step 5: Commit CPU contracts**

```bash
git add tests/contract/deepmd pyproject.toml .github/workflows/deepmd-contract.yml
git commit -m "test: add DeepMD 3.2 CPU contracts"
```

### Task 5: Add one SAI V100 qualification harness and evidence schema

**Files:**
- Create: `scripts/validation/prepare_deepmd_32_qualification.py`
- Create: `scripts/validation/run_recorded_command.py`
- Create: `scripts/validation/run_deepmd_32_qualification.slurm`
- Create: `scripts/validation/submit_deepmd_32_qualification.py`
- Create: `scripts/validation/collect_deepmd_32_qualification.py`
- Create: `tests/unit/scripts/test_deepmd_32_qualification.py`
- Create: `docs/reports/templates/deepmd-3.2-qualification.md`
- Modify: `scripts/env/dpeva-dpa4.env`
- Modify: `docs/guides/testing/integration-slurm.md`

**Test strategy:**
- Behavior boundary: one bounded SAI job records environment lock, GPU, JobID, command outcomes, artifact checks, and regular/EMA or DPA4C evidence without promoting capabilities automatically.
- Existing suite to extend: none; existing validation scripts are ad hoc and do not emit this schema.
- New test file justification: the collector's report schema and fail-closed aggregation are locally testable without SAI.
- Temporary probes: SAI job directories live outside the repository and are referenced, not copied wholesale.

**Interfaces:**
- Consumes: `dpeva-dpa4` environment, SAI Slurm variables, explicit fixture/model paths, and capability keys.
- Produces: `build/deepmd-qualification/latest.json` (the submitted job reference), an immutable external job directory containing `qualification.json`, `commands/*.json`, logs and artifact references, plus a human report instantiated from the template.

- [ ] **Step 1: Write the collector schema test**

```python
def test_collector_refuses_partial_success(tmp_path) -> None:
    write_command_result(tmp_path, "pt-test", returncode=0, artifacts=["results.e.out"])
    write_command_result(tmp_path, "dpa4c-eval-desc", returncode=1, artifacts=[])
    report = collect_qualification(tmp_path, job_id="123", gpu="Tesla V100-SXM2-32GB")
    assert report["status"] == "failed"
    assert report["job_id"] == "123"
    assert report["failed_commands"] == ["dpa4c-eval-desc"]
```

Run: `pytest tests/unit/scripts/test_deepmd_32_qualification.py -q`

Expected: FAIL because the collector is absent.

- [ ] **Step 2: Implement fail-closed SAI commands**

The Slurm script starts with `set -Eeuo pipefail`, records `python -m pip freeze`, `dp --version`, `python -c 'import torch; ...'`, and `nvidia-smi`. It executes bounded single-GPU DPA4 test/eval-desc/embed, periodic DPA4C eval-desc, and regular/EMA checks. Each command writes argv, start/end time, return code, and declared artifacts to its own JSON result. The final collector runs from a shell trap and returns non-zero if any required command or artifact fails.

The preparer creates a four-element Fe/C/H/O periodic fixture under `build/deepmd-qualification/input/`, records the existing research-pipeline regular/EMA artifacts at `../v2.2-ft/runs/dpa4_multitask_forgetting_20260711/raw_snapshot/runs/balanced_lr350e-3/model.ckpt.pt` and `model_ema.ckpt.pt`, and writes their SHA-256 values to `build/deepmd-qualification/input.json`. The submitter re-verifies every path and checksum on the login node before `sbatch`; an absent path or mismatch exits non-zero without submitting. This deliberately consumes the research artifact without copying it into DP-EVA. The Slurm script receives only the verified config path and immutable output directory. Its control skeleton is:

```bash
#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
set -Eeuo pipefail
trap 'python scripts/validation/collect_deepmd_32_qualification.py --job-dir "$DPEVA_QUALIFICATION_DIR"' EXIT
python -m pip freeze > "$DPEVA_QUALIFICATION_DIR/pip-freeze.txt"
dp --version > "$DPEVA_QUALIFICATION_DIR/deepmd-version.txt"
nvidia-smi -L > "$DPEVA_QUALIFICATION_DIR/gpu.txt"
python scripts/validation/run_recorded_command.py --config "$DPEVA_QUALIFICATION_INPUT" --case pt-test
python scripts/validation/run_recorded_command.py --config "$DPEVA_QUALIFICATION_INPUT" --case pt-eval-desc
python scripts/validation/run_recorded_command.py --config "$DPEVA_QUALIFICATION_INPUT" --case pt-embed
python scripts/validation/run_recorded_command.py --config "$DPEVA_QUALIFICATION_INPUT" --case dpa4c-periodic-eval-desc
```

- [ ] **Step 3: Run local collector tests**

Run: `pytest tests/unit/scripts/test_deepmd_32_qualification.py -q && bash -n scripts/validation/run_deepmd_32_qualification.slurm`

Expected: tests pass and shell syntax is valid.

- [ ] **Step 4: Submit one bounded qualification job during execution**

Run: `python scripts/validation/prepare_deepmd_32_qualification.py --model-root ../v2.2-ft/runs/dpa4_multitask_forgetting_20260711/raw_snapshot/runs/balanced_lr350e-3 --output build/deepmd-qualification/input.json && python scripts/validation/submit_deepmd_32_qualification.py --input build/deepmd-qualification/input.json --slurm-script scripts/validation/run_deepmd_32_qualification.slurm --write-ref build/deepmd-qualification/latest.json`

Expected: submits exactly one job, prints one JobID, and atomically writes its immutable external job directory plus JobID to `build/deepmd-qualification/latest.json`. Monitor with `squeue`/`sacct`; submission is not completion.

- [ ] **Step 5: Validate the returned evidence**

Run: `python scripts/validation/collect_deepmd_32_qualification.py --job-ref build/deepmd-qualification/latest.json --require-complete`

Expected: exit `0`, `status="finished"`, exact DeepMD build and GPU identity present, and every declared artifact verified. The collector resolves only the recorded job directory and refuses a mutable `latest` directory or an unrecorded path.

- [ ] **Step 6: Commit the harness before attaching external evidence**

```bash
git add scripts/validation/prepare_deepmd_32_qualification.py scripts/validation/run_recorded_command.py scripts/validation/run_deepmd_32_qualification.slurm scripts/validation/submit_deepmd_32_qualification.py scripts/validation/collect_deepmd_32_qualification.py tests/unit/scripts/test_deepmd_32_qualification.py docs/reports/templates/deepmd-3.2-qualification.md scripts/env/dpeva-dpa4.env docs/guides/testing/integration-slurm.md
git commit -m "feat: add SAI DeepMD qualification harness"
```

### Task 6: Promote only evidence-backed capabilities and close Phase 2B

**Files:**
- Modify: `src/dpeva/compatibility/deepmd-3.2.json`
- Create: `docs/reports/2026-09-04-deepmd-3.2-compatibility.md`
- Modify: `docs/reference/upstream-software.md`
- Modify: `docs/guides/configuration.md`
- Modify: `examples/recipes/training/dpa4/README.md`

**Test strategy:**
- Behavior boundary: a capability becomes supported only when its exact verification command and CPU/SAI evidence requirements are satisfied.
- Existing suite to extend: capability matrix tests and CPU contracts.
- New test file justification: none.
- Temporary probes: none.

**Interfaces:**
- Consumes: CPU contract artifacts and the completed SAI `qualification.json`.
- Produces: updated statuses/evidence refs and a compatibility report bounded to the tested environment.

- [ ] **Step 1: Add promotion-validation tests**

For every `supported` record, assert `evidence_ref` is non-null, exists when repository-local, and names both CPU plus SAI evidence when the capability requires GPU qualification. Assert non-PBC pt-expt remains blocked until issue #6002 is closed and a fixed-version non-PBC contract is added.

```python
def test_supported_capabilities_have_required_evidence() -> None:
    matrix = CapabilityMatrix.load_default()
    for record in matrix.records:
        if record.status != "supported":
            continue
        assert record.evidence_ref
        if record.key.environment.startswith("sai-v100"):
            assert record.evidence_ref.cpu_contract
            assert record.evidence_ref.sai_qualification


def test_non_pbc_remains_blocked() -> None:
    record = CapabilityMatrix.load_default().get(non_pbc_dpa4c_key())
    assert record.status == "blocked-upstream"
    assert record.upstream_issue == "https://github.com/deepmodeling/deepmd-kit/issues/6002"
```

- [ ] **Step 2: Update only records proven by exact evidence**

Leave unproven entries `experimental`, `unsupported`, or `blocked-upstream`. Do not bulk-promote by DeepMD version number.

For each promoted JSON record, set `status` to `supported` and replace the null evidence with this closed object; if either file is absent, do not edit the state:

```json
{
  "cpu_contract": "docs/reports/2026-09-04-deepmd-3.2-compatibility.md#cpu-contract",
  "sai_qualification": "docs/reports/2026-09-04-deepmd-3.2-compatibility.md#sai-v100-qualification"
}
```

- [ ] **Step 3: Run Phase 2B checks**

Run: `pytest tests/unit/compatibility tests/unit/test_dependency_contracts.py -q && conda run -n ft2dp-post pytest -m deepmd_contract tests/contract/deepmd -q && git diff --check`

Expected: all enabled tests pass; the compatibility report lists exact skips and environment limitations.

- [ ] **Step 4: Commit qualification conclusions**

```bash
git add src/dpeva/compatibility/deepmd-3.2.json docs/reports/2026-09-04-deepmd-3.2-compatibility.md docs/reference/upstream-software.md docs/guides/configuration.md examples/recipes/training/dpa4/README.md
git commit -m "docs: qualify DeepMD 3.2 capabilities"
```
