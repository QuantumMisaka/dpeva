---
title: Dataset Lineage and Evaluation Card Implementation Plan
status: proposed
audience: Developers / AI Agents
last-updated: 2026-09-04
owner: Scientific Owner
---

# Dataset Lineage and Evaluation Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent silent dataset omission, represent model artifacts explicitly, and generate one machine-readable candidate evaluation card for FT2DP handoff.

**Spec:** `docs/superpowers/specs/2026-09-04-project-governance-and-deepmd-3-2-design.html` (`#architecture` §8.6–8.7/8.9, `#contracts` §9.2–9.3, `#requirements` R4/R5/R10, `#rollout` §15.3 and §15.9)

**Architecture:** Extend the proven `dpeva.run` package with three narrow data models and pure validators. `DataIntegrationManager` emits lineage beside existing outputs; inference accepts explicit model references while preserving legacy discovery for one release; a new `eval-card` command assembles references and statuses without copying active FT2DP checklists.

**Tech Stack:** Python 3.10+, Pydantic v2, dpdata, NumPy, JSON, pytest, existing labeling/inference/analysis modules.

## Global Constraints

- Requires Plan B pilot checkpoint `GO` and schema `1.0` conventions.
- DP-EVA stores evidence references to immutable, validated artifacts, not live FT2DP task state
  (`#boundaries`); relative paths provide portability only and are not the immutability mechanism.
- The 12,105 + 4,317 = 16,422 invariant is a mandatory regression boundary.
- Missing evaluation dimensions are explicit `not-run`, `not-applicable`, or `failed`; never encode them as zero.
- Do not implement a campaign database, automatic scientific ranking, or Phase 3 algorithms.

---

### Task 1: Define dataset lineage and count invariants

**Files:**
- Create: `src/dpeva/run/dataset.py`
- Create: `tests/unit/run/test_dataset_lineage.py`
- Modify: `src/dpeva/run/__init__.py`

**Test strategy:**
- Behavior boundary: parent counts, additions, removals, and final counts reconcile; omitted and double-counted frames fail.
- Existing suite to extend: none; this is a new pure schema boundary.
- New test file justification: the lineage invariant is independently reusable by integrate, collect, and clean.
- Temporary probes: none.

**Interfaces:**
- Consumes: parent dataset references and observed frame counts.
- Produces: `DatasetParent`, `DatasetManifest`, `LineageValidationError`, and `validate_lineage_counts(manifest: DatasetManifest) -> None`.

- [ ] **Step 1: Write the failing 16,422-frame tests**

```python
import pytest

from dpeva.run.dataset import DatasetManifest, DatasetParent, LineageValidationError, validate_lineage_counts


def test_iter11_accumulation_reconciles() -> None:
    manifest = DatasetManifest(
        dataset_id="ft2dp-iter11-cumulative",
        parents=[
            DatasetParent(dataset_id="iter10-cumulative", frame_count=12105),
            DatasetParent(dataset_id="iter11-new", frame_count=4317),
        ],
        transformation="merge",
        frame_count=16422,
        removed_frame_count=0,
        system_count=2,
        type_map=["Fe", "C", "H", "O"],
        format="deepmd/npy/mixed",
    )
    validate_lineage_counts(manifest)


def test_omitted_parent_frames_fail() -> None:
    manifest = DatasetManifest(
        dataset_id="broken",
        parents=[DatasetParent(dataset_id="old", frame_count=12105), DatasetParent(dataset_id="new", frame_count=4317)],
        transformation="merge",
        frame_count=4317,
        removed_frame_count=0,
        system_count=1,
        type_map=["Fe", "C", "H", "O"],
        format="deepmd/npy/mixed",
    )
    with pytest.raises(LineageValidationError, match="expected 16422, observed 4317"):
        validate_lineage_counts(manifest)
```

Run: `pytest tests/unit/run/test_dataset_lineage.py -q`

Expected: collection fails because `dpeva.run.dataset` does not exist.

- [ ] **Step 2: Implement the minimal lineage schema and validator**

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetParent(DatasetModel):
    dataset_id: str
    frame_count: int = Field(ge=0)
    manifest_ref: str | None = None


class DatasetManifest(DatasetModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    parents: list[DatasetParent]
    transformation: Literal["merge", "collect", "label", "clean", "import"]
    frame_count: int = Field(ge=0)
    removed_frame_count: int = Field(default=0, ge=0)
    system_count: int = Field(ge=0)
    type_map: list[str]
    format: str
    source_entries: list[str] = Field(default_factory=list)
    intersection_summary: DatasetIntersectionSummary
    content_identity: str | None = None
    validation_result: DatasetValidationResult


class LineageValidationError(ValueError):
    pass


def validate_lineage_counts(manifest: DatasetManifest) -> None:
    if not manifest.parents:
        return
    expected = sum(parent.frame_count for parent in manifest.parents) - manifest.removed_frame_count
    if expected != manifest.frame_count:
        raise LineageValidationError(f"lineage frame count mismatch: expected {expected}, observed {manifest.frame_count}")
```

- [ ] **Step 3: Run lineage tests and commit**

Run: `pytest tests/unit/run/test_dataset_lineage.py -q`

Expected: all tests pass.

```bash
git add src/dpeva/run/__init__.py src/dpeva/run/dataset.py tests/unit/run/test_dataset_lineage.py
git commit -m "feat: add dataset lineage invariants"
```

### Task 2: Emit lineage from labeling integration

**Files:**
- Modify: `src/dpeva/labeling/integration.py`
- Modify: `tests/unit/labeling/test_integration.py`
- Modify: `tests/integration/test_e2e_cycle.py`
- Modify: `docs/guides/cli.md`
- Modify: `docs/guides/configuration.md`

**Test strategy:**
- Behavior boundary: successful integration writes `dataset-manifest.json`; count/type-map conflicts,
  undeclared sources, and unexplained overlap fail before downstream handoff. Explicit deduplication
  persists overlap/removal evidence and a versioned validation result.
- Existing suite to extend: labeling integration unit and E2E tests.
- New test file justification: none.
- Temporary probes: none.

**Interfaces:**
- Consumes: `DatasetManifest`, `DatasetParent`, `validate_lineage_counts()` and existing dpdata counts.
- Produces: `{merged_output_path}/dataset-manifest.json` and returns `dataset_manifest_path` in `integration_summary.json`.

- [ ] **Step 1: Add failing integration assertions**

```python
summary = manager.integrate(
    new_labeled_data_path=new_dir,
    merged_output_path=out_dir,
    existing_training_data_path=existing_dir,
)
manifest = json.loads((out_dir / "dataset-manifest.json").read_text())
assert manifest["frame_count"] == summary["merged_frame_count_after_dedup"]
assert [parent["frame_count"] for parent in manifest["parents"]] == [
    summary["existing_frame_count"], summary["new_frame_count"]
]
assert summary["dataset_manifest_path"] == str(out_dir / "dataset-manifest.json")
```

Run: `pytest tests/unit/labeling/test_integration.py -q`

Expected: FAIL because the lineage file is absent.

- [ ] **Step 2: Build and validate the lineage before writing the summary**

```python
manifest = DatasetManifest(
    dataset_id=f"integration-{hashlib.sha256(str(merged_output_path).encode()).hexdigest()[:12]}",
    parents=[
        DatasetParent(dataset_id="existing-training", frame_count=existing_frames),
        DatasetParent(dataset_id="new-labeled", frame_count=new_frames),
    ],
    transformation="merge",
    frame_count=merged_frames_after_dedup,
    removed_frame_count=filtered_frames,
    system_count=after_dedup,
    type_map=list(reference_atom_names or []),
    format=self.output_format,
    source_entries=[str(path) for path in (existing_training_data_path, new_labeled_data_path) if path is not None],
)
validate_lineage_counts(manifest)
manifest_path = merged_output_path / "dataset-manifest.json"
manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
summary["dataset_manifest_path"] = str(manifest_path)
```

The path-derived ID is identity metadata, not an algorithm-correctness assertion. A future caller-supplied ID requires its own approved contract change rather than an unused field now.

- [ ] **Step 3: Run labeling tests and commit**

Run: `pytest tests/unit/labeling/test_integration.py tests/integration/test_e2e_cycle.py -q`

Expected: all tests pass and both summary plus manifest are present.

```bash
git add src/dpeva/labeling/integration.py tests/unit/labeling/test_integration.py tests/integration/test_e2e_cycle.py docs/guides/cli.md docs/guides/configuration.md
git commit -m "feat: emit lineage for integrated datasets"
```

### Task 3: Define explicit model artifact references and discovery

**Files:**
- Create: `src/dpeva/run/model.py`
- Create: `tests/unit/run/test_model_ref.py`
- Modify: `src/dpeva/config.py`
- Modify: `src/dpeva/workflows/infer.py`
- Modify: `src/dpeva/inference/managers.py`
- Modify: `tests/unit/inference/test_inference_io_manager.py`
- Modify: `src/dpeva/run/__init__.py`

**Test strategy:**
- Behavior boundary: regular/EMA and checkpoint/frozen/exportable roles are explicit; directory gaps do not truncate discovery; incompatible operation/artifact combinations fail before submission.
- Existing suite to extend: inference IO manager tests.
- New test file justification: model identity and operation compatibility are reusable outside inference.
- Temporary probes: none.

**Interfaces:**
- Consumes: model paths or aliases plus producer/runtime metadata.
- Produces: `ModelArtifactRef`, `ModelArtifactKind`, `ModelRole`, `load_model_ref(path: Path) -> ModelArtifactRef`, `resolve_model_refs(work_dir: Path, *, family: str, backend: str) -> list[ModelArtifactRef]`, `InferenceConfig.model_ref_paths`, and `require_operation(ref, operation) -> None`.

- [ ] **Step 1: Write failing model-reference tests**

```python
def test_discovery_handles_gaps_and_regular_ema(tmp_path) -> None:
    for relative in ("0/model.ckpt.pt", "0/model_ema.ckpt.pt", "2/model.ckpt.pt"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
    refs = resolve_model_refs(tmp_path, family="DPA4C", backend="pt-expt")
    assert [(ref.path, ref.role) for ref in refs] == [
        (str(tmp_path / "0/model.ckpt.pt"), ModelRole.REGULAR),
        (str(tmp_path / "0/model_ema.ckpt.pt"), ModelRole.EMA),
        (str(tmp_path / "2/model.ckpt.pt"), ModelRole.REGULAR),
    ]


def test_pretrained_alias_must_be_resolved_before_execution() -> None:
    ref = ModelArtifactRef(kind="pretrained-alias", family="DPA4", backend="pt", alias="DPA4-Air-OMat24-v20260805")
    with pytest.raises(ValueError, match="resolve pretrained alias"):
        require_operation(ref, "test")
```

Run: `pytest tests/unit/run/test_model_ref.py -q`

Expected: collection fails because the model-reference API is absent.

- [ ] **Step 2: Implement the closed model-reference schema**

```python
class ModelArtifactKind(str, Enum):
    CHECKPOINT = "checkpoint"
    FROZEN = "frozen"
    EXPORTABLE = "exportable"
    PRETRAINED_ALIAS = "pretrained-alias"


class ModelRole(str, Enum):
    REGULAR = "regular"
    EMA = "ema"


class ModelArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    kind: ModelArtifactKind
    family: str
    backend: str
    path: str | None = None
    alias: str | None = None
    resolved_path: str | None = None
    checksum: str | None = None
    head: str | None = None
    role: ModelRole = ModelRole.REGULAR
    deepmd_version: str | None = None
    producer_run: str | None = None
    supported_operations: list[str] = Field(default_factory=list)


def load_model_ref(path: Path) -> ModelArtifactRef:
    return ModelArtifactRef.model_validate_json(path.read_text(encoding="utf-8"))


def require_operation(ref: ModelArtifactRef, operation: str) -> None:
    if ref.kind is ModelArtifactKind.PRETRAINED_ALIAS and not ref.resolved_path:
        raise ValueError("resolve pretrained alias before execution")
    if operation not in ref.supported_operations:
        raise ValueError(f"model artifact does not declare operation: {operation}")
```

`resolve_model_refs()` uses `Path.iterdir()` over numeric directories sorted by integer, checks both `model.ckpt.pt` and `model_ema.ckpt.pt`, computes SHA-256, and never stops at a missing numeric directory.

- [ ] **Step 3: Replace inference's contiguous discovery assumption**

Add `model_ref_paths: list[Path] = Field(default_factory=list)` to `InferenceConfig`. `InferenceWorkflow` loads those JSON references with `load_model_ref()` and calls `require_operation(ref, "test")`; when the list is non-empty it does not scan `work_dir`. For one release, an empty list invokes `resolve_model_refs(work_dir, family="legacy-unknown", backend=config.dp_backend)`, logs one migration warning, and converts refs to resolved paths at the manager boundary. Do not add another directory convention, and do not claim a concrete model family for legacy discovery.

```python
def resolve_model_refs(work_dir: Path, *, family: str, backend: str) -> list[ModelArtifactRef]:
    refs: list[ModelArtifactRef] = []
    model_dirs = sorted(
        (path for path in work_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for model_dir in model_dirs:
        for filename, role in (("model.ckpt.pt", ModelRole.REGULAR), ("model_ema.ckpt.pt", ModelRole.EMA)):
            path = model_dir / filename
            if path.is_file():
                refs.append(ModelArtifactRef(
                    kind=ModelArtifactKind.CHECKPOINT,
                    family=family,
                    backend=backend,
                    path=str(path),
                    checksum=_sha256(path),
                    role=role,
                ))
    return refs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 4: Run model and inference tests, then commit**

Run: `pytest tests/unit/run/test_model_ref.py tests/unit/inference/test_inference_io_manager.py tests/unit/inference/test_inference_execution_manager.py -q`

Expected: all tests pass.

```bash
git add src/dpeva/run/__init__.py src/dpeva/run/model.py src/dpeva/config.py src/dpeva/workflows/infer.py src/dpeva/inference/managers.py tests/unit/run/test_model_ref.py tests/unit/inference/test_inference_io_manager.py
git commit -m "feat: make model artifact identity explicit"
```

### Task 4: Add the evaluation-card schema and assembler

**Files:**
- Create: `src/dpeva/evaluation/__init__.py`
- Create: `src/dpeva/evaluation/card.py`
- Create: `tests/unit/evaluation/test_card.py`
- Modify: `src/dpeva/config.py`

**Test strategy:**
- Behavior boundary: all six scientific dimensions have explicit status and evidence; absent values are not coerced to zero.
- Existing suite to extend: none.
- New test file justification: evaluation-card assembly is a new scientific handoff boundary.
- Temporary probes: none.

**Interfaces:**
- Consumes: dataset manifests, model refs, metric JSON paths, cost JSON, surface-slice JSON, and downstream feedback references.
- Produces: `EvaluationMetric`, `EvaluationCard`, `EvaluationCardConfig`, and `build_evaluation_card(config) -> EvaluationCard`.

- [ ] **Step 1: Write failing status-completeness tests**

```python
def test_missing_metrics_remain_not_run(tmp_path) -> None:
    config = EvaluationCardConfig(
        candidate_id="dpa4-air-iter11",
        output_path=tmp_path / "evaluation-card.json",
        model_ref_path=tmp_path / "model-ref.json",
        in_domain_cumulative_path=None,
        iter11_last_wave_path=None,
        historical_domain_path=None,
        matpes_retention_path=None,
        training_cost_path=None,
        surface_slice_path=None,
    )
    card = build_evaluation_card(config)
    assert set(card.metrics) == {
        "in_domain_cumulative", "iter11_last_wave", "historical_domain",
        "matpes_retention", "training_cost", "surface_slice",
    }
    assert all(metric.status == "not-run" for metric in card.metrics.values())
    assert all(metric.value is None for metric in card.metrics.values())
```

Run: `pytest tests/unit/evaluation/test_card.py -q`

Expected: collection fails because the evaluation package does not exist.

- [ ] **Step 2: Implement the schema and deterministic assembler**

```python
class EvaluationMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["passed", "failed", "not-run", "not-applicable"]
    value: dict[str, Any] | None = None
    evidence_ref: str | None = None
    detail: str | None = None


class EvaluationCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    candidate_id: str
    model_ref: str
    dataset_refs: list[str]
    metrics: dict[str, EvaluationMetric]
    downstream_feedback_ref: str | None = None


_DIMENSIONS = (
    "in_domain_cumulative", "iter11_last_wave", "historical_domain",
    "matpes_retention", "training_cost", "surface_slice",
)


class EvaluationCardConfig(StrictConfigModel):
    candidate_id: str
    model_ref_path: Path
    output_path: Path
    in_domain_cumulative_path: Path | None = None
    iter11_last_wave_path: Path | None = None
    historical_domain_path: Path | None = None
    matpes_retention_path: Path | None = None
    training_cost_path: Path | None = None
    surface_slice_path: Path | None = None
    dataset_manifest_paths: list[Path] = Field(default_factory=list)
    downstream_feedback_ref: str | None = None
```

`build_evaluation_card()` reads each configured JSON file, accepts only an object containing `status` plus optional `value/detail`, and emits `not-run` when no path is supplied. A malformed or missing configured path produces `failed` with the path as evidence; it does not abort assembly.

- [ ] **Step 3: Run evaluation unit tests and commit**

Run: `pytest tests/unit/evaluation/test_card.py -q`

Expected: all tests pass.

```bash
git add src/dpeva/evaluation/__init__.py src/dpeva/evaluation/card.py src/dpeva/config.py tests/unit/evaluation/test_card.py
git commit -m "feat: add candidate evaluation card schema"
```

### Task 5: Expose one `eval-card` command and candidate-package recipe

**Files:**
- Modify: `src/dpeva/cli.py`
- Modify: `tests/unit/test_cli.py`
- Create: `examples/recipes/evaluation/config_eval_card.json`
- Modify: `examples/recipes/README.md`
- Modify: `docs/guides/cli.md`
- Create: `tests/integration/test_evaluation_card_cli.py`

**Test strategy:**
- Behavior boundary: one command creates a stable card containing all six dimensions and references to
  immutable, validated evidence; missing inputs remain visible. Local references are portable POSIX paths
  relative to the card directory and must not leak machine absolute paths.
- Existing suite to extend: CLI tests.
- New test file justification: command-to-file handoff spans configuration, assembler, and JSON persistence.
- Temporary probes: none.

**Interfaces:**
- Consumes: `EvaluationCardConfig` and `build_evaluation_card()` from Task 4.
- Produces: `dpeva eval-card CONFIG.json` and `evaluation-card.json`.

- [ ] **Step 1: Write the failing CLI integration test**

```python
def test_eval_card_cli_writes_all_dimensions(tmp_path, monkeypatch) -> None:
    model_ref = tmp_path / "model-ref.json"
    model_ref.write_text('{"schema_version":"1.0","kind":"checkpoint","family":"DPA4","backend":"pt","path":"model.pt"}')
    output = tmp_path / "evaluation-card.json"
    config = tmp_path / "eval-card.json"
    config.write_text(json.dumps({"candidate_id": "candidate-1", "model_ref_path": str(model_ref), "output_path": str(output)}))
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "eval-card", str(config)])
    cli.main()
    payload = json.loads(output.read_text())
    assert len(payload["metrics"]) == 6
    assert payload["metrics"]["surface_slice"]["status"] == "not-run"
```

Run: `pytest tests/integration/test_evaluation_card_cli.py -q`

Expected: FAIL because the command is absent.

- [ ] **Step 2: Implement the CLI handler**

```python
def handle_eval_card(args) -> None:
    from dpeva.config import EvaluationCardConfig
    from dpeva.evaluation.card import build_evaluation_card
    migrated = load_and_resolve_config(args.config)
    config = EvaluationCardConfig.model_validate(migrated.normalized)
    card = build_evaluation_card(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
```

Register `eval-card` with the same validated JSON path behavior as other commands.

- [ ] **Step 3: Add the recipe without campaign-local absolute paths**

The recipe uses relative references for `model_ref_path`, optional metric JSON files, `output_path`, and
`downstream_feedback_ref`; it is a template requiring real model-ref and metric artifacts, not a directly
executable command. It must not duplicate FT2DP task status. Generated cards store POSIX references relative
to the card directory; readers resolve input paths first and never persist machine absolute paths.

```json
{
  "candidate_id": "candidate-1",
  "model_ref_path": "./model-ref.json",
  "output_path": "./evaluation-card.json",
  "in_domain_cumulative_path": null,
  "iter11_last_wave_path": null,
  "historical_domain_path": null,
  "matpes_retention_path": null,
  "training_cost_path": null,
  "surface_slice_path": null,
  "dataset_manifest_paths": [],
  "downstream_feedback_ref": null
}
```

- [ ] **Step 4: Run the R4/R5/R10 acceptance set**

Run: `pytest tests/unit/run/test_dataset_lineage.py tests/unit/run/test_model_ref.py tests/unit/labeling/test_integration.py tests/unit/evaluation/test_card.py tests/integration/test_evaluation_card_cli.py -q`

Expected: all tests pass, including 16,422-frame reconciliation, gap-tolerant regular/EMA discovery, and six explicit card statuses.

- [ ] **Step 5: Commit the command and recipe**

```bash
git add src/dpeva/cli.py tests/unit/test_cli.py tests/integration/test_evaluation_card_cli.py examples/recipes/evaluation/config_eval_card.json examples/recipes/README.md docs/guides/cli.md
git commit -m "feat: generate candidate evaluation cards"
```

### Task 6: Verify Phase 2A without expanding into campaign orchestration

**Files:**
- Create: `docs/reports/2026-09-04-dataset-lineage-eval-card-acceptance.md`

**Test strategy:**
- Behavior boundary: deliverables prove lineage and candidate evidence assembly only; no claim is made about final model quality.
- Existing suite to extend: full unit suite plus the two focused integrations.
- New test file justification: none.
- Temporary probes: none.

**Interfaces:**
- Consumes: Tasks 1–5 outputs.
- Produces: Phase 2A acceptance report with exact artifact paths and known scientific omissions.

- [ ] **Step 1: Run the phase gate**

Run: `ruff check src tests scripts && pytest tests/unit -q && pytest tests/integration/test_e2e_cycle.py tests/integration/test_evaluation_card_cli.py -q && git diff --check`

Expected: all commands exit `0`.

- [ ] **Step 2: Validate the shipped recipe and executable example boundary**

Run: `python -c "import json; from dpeva.config import EvaluationCardConfig; EvaluationCardConfig.model_validate(json.load(open('examples/recipes/evaluation/config_eval_card.json')))" && pytest tests/integration/test_evaluation_card_cli.py -q`

Expected: exit `0`; the template shape validates, while execution requires the caller to populate real
model-ref and metric artifacts. The integration test proves generated schema `1.0` output with unsupplied
dimensions represented as `not-run` rather than zero and all local references portable.

- [ ] **Step 3: Write and commit the acceptance report**

Record commands, artifact paths, schema versions, explicit missing dimensions, and the statement: “This gate validates evidence plumbing, not scientific superiority or downstream Fischer–Tropsch acceptance.”

```bash
git add docs/reports/2026-09-04-dataset-lineage-eval-card-acceptance.md
git commit -m "docs: record lineage and evaluation card acceptance"
```
