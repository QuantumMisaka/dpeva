---
title: Run Contract Pilot Stop/Go Report
status: active
audience: Developers / AI Agents
last-updated: 2026-09-04
owner: Project Maintainer
---

# Run Contract Pilot Stop/Go Report

Date: 2026-09-04
Scope: Plan B Tasks 1–7, feature/infer pilot only
Decision: **GO**

This report is an evidence checkpoint. It does not promote the run contract
to other workflows and does not treat a passing test suite as evidence that
every schema field has a consumer.

## Four-question decision table

| Check | Measurement/evidence | Pass condition | Result |
|---|---|---|---|
| diagnostic value | Injected CONFIG, CAPABILITY, EXECUTION, ARTIFACT, and local partial cases are mapped below. Doctor JSON gives a stable capability status/version/detail and a non-zero exit for unusable capability; feature/infer manifests add typed run and child evidence. | all injected failures improve diagnosis | PASS |
| median overhead | 41 paired repetitions of a local fake command; baseline measured fake command only, treatment measured the same command plus `StatusRecorder.create()` and atomic manifest publication. Median baseline 14.536ms, treatment 25.652ms, additional manifest overhead 11.047ms; p95 additional overhead 14.061ms. | < 100 ms | PASS |
| unused fields | Closed schema audit below. `source` and `inputs` are populated and asserted for both pilot workflows; `RunEvent.at` is tested as UTC, monotonic serialized evidence; legacy `environment` is preserved when present while new writes omit it. | zero | PASS |
| migration burden | `git diff 72620ad..HEAD -- examples/recipes` contains only the seven-line `examples/recipes/README.md` documentation addition. The 21 versioned JSON recipes were validated in Task 3; no recipe JSON was semantically rewritten. | zero semantic rewrites | PASS |

Decision: **GO**. Plans C and D may proceed, while preserving the feature/infer
scope boundary. The pilot does not authorize wiring the remaining workflows
until their own evidence contracts are implemented and reviewed.

## 1. Pilot test duration

Command:

```text
conda run -n dpeva-dpa4 pytest tests/unit/run tests/integration/test_run_contract_pilot.py --durations=20 -q
```

Result: `124 passed in 6.79s`.

Slowest 20 tests:

```text
0.26s call tests/integration/test_run_contract_pilot.py::test_cli_partial_exit_and_snapshots
0.24s call tests/unit/run/test_context.py::test_concurrent_force_allocates_unique_attempts
0.10s call tests/integration/test_run_contract_pilot.py::test_slurm_feature_and_infer_record_parsed_ids
0.09s call tests/integration/test_run_contract_pilot.py::test_infer_mixed_artifact_and_execution_failures_are_deterministic
0.08s call tests/unit/run/test_context.py::test_sequential_force_archives_resolve_all_config_references
0.08s call tests/integration/test_run_contract_pilot.py::test_infer_mixed_children_write_partial_manifest
0.08s call tests/integration/test_run_contract_pilot.py::test_infer_resume_of_submitted_slurm_is_legal
0.07s call tests/integration/test_run_contract_pilot.py::test_infer_analysis_failure_preserves_artifacts_and_failed_state
0.07s call tests/integration/test_run_contract_pilot.py::test_infer_success_manifest_and_artifact
0.07s call tests/integration/test_run_contract_pilot.py::test_infer_all_children_failure_writes_failed_manifest
0.07s call tests/integration/test_run_contract_pilot.py::test_feature_resume_of_submitted_slurm_is_legal
0.07s call tests/integration/test_run_contract_pilot.py::test_infer_empty_output_is_artifact_failure
0.07s call tests/unit/run/test_context.py::test_force_archives_previous_manifest_and_records_attempt
0.07s call tests/unit/run/test_context.py::test_force_failure_after_archive_reuses_archive_on_retry
0.06s call tests/integration/test_run_contract_pilot.py::test_infer_slurm_mixed_submission_stays_submitted
0.06s call tests/unit/run/test_context.py::test_force_publishes_manifest_with_event_in_one_replace
0.06s call tests/integration/test_run_contract_pilot.py::test_feature_success_manifest_contains_verified_output
0.06s call tests/unit/run/test_context.py::test_force_failure_before_archive_is_retry_stable
0.05s call tests/integration/test_run_contract_pilot.py::test_feature_multi_pool_requires_each_pool
0.05s call tests/integration/test_run_contract_pilot.py::test_infer_malformed_slurm_response_is_execution_failure[None]
```

## 2. Diagnostic value and evidence mapping

The pre-pilot baseline recorded a failing `test.log` and still accepted an
unconditional `WORKFLOW_FINISHED` marker for the local multidatapool case.
It did not provide a typed category, immutable run identity, child-level
outcome, or verified artifact list. The pilot closes that false-success path
for feature/infer runs.

| Injected case | Pilot evidence | Comparison with pre-pilot log | Assessment |
|---|---|---|---|
| CONFIG | `tests/unit/run/test_context.py::test_non_json_config_fails_closed_without_removing_run`; `.dpeva/runs/bad-config/run.json` has `status=failed`, `failure.category=CONFIG`, and the preserved configuration reference. | Pre-pilot had an exception without a durable run record or typed configuration category. | Improved |
| CAPABILITY | `tests/unit/run/test_doctor.py` and `tests/unit/test_cli.py` inject missing, unparsable, incompatible, and failed `dp --version` responses; `dpeva doctor --json` exposes `checks[0].status`, `version`, and `detail`, and exits non-zero when unusable. No run manifest is created because doctor is intentionally config-free; its JSON is the capability evidence pointer. | Pre-pilot import-time probing emitted a PATH warning and had no stable JSON capability evidence. | Improved with a dedicated, machine-readable capability boundary. |
| EXECUTION | `tests/integration/test_run_contract_pilot.py::test_feature_failure_writes_failed_manifest`; `.dpeva/runs/feature-failure/run.json` has `status=failed`, `failure.category=EXECUTION`. Slurm malformed/all-fail cases also preserve child `failure_category=EXECUTION`. | Pre-pilot local DeepMD failure was followed by a completion marker and was treated as submission success. | Improved |
| ARTIFACT | `test_feature_missing_output_is_artifact_failure`, `test_feature_multi_pool_requires_each_pool`, and `test_infer_empty_output_is_artifact_failure`; each manifest has `status=failed`, `failure.category=ARTIFACT`, and inference child records retain `failure_category=ARTIFACT`. | Pre-pilot only showed absent `results.*.out` files after a misleading successful marker; no durable artifact verdict existed. | Improved |
| local partial | `test_infer_mixed_children_write_partial_manifest`; `.dpeva/runs/infer-partial/run.json` has `status=partial`, typed top-level failure, one `finished` child, one `failed` child, and a verified artifact. | Pre-pilot had no partial state or child-level aggregation; the marker could not distinguish partial scientific output from success. | Improved |

The pilot demonstrates unique diagnostic value for the feature/infer execution
boundary and for capability preflight. Capability evidence intentionally lives
in the doctor JSON contract rather than a workflow manifest, because doctor is
configuration-free and must not allocate a run identity.

## 3. Reproducible fake-command benchmark

The benchmark was run inside `dpeva-dpa4` with 5 warmups and 41 measured
pairs. Each baseline repetition runs the same no-op Python subprocess. Each
treatment repetition runs that subprocess and then creates a fresh run
manifest using `StatusRecorder.create()`, including file flush/fsync,
atomic replace, and parent-directory fsync. Timings cover only the paired
operation, not conda startup.

Exact command:

```text
conda run -n dpeva-dpa4 python scripts/benchmark_run_manifest.py --repetitions 41 --warmups 5
```

The benchmark implementation is versioned at
`scripts/benchmark_run_manifest.py`; it emits one JSON object containing both
raw arrays, summary values, and the percentile method.

Raw baseline milliseconds:

```text
[14.707, 13.426, 14.418, 13.897, 14.591, 14.363, 14.337, 14.323, 13.356, 13.826, 13.639, 15.307, 15.282, 15.214, 14.642, 15.635, 15.297, 15.996, 14.536, 14.764, 13.813, 14.378, 14.742, 13.702, 14.192, 15.127, 18.294, 14.875, 13.544, 13.303, 14.132, 13.885, 15.003, 14.323, 14.302, 15.820, 14.506, 15.326, 15.311, 23.262, 17.521]
```

Raw treatment milliseconds:

```text
[26.673, 24.213, 24.272, 24.486, 26.955, 25.058, 24.378, 23.508, 23.387, 24.873, 24.080, 27.064, 29.977, 27.265, 26.676, 34.604, 24.243, 26.060, 26.195, 28.824, 24.885, 27.164, 25.868, 25.036, 27.378, 27.224, 26.848, 22.833, 24.219, 24.441, 26.541, 23.884, 24.418, 23.401, 26.341, 25.652, 23.255, 27.264, 25.262, 28.424, 30.625]
```

Raw paired overhead milliseconds (`treatment - baseline`):

```text
[11.966, 10.787, 9.854, 10.589, 12.363, 10.694, 10.041, 9.185, 10.031, 11.047, 10.441, 11.757, 14.695, 12.051, 12.034, 18.969, 8.945, 10.064, 11.659, 14.061, 11.073, 12.786, 11.126, 11.334, 13.186, 12.098, 8.553, 7.957, 10.675, 11.138, 12.410, 9.999, 9.415, 9.079, 12.039, 9.831, 8.748, 11.938, 9.950, 5.162, 13.104]
```

Summary: median baseline `14.536ms`, median with manifest `25.652ms`,
median manifest overhead `11.047ms`, and p95 overhead `14.061ms`. This
passes the `<100ms` threshold for this local fake-command workload; it does
not claim anything about scheduler or real DeepMD runtime overhead.

## 4. Schema 1.0 field audit

The audit covers every field in the persisted run models, including fields
added during reviewer fixes. “Writer” alone is not counted as a consumer.
The companion doctor schema is also fully consumed: `DoctorReport.schema_version`
is asserted by `test_doctor_report_is_json_serializable`, `DoctorReport.status`
drives the CLI exit code, `DoctorReport.checks` drives human/JSON output, and
`DoctorCheck.name/status/version/detail` are asserted by the injected probe and
CLI tests. It is intentionally separate from run-manifest evidence.

| Model.field | Concrete current consumer/test/recovery | Downstream plan, if any | Audit |
|---|---|---|---|
| `RunManifest.schema_version` | `StatusRecorder.create/load`; `test_failed_run_is_persisted_atomically` | Plans C/D consume versioned run evidence | Used |
| `RunManifest.run_id` | `RunContext` allocation and identity checks; rerun/force tests | Plan E traceability will link evidence by identity | Used |
| `RunManifest.workflow` | `RunContext._load_existing()` verifies workflow; pilot manifests | Plans C/D preserve workflow ownership in their evidence | Used |
| `RunManifest.status` | `StatusRecorder.transition`; feature/infer terminal decisions; status tests | Plans C/D use terminal status for acceptance | Used |
| `RunManifest.source` | Feature/infer pass `{"dpeva_version": ...}`; pilot success tests read and assert it | Plans C/D retain source identity when extending run evidence | Used |
| `RunManifest.environment` | Optional legacy-only schema 1.0 evidence; `StatusRecorder.load()` validates and preserves a present mapping through save/transition/resume, while force archives retain the old mapping and the new current manifest omits `None`; exact compatibility tests cover each path | Legacy compatibility/preservation is the consumer; no new pilot data is written | Legacy-preserved |
| `RunManifest.config` | `RunContext` writes original/resolved snapshot references; force/recovery tests read them | Plan E traceability can link config snapshots after an explicit consumer is specified | Used |
| `RunManifest.inputs` | Feature/infer pass absolute dataset/model references; pilot success tests read and assert the exact list | Plans C/D extend these explicit input references with lineage/model evidence | Used |
| `RunManifest.jobs` | Feature/infer managers append `JobRecord`; pilot asserts statuses and JobIDs | Plan D qualification has a separate command-result schema | Used |
| `RunManifest.artifacts` | `RunContext.register_verified_artifacts()` and pilot artifact assertions | Plan C consumes dataset/model artifact references | Used |
| `RunManifest.events` | Recorder appends transitions/resume/recovery/force; recovery inspects history | Plan E can audit event history after a concrete reader is defined | Used |
| `RunManifest.failure` | `fail/partial`, workflow aggregation, and failure assertions | Plans C/D require fail-closed evidence for their own outputs | Used |
| `RunEvent.state` | State transition and recovery logic; recorder/status tests | Downstream acceptance reads event state history | Used |
| `RunEvent.at` | Recorder timestamp is asserted UTC, monotonic, and equal after JSON load by `test_event_timestamps_are_utc_monotonic_and_round_trip` | Plan E consumes event timestamps for audit chronology | Used |
| `RunEvent.kind` | Resume/recovery/force behavior and tests | Plan E event audit can consume it | Used |
| `RunEvent.attempt_id` | Context attempt allocation and recovery attribution; tests | Plan E traceability can distinguish attempts | Used |
| `RunEvent.reason` | Force validation and force-event tests | Future operator audit consumes explicit rerun reason | Used |
| `RunEvent.failure` | Terminal event evidence and legacy recovery enrichment; recorder tests | Plan E audit can preserve terminal failure evidence | Used |
| `FailureRecord.category` | Workflow aggregation and category assertions for CONFIG/EXECUTION/ARTIFACT | Plan D capability lane can add CAPABILITY evidence only with an actual workflow consumer | Used |
| `FailureRecord.message` | Recorder persistence and failure assertions | Downstream reports retain actionable failure text | Used |
| `ArtifactRecord.kind` | Feature/infer registration and artifact assertions | Plan C distinguishes dataset/model/evaluation artifacts | Used |
| `ArtifactRecord.path` | Relative-path validation and artifact assertions | Plans C/D consume evidence references | Used |
| `ArtifactRecord.producer_run` | Registration sets producer identity; serialized records preserve it | Plan C candidate card links artifacts to producer runs | Used |
| `ArtifactRecord.status` | Validators only register verified artifacts; pilot asserts verified status | Plans C/D acceptance checks require verified outputs | Used |
| `ArtifactRecord.checksum` | Streaming SHA-256 registration; context checksum test | Plan C model/dataset identity uses immutable checksums | Used |
| `JobRecord.name` | Managers create stable model names; pilot records child evidence | Plan D qualification keeps per-command names in its own schema | Used |
| `JobRecord.backend` | Local/Slurm manager branches and serialized records | Plans C/D retain backend-specific evidence | Used |
| `JobRecord.job_id` | Slurm parsing and submitted-state tests | Plan D qualification records scheduler JobID | Used |
| `JobRecord.status` | Manager aggregation and pilot status assertions | Plans C/D acceptance reads child outcomes | Used |
| `JobRecord.failure` | Inference manager stores caught command/artifact text for child diagnostics | Plan D can retain command failure text in its qualification results | Used |
| `JobRecord.failure_category` | Inference aggregation and mixed artifact/execution assertions | Plan D's typed command outcomes follow the same distinction | Used |

`RunManifest.environment` is retained only as optional legacy schema 1.0
evidence. Loading, state transitions, resume, and force archiving preserve a
present mapping unchanged; newly-created and newly-forced current manifests
omit the unset field. Legacy compatibility/preservation is the consumer, not
new pilot data collection. No other field is retained without a concrete
current consumer, test, recovery behavior, or named downstream plan.

## 5. Recipe audit

The complete recipe diff from planning commit `72620ad` to the current
checkpoint is:

```text
M examples/recipes/README.md (7 insertions, 0 deletions)
```

The change documents `feature`/`infer` run options and the distinction between
local partial and Slurm submitted states. It changes no versioned JSON input,
scientific parameter, model path, backend default, or output convention. The
exact Task 3 validation command, rerun for this checkpoint, is:

```text
conda run -n dpeva-dpa4 python -c 'exec("""import json
from pathlib import Path
from dpeva.config import AnalysisConfig, CollectionConfig, DataCleaningConfig, ExplorationConfig, FeatureConfig, InferenceConfig, LabelingConfig, TrainingConfig
from dpeva.config_migration import migrate_legacy_config
classes = {"analysis": AnalysisConfig, "collection": CollectionConfig, "data_cleaning": DataCleaningConfig, "exploration": ExplorationConfig, "feature_generation": FeatureConfig, "inference": InferenceConfig, "labeling": LabelingConfig, "training": TrainingConfig}
paths = sorted(p for p in Path("examples/recipes").rglob("*.json") if p.name != "input.json")
for path in paths:
    normalized = migrate_legacy_config(json.loads(path.read_text(encoding="utf-8"))).normalized
    model = classes.get(path.parts[-2])
    if model is not None:
        model.model_validate(normalized)
    else:
        assert isinstance(normalized, dict), path
print(f"validated {len(paths)} versioned recipe configs")
""")'
```

Result: `validated 21 versioned recipe configs`. See the detailed Task 3
evidence in `.superpowers/sdd/2026-09-04-run-contract-strict-config/task-3-report.md`.
Therefore the migration-burden check is PASS: there are zero semantic recipe
rewrites.

## 6. Required repository checks

These commands are the Task 7 repository gate; their exact results are
recorded below after the report was written:

```text
ruff check src tests scripts
pytest tests/unit -q
python3 scripts/doc_check.py
git diff --check
```

Any pre-existing documentation failures outside this report and the Plan B
files are classified by path and message rather than silently ignored.

Observed results:

- `conda run -n dpeva-dpa4 ruff check src tests scripts` — exit `0`,
  `All checks passed!`.
- `conda run -n dpeva-dpa4 pytest tests/unit -q` — exit `0`, `640 passed in
  23.53s`.
- `python3 scripts/doc_check.py` — exit `0`; structure, metadata, links,
  forbidden-path, and owner checks all pass, including the repaired Plan A
  integration classification report.
- `git diff --check` — exit `0`.

No pre-existing documentation failure remains in this checkpoint.
