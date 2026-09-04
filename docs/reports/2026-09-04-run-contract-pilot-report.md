---
title: Run Contract Pilot Stop/Go Report
status: active
audience: Developers / AI Agents
last-updated: 2026-09-04
owner: Project Maintainer
---

# Run Contract Pilot Stop/Go Report

Date: 2026-09-04  
Scope: Plan B Tasks 1–6, feature/infer pilot only  
Decision: **STOP**

This report is an evidence checkpoint. It does not promote the run contract
to other workflows and does not treat a passing test suite as evidence that
every schema field has a consumer.

## Four-question decision table

| Check | Measurement/evidence | Pass condition | Result |
|---|---|---|---|
| diagnostic value | Injected CONFIG, CAPABILITY, EXECUTION, ARTIFACT, and local partial cases are mapped below; stable run paths, categories, child records, and retained artifacts are present for the feature/infer cases. The CAPABILITY case is a doctor JSON report rather than a run manifest, so the requested all-cases manifest mapping is incomplete. | all injected failures improve diagnosis | FAIL |
| median overhead | 41 paired repetitions of a local fake command; baseline measured fake command only, treatment measured the same command plus `StatusRecorder.create()` and atomic manifest publication. Median baseline 15.407ms, treatment 26.281ms, additional manifest overhead 10.074ms; p95 additional overhead 12.845ms. | < 100 ms | PASS |
| unused fields | Closed schema audit below. `RunManifest.source`, `RunManifest.environment`, `RunManifest.inputs`, and `RunEvent.at` have no current reader, assertion, recovery consumer, or concrete downstream-plan consumer. They are currently write/persist fields only. | zero | FAIL |
| migration burden | `git diff 72620ad..HEAD -- examples/recipes` contains only the seven-line `examples/recipes/README.md` documentation addition. The 21 versioned JSON recipes were validated in Task 3; no recipe JSON was semantically rewritten. | zero semantic rewrites | PASS |

Decision: **STOP**. Plans C and D must remain blocked. Before a new
checkpoint, remove or explicitly re-scope the four unused fields through a
spec/plan change with tests; do not add speculative consumers merely to turn
this result into GO. The feature/infer pilot remains available as-is.

## 1. Pilot test duration

Command:

```text
conda run -n dpeva-dpa4 pytest tests/unit/run tests/integration/test_run_contract_pilot.py --durations=20 -q
```

Result: `120 passed in 7.41s`.

Slowest 20 tests:

```text
0.33s call tests/integration/test_run_contract_pilot.py::test_cli_partial_exit_and_snapshots
0.26s call tests/unit/run/test_context.py::test_concurrent_force_allocates_unique_attempts
0.13s call tests/integration/test_run_contract_pilot.py::test_infer_mixed_artifact_and_execution_failures_are_deterministic
0.11s call tests/integration/test_run_contract_pilot.py::test_slurm_feature_and_infer_record_parsed_ids
0.11s call tests/integration/test_run_contract_pilot.py::test_infer_analysis_failure_preserves_artifacts_and_failed_state
0.10s call tests/integration/test_run_contract_pilot.py::test_infer_resume_of_submitted_slurm_is_legal
0.10s call tests/integration/test_run_contract_pilot.py::test_infer_mixed_children_write_partial_manifest
0.08s call tests/integration/test_run_contract_pilot.py::test_infer_all_children_failure_writes_failed_manifest
0.08s call tests/integration/test_run_contract_pilot.py::test_feature_resume_of_submitted_slurm_is_legal
0.08s call tests/integration/test_run_contract_pilot.py::test_infer_success_manifest_and_artifact
0.07s call tests/integration/test_run_contract_pilot.py::test_feature_success_manifest_contains_verified_output
0.07s call tests/integration/test_run_contract_pilot.py::test_infer_malformed_slurm_response_is_execution_failure[None]
0.07s call tests/unit/run/test_context.py::test_force_archives_previous_manifest_and_records_attempt
0.07s call tests/unit/run/test_context.py::test_sequential_force_archives_resolve_all_config_references
0.07s call tests/integration/test_run_contract_pilot.py::test_infer_malformed_slurm_response_is_execution_failure[sbatch output without a job id]
0.07s call tests/integration/test_run_contract_pilot.py::test_infer_slurm_mixed_submission_stays_submitted
0.07s call tests/integration/test_run_contract_pilot.py::test_infer_empty_output_is_artifact_failure
0.06s call tests/integration/test_run_contract_pilot.py::test_infer_slurm_all_fail_is_execution_failure
0.06s call tests/unit/run/test_recorder.py::test_record_event_on_current_failure_attaches_failure_evidence[failed]
0.06s call tests/integration/test_run_contract_pilot.py::test_feature_missing_output_is_artifact_failure
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
| CAPABILITY | `tests/unit/run/test_doctor.py` and `tests/unit/test_cli.py` inject missing, unparsable, incompatible, and failed `dp --version` responses; `dpeva doctor --json` exposes `checks[0].status`, `version`, and `detail`. No run manifest is created because doctor is intentionally config-free. | Pre-pilot import-time probing emitted a PATH warning and had no stable JSON capability evidence. | Improved doctor evidence, but not a manifest mapping; this is why the diagnostic row is FAIL under the stated all-cases rule. |
| EXECUTION | `tests/integration/test_run_contract_pilot.py::test_feature_failure_writes_failed_manifest`; `.dpeva/runs/feature-failure/run.json` has `status=failed`, `failure.category=EXECUTION`. Slurm malformed/all-fail cases also preserve child `failure_category=EXECUTION`. | Pre-pilot local DeepMD failure was followed by a completion marker and was treated as submission success. | Improved |
| ARTIFACT | `test_feature_missing_output_is_artifact_failure`, `test_feature_multi_pool_requires_each_pool`, and `test_infer_empty_output_is_artifact_failure`; each manifest has `status=failed`, `failure.category=ARTIFACT`, and inference child records retain `failure_category=ARTIFACT`. | Pre-pilot only showed absent `results.*.out` files after a misleading successful marker; no durable artifact verdict existed. | Improved |
| local partial | `test_infer_mixed_children_write_partial_manifest`; `.dpeva/runs/infer-partial/run.json` has `status=partial`, typed top-level failure, one `finished` child, one `failed` child, and a verified artifact. | Pre-pilot had no partial state or child-level aggregation; the marker could not distinguish partial scientific output from success. | Improved |

The pilot therefore demonstrates unique diagnostic value for the feature and
infer execution boundary, but does not yet demonstrate the exact requested
manifest evidence boundary for capability failures.

## 3. Reproducible fake-command benchmark

The benchmark was run inside `dpeva-dpa4` with 5 warmups and 41 measured
pairs. Each baseline repetition runs the same no-op Python subprocess. Each
treatment repetition runs that subprocess and then creates a fresh run
manifest using `StatusRecorder.create()`, including file flush/fsync,
atomic replace, and parent-directory fsync. Timings cover only the paired
operation, not conda startup.

Command shape:

```text
conda run -n dpeva-dpa4 python -c '<41 paired subprocess-only vs subprocess-plus-StatusRecorder.create repetitions>'
```

Raw baseline milliseconds:

```text
[15.477, 17.756, 17.040, 16.529, 15.927, 16.013, 14.995, 15.273, 15.258, 14.469, 14.860, 16.162, 14.957, 14.452, 14.659, 15.184, 15.084, 16.247, 15.192, 14.693, 16.127, 15.082, 16.741, 17.103, 15.994, 20.151, 15.873, 15.965, 14.696, 15.111, 16.702, 16.367, 14.923, 15.407, 15.190, 14.862, 15.880, 16.741, 16.605, 13.902, 14.559]
```

Raw treatment milliseconds:

```text
[28.599, 26.565, 27.007, 26.603, 25.250, 28.269, 29.246, 25.809, 25.885, 24.761, 26.281, 25.900, 27.275, 24.937, 25.677, 24.997, 25.112, 28.041, 25.090, 27.489, 26.004, 27.168, 26.490, 29.948, 27.858, 27.141, 27.244, 28.133, 27.425, 25.157, 26.655, 24.720, 29.214, 27.865, 24.140, 23.827, 23.624, 22.745, 25.532, 22.264, 23.579]
```

Raw paired overhead milliseconds (`treatment - baseline`):

```text
[13.121, 8.809, 9.967, 10.074, 9.323, 12.256, 14.252, 10.536, 10.628, 10.292, 11.420, 9.739, 12.318, 10.485, 11.017, 9.813, 10.028, 11.794, 9.897, 12.795, 9.877, 12.086, 9.749, 12.845, 11.864, 6.990, 11.370, 12.168, 12.728, 10.047, 9.953, 8.353, 14.291, 12.459, 8.950, 8.965, 7.745, 6.005, 8.927, 8.362, 9.020]
```

Summary: median baseline `15.407ms`, median with manifest `26.281ms`,
median manifest overhead `10.074ms`, and p95 overhead `12.845ms`. This
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
| `RunManifest.source` | No production caller supplies it; no test reads it; recovery does not use it | No concrete source-field consumer is specified in Plans C, D, or E | **UNUSED** |
| `RunManifest.environment` | No production caller supplies it; no test reads it; recovery does not use it | Plan D records environment in its separate qualification schema, not this run field | **UNUSED** |
| `RunManifest.config` | `RunContext` writes original/resolved snapshot references; force/recovery tests read them | Plan E traceability can link config snapshots after an explicit consumer is specified | Used |
| `RunManifest.inputs` | No production caller supplies it; no test reads it; recovery does not use it | Plans C/D define separate dataset/model/qualification inputs, but do not consume this field | **UNUSED** |
| `RunManifest.jobs` | Feature/infer managers append `JobRecord`; pilot asserts statuses and JobIDs | Plan D qualification has a separate command-result schema | Used |
| `RunManifest.artifacts` | `RunContext.register_verified_artifacts()` and pilot artifact assertions | Plan C consumes dataset/model artifact references | Used |
| `RunManifest.events` | Recorder appends transitions/resume/recovery/force; recovery inspects history | Plan E can audit event history after a concrete reader is defined | Used |
| `RunManifest.failure` | `fail/partial`, workflow aggregation, and failure assertions | Plans C/D require fail-closed evidence for their own outputs | Used |
| `RunEvent.state` | State transition and recovery logic; recorder/status tests | Downstream acceptance reads event state history | Used |
| `RunEvent.at` | Default is persisted, but no current reader/assertion/recovery decision uses timestamp | No concrete downstream plan consumer is specified | **UNUSED** |
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

The four unused fields are not removed in this checkpoint because their
removal changes the public manifest contract and the approved SPEC §9.1;
that change must be made as an explicit spec/plan revision before the next
expansion attempt. Keeping them while declaring GO would violate §15.9 and
the use-it-or-lose-it rule.

## 5. Recipe audit

The complete recipe diff from planning commit `72620ad` to the current
checkpoint is:

```text
M examples/recipes/README.md (7 insertions, 0 deletions)
```

The change documents `feature`/`infer` run options and the distinction between
local partial and Slurm submitted states. It changes no versioned JSON input,
scientific parameter, model path, backend default, or output convention. Task
3 separately validated 21 versioned recipe configurations after migration and
strict validation. Therefore the migration-burden check is PASS: there are
zero semantic recipe rewrites.

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
- `conda run -n dpeva-dpa4 pytest tests/unit -q` — exit `0`, `634 passed in
  24.40s`.
- `python3 scripts/doc_check.py` — exit `1` only for the pre-existing Plan A
  file `docs/reports/2026-09-04-integration-failure-classification.md`, which
  lacks YAML front matter. The newly created pilot report passes front matter,
  links, absolute-path, and owner checks.
- `git diff --check` — exit `0`.

The documentation failure is not silently treated as green; it is outside
Task 7's owned file and remains a pre-existing repository issue to be fixed by
the owning Plan A documentation follow-up.
