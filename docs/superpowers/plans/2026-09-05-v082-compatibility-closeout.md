---
title: v0.8.2 compatibility closeout
status: completed
audience: Developers / Maintainers
last-updated: 2026-09-06
owner: Quantum Misaka
---

# v0.8.2 Compatibility Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver the approved DeepMD 3.2 compatibility and engineering-hardening work as DP-EVA 0.8.2 without silently changing existing scientific workflows.

**Spec:** none - requirements supplied directly (2026-09-05 user patch-release positioning, comprehensive review of v0.8.1..113fdfb, and explicit approval to implement all identified optimizations in SDD mode). These later user decisions supersede conflicting release/default-behavior recommendations in the earlier governance SPEC; historical evidence is retained, not rewritten as new measurements.

**Architecture:** Keep the existing adapters, recorders, evidence matrix and gate runner. Restore compatibility at public boundaries, narrow provenance to scientific runtime source, and make successful artifact attribution attempt-specific. Keep legacy defaults while new references and experimental capabilities remain explicit.

**Tech Stack:** Python 3.10+, Pydantic 2, pytest, Bash, GitHub Actions, TOML gate manifest.

## Global Constraints

- Target DP-EVA 0.8.2; DeepMD version changes do not mechanically bump DP-EVA minor versions.
- Preserve valid v0.8.1 CLI/Python usage and regular-only default inference; retain explicit failures for misspellings, conflicting inputs, unsuccessful commands and invalid scientific artifacts.
- Preserve the three approved DPA4 supported claims and their historical CPU/SAI evidence; do not promote DPA4C or require a new GPU job for unchanged evidence.
- Preserve user files on main; implement only in the isolated branch. No push, tag, publish, remote configuration changes or new SAI submissions.
- Reuse scripts/gates.toml and scripts/run_gate.py. No new approval platform or mandatory cross-family gate; independent task and final review remain required.
- Use apply_patch for edits; do not delete user data. Runtime fixes must not silently delete preexisting outputs.
- Use project dpeva-dpa4 interpreter with absolute worktree PYTHONPATH; do not rebind the shared editable install. Retain RED/GREEN raw output and exit codes in this plan's ignored SDD workspace.

## Execution Environment and Baseline

Worktree: `/home/james/work/ft2dp-dpeva/dpeva/.worktrees/compat-082`, branch `fix/compat-082`, base `113fdfb4939acecf94f37110772060623c56ee78`.

For every verification shell, first run:

```bash
export PATH="/home/james/apps/miniforge3/envs/dpeva-dpa4/bin:$PATH"
export PYTHONPATH="/home/james/work/ft2dp-dpeva/dpeva/.worktrees/compat-082/src"
```

Same-base prior review: unit+integration 925 passed / 7 named skips; isolated `test_feature_workflow_local_rejects_empty_recursion_output` fails because marker precedes validation. Root user drafts fail metadata audit and are deliberately absent from this clean worktree. No full baseline rerun is needed; verify worktree import location and the isolated failure.

## File Responsibilities and Order

1. Public Python config/helper bridges: config.py, config_migration.py, cli.py, utils/command.py, utils/env_check.py and their owning tests/docs.
2. Scientific model selection and run identity: run/model.py, run/context.py, inference/workflow owners and run tests.
3. Attempt artifact attribution and completion: run/artifacts.py, feature/inference/training managers/workflows, adapter artifact contract and owning tests.
4. Installation/runtime diagnostics: pyproject.toml, constants.py, run/doctor.py, installation docs and tests.
5. Hosted gates and qualification scope: workflows, gates.toml, qualification scripts, contract fixtures/tests.
6. Version/release documentation and packaging: version source, release_helper, Sphinx, current SPEC/index/policies and regression tests.

### Task 1: Restore Python compatibility bridges

**Files:** Modify `src/dpeva/config.py`, `src/dpeva/config_migration.py`, `src/dpeva/cli.py`, `src/dpeva/utils/command.py`, `src/dpeva/utils/env_check.py`, `scripts/fp11_1344_recover_after_false_finish.py`; extend `tests/unit/test_config_migration.py`, `tests/unit/test_cli.py`, `tests/unit/utils/test_backend_config.py`, `tests/unit/utils/test_env_check.py`, `tests/unit/scripts/test_fp11_1344_recover.py`; synchronize `docs/guides/configuration.md`, `docs/guides/cli.md`, `examples/recipes/README.md`.

**Test strategy:** Existing suites own these public boundaries; no new suite needed. Use actual model construction and actual command strings, not source-string assertions. Temporary probes: none.

**Interfaces:** Keep `migrate_legacy_config(raw)` as the single migration implementation. Restore `load_and_resolve_config(path) -> dict`; add `load_config_with_metadata(path) -> MigrationResult` for internal evidence consumers. Managers continue using instance DeepMDAdapter and never consume legacy global state.

- [x] Add RED cases for direct BaseWorkflowConfig descendants and AnalysisConfig using flat submission fields, equality/conflict semantics, input nonmutation and exploration-native backend. Add old command-builder calls, old loader dict behavior and legacy warning behavior.

```python
raw = {"data_path": "data", "backend": "slurm"}
assert InferenceConfig.model_validate(raw).submission.backend == "slurm"
assert raw == {"data_path": "data", "backend": "slurm"}
assert DPCommandBuilder.train("input.json") == "dp --pt train input.json"
DPCommandBuilder.set_backend("tf")
assert DPCommandBuilder.freeze() == "dp --tf freeze"
```

- [x] Run focused tests and retain expected old-API failures.
- [x] Normalize known legacy submission keys in the public workflow/analysis model boundary using the existing migration function, emit deprecation/migration warnings, retain `extra=forbid` for unknown keys. Do not inject submission into unrelated models.
- [x] Restore original DPCommandBuilder signatures and set_backend behavior in the deprecated facade only; use optional keyword backend overrides for stateless new use. Internal managers stay isolated instance adapters. Update new tests to exercise independent adapters rather than requiring old API removal.
- [x] Split detailed loader from old dict helper and update CLI/recovery consumers explicitly. Deprecated environment-check wrapper preserves actionable UserWarning on non-ok probe, with no import-time probing.
- [x] Run `python -m pytest tests/unit/test_config_migration.py tests/unit/test_cli.py tests/unit/utils/test_backend_config.py tests/unit/utils/test_env_check.py tests/unit/utils/test_config_paths.py tests/unit/scripts/test_fp11_1344_recover.py -q` → exit 0; synchronize boundary docs; commit `fix: preserve v0.8 Python configuration and helper APIs`.

### Task 2: Preserve ensemble defaults and narrow run identity

**Files:** Modify `src/dpeva/run/model.py`, `src/dpeva/run/context.py`, `src/dpeva/inference/managers.py`, `src/dpeva/workflows/infer.py` as needed; extend `tests/unit/run/test_model_ref.py`, `tests/unit/run/test_context.py`, `tests/unit/run/test_final_review_contract.py`, `tests/unit/inference/test_inference_io_manager.py`, `tests/unit/workflows/test_infer_workflow_exec.py`, `tests/integration/test_run_contract_pilot.py`; update `docs/guides/configuration.md` and `examples/recipes/README.md`.

**Test strategy:** Real temp directory files and temporary git repositories expose ensemble identity and provenance; no host repository mutation in tests. Existing suites own these behaviors. Temporary probes: none.

**Interfaces:** `resolve_model_refs` default returns regular-only numeric-directory models; explicit JSON refs remain the mechanism for EMA. `input_identity` retains existing result shape with collision-free stable external refs. `source_identity` retains package/git metadata but defines a versioned, scoped runtime fingerprint used for resume.

- [x] RED: regular+EMA fixtures return only regular in legacy fallback; explicit references still execute both roles. Two external same-basename models get distinct logical refs and proceed past context creation.

```python
refs = resolve_model_refs(work, family="legacy-unknown", backend="pt")
assert [ref.role.value for ref in refs] == ["regular"]
a = input_identity(first_model, "model", work)
b = input_identity(second_model, "model", work)
assert a["ref"] != b["ref"]
```

- [x] RED: nested run evidence, logs and unrelated docs/untracked data do not alter resume source identity; modifying tracked runtime Python source does. Retain tests for clean/dirty source changes and root-independent identity.
- [x] Generate external model logical refs from content identity plus basename (not absolute paths). Preserve same-input deduplication and checksum conflict detection.
- [x] Scope provenance to package runtime source and packaging/runtime configuration (`src/dpeva`, `pyproject.toml`); exclude arbitrary docs, datasets, logs and any `.dpeva` component. Compute a content fingerprint of the selected tracked runtime files (including clean content, deletions and symlinks), plus untracked Python runtime additions under package source without scanning arbitrary untracked scientific data. Record fingerprint scope. Resume comparison uses scoped fingerprint for new records, treating git commit as informational when only irrelevant docs changed; committed runtime edits still invalidate resume. Legacy unscoped records must not silently be reinterpreted as matching.
- [x] Run `python -m pytest tests/unit/run tests/unit/inference/test_inference_io_manager.py tests/unit/workflows/test_infer_workflow_exec.py tests/integration/test_run_contract_pilot.py -q` → exit 0; update docs; commit `fix: preserve ensemble defaults and scoped run provenance`.

### Task 3: Bind completion and outputs to the current attempt

**Files:** Modify `src/dpeva/run/artifacts.py`, `src/dpeva/run/context.py`, `src/dpeva/workflows/feature.py`, `src/dpeva/workflows/infer.py`, `src/dpeva/feature/managers.py`, `src/dpeva/inference/managers.py`, `src/dpeva/training/managers.py`, `src/dpeva/compatibility/adapter.py`; extend owning workflow/manager/run tests including `tests/unit/workflows/test_final_review_execution_contract.py`, `tests/unit/training/test_training_managers.py`, `tests/unit/compatibility/test_deepmd_adapter.py`, `tests/integration/test_run_contract_pilot.py`; sync `docs/guides/cli.md`, `docs/guides/configuration.md`, recipes README.

**Test strategy:** Real temp files and generated shell execution with bounded fake dp commands. Existing suites protect outputs and markers. Isolate/restore log state so missing captured logging cannot create a passing test. Temporary probes: none.

**Interfaces:** Add a shared attempt-output baseline/freshness helper in run/artifacts.py consumed by feature/infer. Existing validators remain reusable; current run may register only outputs produced/rewritten in its attempt. Keep output paths unchanged and do not remove user outputs.

- [x] RED: isolated empty Python feature test fails before fix; successful Python feature emits marker after validation, failures never do, independent of preceding logger state.
- [x] RED: preexisting nonempty output + successful no-op command must not pass as a newly produced artifact. Real rewrites (including identical bytes rewritten in a new attempt) must remain valid. Retained verified resume artifacts preserve their original provenance.
- [x] Move outer workflow marker emission after artifact validation/registration and FINISHED transition; avoid duplicate or early inner markers. Test both isolated and combined runs.
- [x] Snapshot declared existing output identities before execution using precise file metadata (device/inode, size, mtime_ns, ctime_ns), and accept only newly created or demonstrably rewritten files for new attempt attribution. This is a freshness check, not adversarial producer authentication; avoid hashing every historical large descriptor file before a run. Apply corresponding freshness checks to generated local/Slurm command guards so a shell success marker cannot certify stale files. Preserve expected-pool/per-model checks. Avoid imposing a new output directory layout.
- [x] Training guards select frozen artifact per backend: PT checkpoint file vs TF frozen_model.pb; verify JAX/PD/PT-expt backend conventions from installed upstream code or explicit local contract before choosing, and fail preflight for an unprovable contract rather than falsely requiring PT output. Keep lcurve validation only where the generated training contract produces it.
- [x] Run `python -m pytest tests/unit/workflows/test_final_review_execution_contract.py tests/unit/workflows/test_feature_workflow_submission.py tests/unit/workflows/test_infer_workflow_exec.py tests/unit/feature/test_execution_manager.py tests/unit/inference/test_inference_execution_manager.py tests/unit/training/test_training_managers.py tests/unit/compatibility/test_deepmd_adapter.py tests/unit/run tests/integration/test_run_contract_pilot.py -q` → exit 0. Run the formerly failing test alone → exit 0. Commit `fix: certify only current-attempt workflow outputs`.

### Task 4: Preserve installation compatibility and separate diagnostic lanes

**Files:** Modify `pyproject.toml`, `src/dpeva/constants.py`, `src/dpeva/run/doctor.py`, related `src/dpeva/utils/env_check.py` only if required by Task 1 wrapper; extend `tests/unit/test_dependency_contracts.py`, `tests/unit/run/test_doctor.py`, `tests/unit/utils/test_env_check.py`; synchronize README, `docs/guides/installation.md`, `docs/reference/upstream-software.md`, `docs/guides/cli.md`.

**Test strategy:** Inspect real built metadata/dependency requirements and inject version/command observations into doctor. Do not install a different runtime into the shared environment. Existing suites own the behavior.

**Interfaces:** Keep default installation supplying DeepMD through `deepmd-kit>=3.1.2,<3.3`; retain `[deepmd]` as the explicit `>=3.2,<3.3` lane. Constants/doctor distinguish retained 3.1 legacy envelope from exact 3.2 qualification. Capability matrix remains strictly the 3.2 evidence matrix, never relabel 3.1 as newly qualified.

- [x] RED dependency test expects bounded core DeepMD plus narrower optional lane; diagnostic tests distinguish 3.1.2 legacy acceptance, 3.2 acceptance, unsupported old/future releases, and prereleases not bypassing all bounds.

```python
assert "deepmd-kit>=3.1.2,<3.3" in metadata["project"]["dependencies"]
assert metadata["project"]["optional-dependencies"]["deepmd"] == ["deepmd-kit>=3.2,<3.3"]
```

- [x] Restore bounded default dependency; describe preprovisioned environment installs with explicit `--no-deps` (caller must provision all dependencies), not an allegedly dependency-free default installation.
- [x] Keep existing 3.1.2 runtime envelope without claiming new scientific verification. Doctor reports the qualified 3.2 lane separately; new-only optional surfaces cannot turn otherwise usable legacy runtime into a blanket incompatible result. Do not admit arbitrary dev versions as universally compatible.
- [x] Run `python -m pytest tests/unit/test_dependency_contracts.py tests/unit/run/test_doctor.py tests/unit/utils/test_env_check.py tests/unit/test_cli.py -q` → exit 0; commit `fix: preserve default DeepMD installation compatibility`.

### Task 5: Make CI and qualification scope executable and lightweight

**Files:** Modify `.github/workflows/python-quality.yml`, `.github/workflows/deepmd-contract.yml`, `.github/workflows/docs-deploy.yml`, `scripts/gates.toml`, qualification `scripts/validation/*deepmd_32_qualification*` and `run_recorded_command.py` as needed, `tests/contract/deepmd/conftest.py`; extend `tests/contract/deepmd/test_fixture_gate.py`, `tests/unit/scripts/test_run_gate.py`, `tests/unit/scripts/test_deepmd_32_qualification.py`; add `tests/unit/scripts/test_ci_contracts.py` to own workflow wiring tests if needed; synchronize developer guide and contract fixture/qualification docs.

**Test strategy:** Execute gate argv assembly, fixture selection and qualification case selection without GPU/network. Workflow tests parse actual YAML with available parser rather than matching whole source strings; fallback to focused structural checks only where no parser dependency exists. New CI-wiring suite is justified by a separate external orchestration boundary.

**Interfaces:** Existing gate manifest remains single command source. Explicit qualification scope `dpa4` requires only six regular/EMA cases and corresponding preflight; `all` additionally requires genuine DPA4C. Old scope-less input retains old all-case semantics. Selection must never permit missing required evidence within a scope to pass.

- [x] RED: required DPA4 CPU lane works without DPA4C fixture; explicitly selected experimental lane requires real DPA4C. Missing PT model/data always fails required supported lane.
- [x] Split supported CPU gate from experimental DPA4C gate; scope-specific fixture requirements and skipped-case rejection. CI supported lane runs automatically; experimental lane explicit/manual and fail-closed when requested. Broaden trigger paths to runtime consumers, gate and validation scripts. Invoke pytest through run_gate rather than copy commands.
- [x] Add routine integration CI job using existing integration gate; genuine Slurm/GPU tests retain named opt-in skips. Retain reused supported-evidence validation in unit/release profiles; no universal full SAI rerun.
- [x] Add explicit DPA4-only scope to preparation/submission/preflight/collection where currently all-case coupling prevents rerun; maintain old scope-less job compatibility and historical attestations. No job submissions in this task.
- [x] Gate docs deployment before write permissions are exercised: use a read-only prerequisite validation job calling shared `release` profile, then deployment job with `needs`; no deployment from an unchecked main push. Keep tag/manual behavior with the same prerequisite. Use existing profiles rather than a new governance layer.
- [x] Run `python -m pytest tests/contract/deepmd/test_fixture_gate.py tests/unit/scripts/test_run_gate.py tests/unit/scripts/test_deepmd_32_qualification.py tests/unit/scripts/test_ci_contracts.py -q` → exit 0 (omit final file only if no new suite was necessary and report exact alternative). Run `bash -n scripts/validation/run_deepmd_32_qualification.slurm` → exit 0. Commit `ci: separate supported qualification and gate deployment`.

### Task 6: Prepare 0.8.2 release identity and align current governance docs

**Files:** Modify `src/dpeva/__init__.py`, README, `docs/source/conf.py`, `scripts/release_helper.py`, `scripts/gates.toml`, `docs/guides/developer-guide.md`, relevant current `docs/guides`, `docs/policy`, `docs/reference`, `docs/superpowers/specs/2026-09-04-project-governance-and-deepmd-3-2-design.html`, its index and `docs/superpowers/plans/README.md`; update `tests/integration/test_run_contract_pilot.py`; add `tests/unit/scripts/test_release_helper.py` for the previously untested release-helper boundary; create `docs/reports/2026-09-05-v082-compatibility-closeout.md` and index entry.

**Test strategy:** Actual metadata/version extraction, build wheel+sdist and inspect archives. Release tests assert the declared version relationship rather than hardcode obsolete 0.8.1. No remote upload or tag creation.

**Interfaces:** `dpeva.__version__` is 0.8.2; Sphinx imports/derives that version rather than independent literals where feasible; release helper updates maintained version surfaces and provides a check-only path suitable for shared gate. Historical release notes/attestations retain their recorded old versions.

- [x] RED: release helper/check detects mismatched README/current guide/package version; pilot tests compare manifest version with the imported version source. Add safe explicit-version parsing tests so invalid versions cannot be written.
- [x] Set version 0.8.2; synchronize current version surfaces. Add release notes describing DeepMD3.2 exact supported operations, legacy defaults, intentional fail-closed changes and known experimental boundaries.
- [x] Amend current SPEC/index with the later user-approved patch positioning and completion state, replacing future mandatory cross-family language with optional independent-family review policy; retain historical reviews/evidence. Keep plans as historical records with a superseding pointer rather than rewriting their implementation history.
- [x] Clarify evaluation card is an evidence index, not model revalidation/ranking; do not expand it into another scientific gate. Preserve Linux non-overwriting publication scope and disclose it; no cross-platform rewrite in this release.
- [x] Run `python -m pytest tests/unit/scripts/test_release_helper.py tests/integration/test_run_contract_pilot.py -q` → exit 0, `python scripts/run_gate.py release` → exit 0 with named external-fixture skips only. Build `python -m build --no-isolation` (install build tool only if absent, no runtime replacement), inspect wheel/sdist metadata version and packaged capability JSON. Smoke the wheel in an isolated system-site-packages venv with `pip install --no-deps` and CLI help plus capability-resource loading from outside checkout; record dependency resolution as metadata validation, not a newly qualified GPU runtime.
- [x] Record raw results/revision and known remote CI readiness boundary in report, update indexes, commit `chore: prepare v0.8.2 compatibility release`.

## Completion Boundary

All six tasks require spec+quality approval and revision-bound evidence, followed by one broad final branch review and any required scoped fix re-review. Preserve user main drafts; do not merge, push, tag or publish in this execution. Return branch/worktree, commit range, actual gates, retained ledger and explicit remaining external checks.
