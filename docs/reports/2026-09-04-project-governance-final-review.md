---
title: Project Governance and DeepMD 3.2 Final Review Preparation
status: ready-for-independent-review
audience: Project Maintainers / Compatibility Owner / Scientific Owner
last-updated: 2026-09-05
owner: Project Maintainer
---

# Project Governance and DeepMD 3.2 Final Review Preparation

## Disposition

This report records the implementation evidence for Plans A–E and prepares the
mandatory independent governance review. It is not the independent review and
does not grant final governance approval. The review launcher, reviewer family,
reviewed diff, findings, and final disposition must be appended by the primary
maintainer immediately before merge consideration.

The implementation remains deliberately thin: one executable gate manifest,
one runner, one path/schema traceability registry, and one report-only
use-it-or-lose-it audit. No campaign orchestrator, issue bot, automatic policy
editor, or second scientific source of truth was introduced.

## First independent cross-family review round

The first frozen review package was valid and is recorded here as review input,
not as final approval:

- Package ID: `20260905T000815Z-governance-cf5045a1`
- Frozen package range: `b637ca0..0b9700a`
- Reviewer launcher result: exit `0`; reviewer verdict: `REQUEST_CHANGES`
- Reviewer identity: `opencode-qwen-scnet` / family `qwen` / model
  `Qwen3.8-Max`

The accepted findings were the gate/profile name collision and the stale
`docs/reference/upstream-software.md` date. The gate/profile collision was
fixed in `9353752` (`fix: disambiguate documentation gate names`), and the
upstream reference date is corrected in this documentation closure. The scoped
same-family task review of the accepted fix was approved. The reviewer's
cosmetic observations—CLI section ordering and the duplicate SDD report line—
are explicitly deferred as non-contractual and are not used as governance-gate
evidence.

This round does not close the mandatory governance gate. Final disposition
remains **PENDING a new cross-family review package**; neither the launcher exit
code nor the `REQUEST_CHANGES` package may be relabeled as a passed gate.

## Scope and evidence boundary

The reviewed implementation is the isolated `feat/governance-deepmd-32`
worktree at the following Plan E checkpoints:

| Area | Evidence | Boundary |
|---|---|---|
| Plan A | `docs/reports/2026-09-04-integration-failure-classification.md` and R1/R2 regression tests | execution failure propagation and integration classification; not universal cluster validity |
| Plan B | `docs/reports/2026-09-04-run-contract-pilot-report.md` | feature/infer run-contract pilot and doctor boundary; not all workflows |
| Plan C | `docs/reports/2026-09-04-dataset-lineage-eval-card-acceptance.md` and `docs/reports/2026-09-05-plan-c-final-fix-report.md` | lineage/evaluation evidence plumbing; not scientific superiority |
| Plan D | `docs/reports/2026-09-04-deepmd-3.2-compatibility.md` | exact DeepMD contract and qualification evidence; not a support claim |
| Plan E | `scripts/gates.toml`, `docs/governance/traceability/capability-evidence.json`, `docs/governance/rules.json` | command/path metadata and governance reporting; not evidence that a command has run remotely |

Green repository checks prove only the layer they execute. Scientific,
upstream, GPU, and scheduler conclusions remain bounded by their cited reports.

## Implementation summary

### Reliable execution and run contract

- Local and Slurm execution paths fail closed and aggregate child/process
  outcomes; completion markers are compatibility output, not a success proof.
- Phase 0 integration failures were classified and the same-runtime local
  multidatapool path was rerun successfully. The final repository baseline
  recorded 41 integration passes and 7 explicit capability/environment skips.
- The feature/infer pilot recorded a GO decision after 152 focused tests, a
  median manifest overhead of 9.113 ms in the declared fake-command benchmark,
  and a closed schema-field audit. Its scope remains limited to feature/infer.
- The run contract requires process/job success, verified declared artifacts,
  and a `finished` run manifest; `failed`, `partial`, and `submitted` remain
  explicit evidence states.

### Dataset lineage and evaluation card

- Dataset manifests persist parent/source, counts, type-map, intersection and
  validation evidence; the 12,105 + 4,317 = 16,422 frame conservation boundary
  is covered by regression tests.
- Model references distinguish checkpoint/frozen/exportable/pretrained-alias
  and regular/EMA roles. Evaluation cards expose six explicit metric dimensions
  and retain `not-run`/`failed` evidence rather than inventing zero values.
- Candidate references are portable relative references with checksum and
  validation semantics. Bundle publication guarantees process-visible atomic
  no-overwrite behavior, not crash-durable recursive dpdata-tree persistence.

### DeepMD-kit 3.2 compatibility lane

The capability manifest currently contains 17 records:

| State | Count |
|---|---:|
| `supported` | 0 |
| `experimental` | 11 |
| `unsupported` | 4 |
| `blocked-upstream` | 2 |

The local exact-3.2.0 contract lane recorded 6 passed and 4 explicitly named
fixture skips. The skips are not positive fixture qualification and do not
promote a capability.

The single authorized SAI qualification attempt was JobID `1126627` in the
recorded immutable external directory. The scheduler result was
`CANCELLED by 0` after one second on `4v100n03`, before the payload produced
stdout/stderr, preflight, environment, command, or artifact records. The
collector therefore recorded `status=failed`; `--require-complete` failed as
designed. No capability is promoted, and no second job is submitted under this
plan. A retry requires new authorization and an operations diagnosis.

### Thin governance alignment

- `scripts/gates.toml` is the sole executable command catalog; `scripts/run_gate.py`
  runs argv without shell interpolation, from a stable repository root, and
  stops at the first non-zero result. Local and hosted entry points delegate to
  named gates while preserving job isolation, installation, caching, artifacts,
  and PR-only link checking.
- The traceability registry contains ten public capability records. Validation
  checks exact schema, duplicate IDs, repository-contained existing paths,
  owners, and evidence pointers; it never searches source text to infer
  behavior or qualification. The `traceability` gate is attached to docs and
  ordinary release profiles, not local/unit/PR or DeepMD qualification profiles.
- The governance registry contains three active mechanisms and is capped at
  eight. Each record has an owner, basis, enforcement paths, repository-local
  `trigger_paths`, review date, and interval. The quarterly workflow is
  report-only and cannot edit, delete, commit, or create issues. `--strict` is
  reserved for an explicit release review.
- Python 3.10 hosted paths install `tomli`; Python 3.11+ uses `tomllib`.

## Verification record

The documentation preparation phase updated the stable guides, policies,
governance overview, and the local normative SPEC to point at the manifest and
to record the current DeepMD evidence. Raw duplicate gate argv was removed from
the active documentation. The ordinary release profile was run at pre-review
commit `0b9700a` from the isolated worktree:

```text
conda run -n dpeva-dpa4 python scripts/run_gate.py release
```

Result at `0b9700a`: **exit 0**. The profile recorded Ruff pass, 862 unit tests passed
(83.64% total coverage; five expected deprecation warnings), the audit pass,
41 integration tests passed with 7 explicit capability/environment skips,
documentation audit and freshness pass, Sphinx warning-as-error HTML build
pass, linkcheck pass, and traceability pass. The first attempt stopped at the
Sphinx build because two existing report links were interpreted as source
documents; those links were changed to repository-path references and the
complete profile was rerun successfully.

This is historical pre-review evidence. Commit `9353752` subsequently changed
gate names and profile resolution to address the accepted review finding, and
this report is now changing again; therefore no fresh full release result is
claimed for the current post-fix documentation head. The scoped same-family
task review approved the fix, while the new cross-family review remains the
required next gate.

The documentation preparation also ran:

```text
git diff --check
```

Result: **exit 0; no output**.

The `deepmd_release` profile is intentionally not a passing requirement for
this ordinary software/documentation release: the matrix has zero supported
records and the SAI qualification is failed before payload. If probed, its
non-zero result is expected negative evidence, not a release defect and not a
support claim.

## Mandatory independent review — pending

This section must be completed by the primary maintainer immediately before
merge consideration. It must use the current repository governance launcher
and must not be replaced by author self-review.

- Governance instructions read: **PENDING**
- Launcher command: **PENDING**
- Reviewed commit/diff: **PENDING**
- Reviewer family: **PENDING**
- Exit status: **PENDING**
- Findings and accepted fixes: **PENDING**
- Cross-family attempt: **PENDING**
- Final disposition: **PENDING — do not label this report approved**

If the cross-family backend is unavailable, the final disposition must state
`cross-family review incomplete; follow-up required`. A successful local gate or
same-family review cannot be relabeled as cross-family approval.

## Remaining uncertainty and next action

The next authorized action is the independent governance review, followed by
affected-gate reruns for any accepted documentation or mechanism findings.
DeepMD production support remains unclaimed until a newly authorized complete
SAI qualification and matching producer attestations exist. Phase 3 scientific
expansion and automatic campaign orchestration remain outside this closure.
