---
title: v0.8.2 Compatibility Closeout
status: release-candidate
audience: Project Maintainer / Compatibility Owner / Developers
last-updated: 2026-09-06
owner: Project Maintainer
---

# v0.8.2 Compatibility Closeout

## Decision

v0.8.2 is a compatibility and reliable-run patch closeout. It preserves the
public Python facade and legacy configuration/read boundaries while making
ambiguous execution, stale-output attribution, version drift, and unsupported
capability promotion fail closed. It does not add a scientific qualification,
tag, upload, publication, remote configuration change, or SAI job.

The release candidate retains exactly three `supported` DeepMD records: DPA4
PT `test`, `eval-desc`, and `embed`. Their existing SAI aggregate contains six
historical DPA4 attestations (regular/EMA for each operation). DPA4C periodic
`pt-expt eval-desc` remains `experimental`; its earlier command completion did
not prove DPA4C artifact identity. The remaining manifest distribution stays
3 supported, 8 experimental, 4 unsupported, and 2 blocked-upstream.

## Compatibility and intentional fail-closed changes

- Default installation remains `deepmd-kit>=3.1.2,<3.3`; the explicit
  `dpeva[deepmd]` lane remains `>=3.2,<3.3`, and production qualification
  records remain pinned to exact `3.2.0`. A dependency range is not a claim
  that every included runtime is behaviorally identical.
- Legacy top-level submission fields are migrated at the public config-model
  boundary; conflicts and unknown fields fail instead of being silently
  discarded. Public aliases/facades and loader dictionary access remain
  available.
- Legacy numeric model discovery remains regular-only by default. EMA is
  included only through an explicit model reference and role.
- Feature/inference success requires current-attempt output freshness and
  verified artifacts. A stale file or leaked child completion marker cannot
  promote a parent run to `finished`; partial and failed states remain visible.
- Run provenance uses the scoped runtime fingerprint defined by the run
  contract. Documentation-only changes do not masquerade as runtime changes.

## Evaluation and publication boundary

The evaluation card is an evidence index. It assembles existing model,
dataset-lineage, metric, and downstream references; it does not rerun a model,
revalidate scientific accuracy, recreate an external experiment matrix, or
rank candidates. Missing or invalid evidence remains explicit in the card.

Dataset bundle publication retains the Linux-only
`renameat2(RENAME_NOREPLACE)` contract: sibling staging is atomically published
without overwriting a competing target in process-visible scope. Unsupported
platforms fail closed; v0.8.2 does not add a cross-platform rename fallback or
claim recursive fsync/crash durability for the bundle.

## Release identity and package boundary

`dpeva.__version__`, the README badge, and the current developer-guide version
are synchronized at `0.8.2`. Sphinx imports the package version rather than
maintaining an independent literal. `scripts/release_helper.py --check` is a
read-only shared release gate; explicit update targets accept only plain
`X.Y.Z` versions and validate every maintained replacement before writing.

The wheel/sdist acceptance inspects archive metadata and the packaged
`dpeva/compatibility/deepmd-3.2.json` resource. The isolated wheel smoke uses a
system-site-packages venv and `pip install --no-deps`; this proves packaging,
entry-point help, version import, and capability-resource loading outside the
checkout. It is dependency metadata validation, not a newly qualified GPU or
scientific runtime.

## Verification and retained raw evidence

Raw command output is retained locally under
`.superpowers/sdd/2026-09-05-v082-compatibility-closeout/task-6-logs/` and is
intentionally ignored rather than forced into release history. Each final
record begins with `git rev-parse HEAD`, a clean-worktree observation, the
exact command, and ends with its exit status, so the output is bound to the
release-candidate commit that contains this report.

The retained records cover:

1. RED: the new release-helper suite failed because check-only, guide sync,
   strict explicit-version parsing, and the shared release gate were absent.
2. GREEN: the focused release-helper and run-contract pilot suite.
3. The complete shared `release` profile, with only named external-fixture
   skips accepted.
4. `python -m build --no-isolation` in an isolated build-tool venv, wheel/sdist
   metadata and capability JSON inspection, and outside-checkout wheel smoke.

At the final source revision `b4a29f1`, the focused boundary passed with 38
tests. The corrected shared `release` profile passed with 47 tests and seven
named external-fixture skips. The first release record failed before this
correction because its ignored wrapper replaced `PATH` with a minimal list and
hid the WSL CUDA driver directory; the retained DeepMD stderr identified
`libcuda.so` loading as the first failing operation. Prepending the project
environment to the existing quoted `PATH` restored driver discovery and made
the same release profile pass without a production change.

The exact final record names are `final-focused.typescript`,
`final-release-corrected.typescript`, and `package-corrected-v2.typescript`.
The earlier failed `final-release.typescript`, diagnostic integration
reproduction, and first package-inspection attempt remain retained as
historical evidence rather than being overwritten.

The raw logs, not a copied terminal excerpt in this report, are the
authoritative results and revision binding. The implementation task report in
the sibling SDD directory records the concise observed counts and artifact
names for controller review.

## Remaining external checks

This local closeout does not claim that hosted GitHub Actions ran. The workflow
configuration is release-profile ready, but remote runner behavior remains a
post-push observation. Independent Task 6 review and broad whole-branch review
also remain required before integration. Independent-family/cross-model review
is optional under the current global policy; historical cross-family reviews
retain their original frozen-diff scope and are not reused as v0.8.2 approval.

No merge, push, tag, package upload, SAI submission, or release publication is
performed by this closeout.
