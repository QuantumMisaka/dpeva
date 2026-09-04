# Plan C Task 5 Report

## Result

Implemented `dpeva eval-card CONFIG.json` and the portable candidate-package
recipe. The command validates one strict `EvaluationCardConfig`, assembles the
Task 4 card, and publishes it as immutable JSON evidence.

## Contract decisions

- `output_path`, `model_ref_path`, all six metric paths, dataset-manifest lists,
  and local `downstream_feedback_ref` values resolve relative to the config
  file. Explicit URI references (`https://`, `s3://`, `doi:`, etc.) remain
  unchanged and are treated as opaque evidence identifiers.
- Publication is POSIX fail-closed: the parent directory is fsync-preflighted,
  the temporary JSON file is fsynced, and a hard-link no-replace publication
  closes the concurrent-writer race. A post-link fsync failure raises
  `EvaluationCardPublicationError(published=True)` and preserves the target for
  inspection; retries therefore use a new output path.
- Missing metric evidence remains visible through the Task 4 `not-run`/`failed`
  statuses. The recipe does not copy live FT2DP task status.

## Verification

- Focused Task 5, config-path, and inference warning tests: **18 passed**.
- Acceptance schema/CLI set: **47 passed** before the fix round.
- CLI/unit combination: **66 passed** before the fix round.
- Full unit after the fix round: **735 passed**; the logger-isolation failure
  is fixed in this round.
- Project environment Ruff checks for changed source/tests: passed.
- `git diff --check`: passed before this report was added; rerun before handoff.

## Commit

The implementation baseline is `9c22d1ef43b8c24687585b8341fea13db08a902d`;
this fix round is committed separately after verification.
