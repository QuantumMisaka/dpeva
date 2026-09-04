# Plan C Task 1 report

## Scope

Defined the versioned dataset lineage schema and pure frame-count validator.
The implementation is intentionally limited to the reusable `dpeva.run`
boundary; workflow integration and evaluation-card assembly remain in later
tasks.

## TDD evidence

- RED: `pytest tests/unit/run/test_dataset_lineage.py -q` failed during
  collection with `ModuleNotFoundError: No module named 'dpeva.run.dataset'`.
- GREEN: the same command passed with `10 passed`.

## Implementation decisions

- `DatasetParent` and `DatasetManifest` use Pydantic v2 with
  `extra="forbid"`, explicit schema version `1.0`, and non-negative counts.
- Derived transformations require at least one parent. An `import` manifest
  may be a parentless root, whose observed frame count is authoritative until
  a later transformation records parent evidence.
- Duplicate parent dataset references and duplicate type-map entries are
  rejected at schema construction. Count reconciliation remains a separate
  pure operation so malformed observed totals raise `LineageValidationError`.
- The invariant is `sum(parent.frame_count) - removed_frame_count ==
  manifest.frame_count`. The required `12105 + 4317 = 16422` case, removals,
  omission, over-removal, parentless import, negative counts, duplicate
  references/type-map entries, and unknown fields are covered.

## Verification

- `conda run -n dpeva-dpa4 pytest tests/unit/run -q` — `139 passed`.
- `conda run -n dpeva-dpa4 pytest tests/unit -q` — `678 passed`.
- `conda run -n dpeva-dpa4 ruff check src/dpeva/run/dataset.py src/dpeva/run/__init__.py tests/unit/run/test_dataset_lineage.py` — passed.
- `git diff --check` — passed.

## Changed files

- `src/dpeva/run/dataset.py`
- `src/dpeva/run/__init__.py`
- `tests/unit/run/test_dataset_lineage.py`

## Remaining boundary

No frame-level identity algorithm is introduced here. Later integration code
must supply the observed counts and use this validator before handoff.
