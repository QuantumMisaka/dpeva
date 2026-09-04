# Plan C Task 4 report

## Scope

Added the closed schema and deterministic assembler for one candidate
evaluation card. The assembler indexes evidence and does not execute metrics,
rank candidates, or copy active FT2DP task state.

## TDD evidence

- RED: `pytest tests/unit/evaluation/test_card.py -q` initially failed during
  collection because `EvaluationCardConfig` and the evaluation package were
  absent.
- GREEN: the focused suite passes with `10 passed`.

## Implementation decisions

- `EvaluationCard` requires exactly six dimensions:
  `in_domain_cumulative`, `iter11_last_wave`, `historical_domain`,
  `matpes_retention`, `training_cost`, and `surface_slice`.
- Missing optional metric paths produce `not-run` with `value=None`; malformed
  or missing configured metric files produce `failed`, retain the normalized
  evidence path, and do not abort assembly of other dimensions.
- Metric JSON is a closed object containing `status` and optional `value` and
  `detail`. Unknown fields are rejected as malformed evidence.
- `model_ref_path` and every `dataset_manifest_paths` entry must be an
  existing regular JSON file with a valid strict schema-1.0 reference. Dataset
  lineage count validation runs before the card is assembled.
- The card stores normalized immutable evidence paths. It does not require the
  referenced model binary or dataset source entries to remain online: those
  checks belong to execution/producer boundaries and would reject valid
  external or alias evidence at this handoff layer.

## Verification

- `pytest tests/unit/evaluation/test_card.py -q` — `10 passed`.
- `pytest tests/unit/test_config_migration.py tests/unit/run/test_dataset_lineage.py tests/unit/run/test_model_ref.py -q` — `37 passed`.
- `python -m compileall -q src/dpeva/evaluation` — passed.
- `ruff` was not available in the active interpreter (`ruff: command not found`; `python -m ruff` also unavailable). Parent-agent environment verification remains required.
- `git diff --check` — passed.

## Changed files

- `src/dpeva/config.py`
- `src/dpeva/evaluation/__init__.py`
- `src/dpeva/evaluation/card.py`
- `tests/unit/evaluation/test_card.py`

## Remaining boundary

Task 5 must add config-path resolution for the new singular metric paths and
dataset manifest list, then expose the assembler through the CLI. The output
file is intentionally written by that CLI boundary rather than by this pure
assembler.
