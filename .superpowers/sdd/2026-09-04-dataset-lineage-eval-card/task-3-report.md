# Task 3 report: explicit model artifact references

Implemented the model-reference contract and inference compatibility bridge.

## Decisions

- `ModelArtifactRef` is a closed Pydantic schema (`extra="forbid"`) with
  explicit checkpoint, frozen, exportable, and pretrained-alias kinds, regular
  and EMA roles, producer/runtime metadata, checksum, and declared operations.
- String fields use strict validation while enum values remain ergonomic JSON
  strings. A pretrained alias without `resolved_path` fails before execution;
  an undeclared operation fails closed for every artifact kind.
- Legacy discovery scans all numeric directories in integer order and checks
  both `model.ckpt.pt` and `model_ema.ckpt.pt`; gaps do not terminate the scan.
  Discovered references are `family="legacy-unknown"`, carry SHA-256, and
  declare only `test`, preserving the no-new-directory-convention boundary.
- `InferenceConfig.model_ref_paths` is authoritative when non-empty. Empty
  configuration emits one migration warning and uses the legacy bridge. Paths
  are converted from validated references only at the manager execution
  boundary.

## Verification

- `conda run -n dpeva-dpa4 pytest tests/unit/run/test_model_ref.py tests/unit/inference/test_inference_io_manager.py -q`
  — 11 passed.
- `conda run -n dpeva-dpa4 pytest tests/unit/inference/test_inference_execution_manager.py tests/unit/workflows/test_infer_workflow_exec.py -q`
  — 13 passed.
- Ruff check for all changed Python files — passed.
- `git diff --check` — passed.

## Deviation / uncertainty

No new artifact directory convention or network alias resolver was added. Alias
resolution is deliberately a producer/runtime concern and remains fail-closed
until a resolved local path is present. The broader unit and integration suites
remain parent-agent verification responsibilities.
