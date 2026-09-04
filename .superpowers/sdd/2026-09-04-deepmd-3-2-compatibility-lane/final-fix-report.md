# Plan D final-fix report

## Scope

This bounded final-fix pass closes the independent review findings without
submitting another SAI job or promoting any capability.

## Decisions and changes

- `DeepMDAdapter.train()` authorizes `fine-tune` exactly when `finetune_path`
  is supplied; normal `train` and `fine-tune` keys cannot cross-authorize.
- PT DPA4 checkpoint-to-frozen `freeze` is an explicit experimental manifest
  capability. `candidate-evaluation` remains an upper-layer policy capability
  and is not mapped to a DeepMD CLI method.
- Capability records now use closed `required_evidence` and
  `verification_status` values. Planned/blocked entries have no command;
  implemented entries point at real pytest node IDs. Promotion is a pure
  read-only gate over repository-local JSON evidence and rejects report
  anchors, wrong keys, failed status, non-exact DeepMD versions, and missing
  SAI numeric JobID/V100 evidence.
- The 17-record distribution is explicit: 0 supported, 11 experimental, 4
  unsupported, and 2 blocked-upstream. No current record is promoted.
- Qualification preparation requires the DPA4C model environment variable,
  a regular file, and its SHA-256. The deterministic fixture directory digest
  is recorded and rechecked before submission and again in compute preflight;
  required cases are checked against one closed list at each boundary.

## Verification

- Focused compatibility/qualification tests: `60 passed, 1 skipped` (the
  existing `dpdata`-absent skip).
- Full unit suite: `818 passed, 5 warnings` (intentional deprecated facade
  warnings).
- Full integration suite: `41 passed, 7 skipped`; skips are the named
  rotation-bug fixture, GPU-only LLPR/DPOSE cases, and disabled Slurm suite.
- `ft2dp-post` local DeepMD contracts: `6 passed, 4 skipped`; skips are the
  missing PT and DPA4C model fixtures.
- Ruff, documentation audit, freshness audit, and `git diff --check`: pass.

## External boundary

The only authorized SAI attempt remains JobID `1126627`, which was cancelled
before payload execution. This pass does not resubmit it and does not claim
GPU/runtime qualification.
