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

- Focused compatibility/qualification tests: `62 passed` (compatibility and
  qualification unit scopes; no skips).
- `tests/unit/scripts/test_deepmd_32_qualification.py`: `22 passed`.
- Full unit suite: `820 passed, 5 warnings` (intentional deprecated facade
  warnings).
- `ft2dp-post` local DeepMD contracts: `6 passed, 4 skipped`; skips are the
  missing PT and DPA4C model fixtures.
- Full integration suite: `41 passed, 7 skipped`; skips are the named
  rotation-bug fixture, GPU-only LLPR/DPOSE cases, and disabled Slurm suite.
- `ft2dp-post` local DeepMD contracts: `6 passed, 4 skipped`; skips are the
  missing PT and DPA4C model fixtures.
- Ruff, documentation audit, freshness audit, and `git diff --check`: pass.

## External boundary

The only authorized SAI attempt remains JobID `1126627`, which was cancelled
before payload execution. This pass does not resubmit it and does not claim
GPU/runtime qualification.

## Final-fix round 2

- Added the strict `CapabilityAttestation` schema and changed promotion to
  consume only producer-issued attestations, either as one JSON file or inside
  a validated qualification aggregate. Documentation anchors and raw command
  records cannot satisfy the gate.
- CPU contract tests now issue attestations only after their real command,
  artifact, and shape assertions complete; CI uploads the resulting
  `attestations/` directory.
- SAI preparation derives immutable attestation specs from the manifest;
  submit and compute preflight reject stale mappings. The collector emits
  finished attestations only from a complete validated qualification and
  includes numeric JobID, V100, and exact-version evidence.
- PT train/fine-tune/test and periodic DPA4C eval-desc require both CPU and SAI
  evidence. Test/eval/embed SAI case mappings cover regular and EMA cases;
  planned routes have null cases and cannot be promoted. Candidate evaluation
  is explicitly planned policy-only and has no CLI verification command.
