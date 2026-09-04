# DPA4 Training Templates

This directory provides `input.json` templates for DPA4 training in the DP-EVA examples.

## Variants

- [air/input.json](air/input.json) and [air/config_train.json](air/config_train.json)
- [neo/input.json](neo/input.json) and [neo/config_train.json](neo/config_train.json)
- [mini/input.json](mini/input.json) and [mini/config_train.json](mini/config_train.json)

## Notes

- These files are aligned with the DPA4 end-to-end test recipes under `test/dpa4-dpeva-test/e2e_dpa4_*_filter128/`.
- Each template uses `batch_size` set to `filter:128` and the same `OMAT24.hdf5` training statistics file used by the corresponding test recipe.
- Copy the desired template into your working training directory, or reference it from a matching `config_train.json`.
- The wrapper configs keep the same Slurm backend defaults used by the test recipes.

## DeepMD 3.2 compatibility boundary

These templates are configuration examples, not a DeepMD 3.2 qualification.
Install the optional runtime extra, but lock `deepmd-kit==3.2.0` for research
production and retain `dp --version` in the run evidence. The compatibility
manifest currently declares zero `supported` capabilities (17 records: 11
experimental, 4 unsupported, 2 blocked-upstream): local CPU contract
evidence is 6 passed / 4 explicit fixture skips, and the single SAI V100
attempt (JobID `1126627`) was cancelled by the scheduler before its payload
ran. Do not infer GPU/runtime correctness or promote a capability from the
version number alone.

Before using a route in a research run, consult
[`DeepMD 3.2 compatibility report`](../../../reports/2026-09-04-deepmd-3.2-compatibility.md)
and `src/dpeva/compatibility/deepmd-3.2.json`. The two DPA4C non-PBC routes
remain blocked by upstream issue #6002. A cancelled qualification may only be
retried after new explicit authorization and an operations diagnosis.
