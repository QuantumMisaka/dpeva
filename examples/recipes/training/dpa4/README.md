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

These templates are configuration examples; three bounded DeepMD 3.2 DPA4 PT
capabilities are qualified, but this does not constitute a general scientific
or all-downstream-head compatibility claim.
Install the optional runtime extra, but lock `deepmd-kit==3.2.0` for research
production and retain `dp --version` in the run evidence. The compatibility
manifest currently declares three `supported` capabilities (17 records: 3
supported, 8 experimental, 4 unsupported, 2 blocked-upstream), backed by CPU
contract JobID `1128442` and completed SAI V100 qualification JobID `1128260`.
Do not infer scientific precision, all-head support, or broader GPU/runtime
correctness from these bounded contract results or from the version number
alone. The periodic DPA4C `pt-expt eval-desc` route remains experimental until
a genuine DPA4C artifact passes model-family preflight.

Before using a route in a research run, consult
[`DeepMD 3.2 compatibility report`](../../../reports/2026-09-04-deepmd-3.2-compatibility.md)
and `src/dpeva/compatibility/deepmd-3.2.json`. The two DPA4C non-PBC routes
remain blocked by upstream issue #6002. Future promotions require a new bounded
qualification and complete producer-issued evidence.
