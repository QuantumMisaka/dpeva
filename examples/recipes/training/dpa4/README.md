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
