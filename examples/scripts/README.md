# DP-EVA Example Scripts

This directory contains executable helper scripts for recipe configurations.

## Layout

- `training/train_recipe.py`
- `inference/infer_recipe.py`
- `inference/check_model_force_stats.py`
- `feature_generation/feature_recipe.py`
- `collection/collect_recipe.py`
- `analysis/analysis_recipe.py`
- `labeling/run_labeling.sh`
- `sampling_direct/run_direct.py`
- `sampling_2direct/run_2direct.py`

## Notes

- Recipe configuration files remain under `examples/recipes/**/config*.json`.
- Scripts in this directory default to recipe config locations and accept custom config paths where applicable.
