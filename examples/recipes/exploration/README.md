# Exploration Recipes

`dpeva explore` is a thin DP-EVA wrapper around optional trajectory exploration
backends. The default recipe uses MD because active-learning exploration usually
generates candidate configurations by running molecular dynamics with an ML
potential. The first backend is `atst-tools`, installed with:

```bash
python -m pip install -e '.[explore]'
```

The DP-EVA JSON config records workflow metadata and result collection paths.
The `backend_config_path` points to a native ATST YAML file. DP-EVA does not
rewrite or generate that YAML in v0.8.0.

```bash
cd examples/recipes/exploration
dpeva explore config_explore_md.json
```

The backend writes `dpeva_exploration_result.json` in `work_dir`, along with
any DP-EVA-side input structure snapshots under `work_dir/dpeva_inputs/`.
