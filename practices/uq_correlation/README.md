# DPA4 Mini UQ-Error Correlation Experiment

This directory contains a reproducible experiment wrapper for comparing
frame-level uncertainty metrics against prediction errors on the DPA4 Mini
active-learning pool.

Default data split:

- Existing training pool: `../sampled_dpdata`
- Labeled candidate pool: `../other_dpdata`
- Ensemble work directory: `work/dpa4_mini_ensemble`
- Reports and CSV outputs: `outputs/`

The experiment intentionally does not modify DP-EVA workflow semantics. It
uses `dpeva train`, `dpeva infer`, `dpeva feature`, and the existing
`UQManager` APIs, then merges their outputs into a correlation report.

## Run Order

From the repository root:

```bash
python practices/uq_correlation/scripts/preflight.py
dpeva train practices/uq_correlation/configs/train_dpa4_mini_ensemble.json
dpeva infer practices/uq_correlation/configs/infer_other_dpdata.json
dpeva feature practices/uq_correlation/configs/feature_train_last_layer.json
dpeva feature practices/uq_correlation/configs/feature_candidate_last_layer.json
python practices/uq_correlation/scripts/extract_candidate_energy.py
python practices/uq_correlation/scripts/run_qbc_force_uq.py
python practices/uq_correlation/scripts/resolve_last_layer_weights.py
python practices/uq_correlation/scripts/run_llpr_energy_uq.py
python practices/uq_correlation/scripts/uq_correlation_report.py --config practices/uq_correlation/configs/report_uq_error_correlation.json
```

`DPA4-Mini-OMat24.pt` must be available at the path configured in
`configs/train_dpa4_mini_ensemble.json`. Update `model_head` if your base
model requires a non-default head.

## Main Outputs

- `outputs/preflight_summary.json`
- `outputs/qbc_force_frame_table.csv`
- `outputs/llpr_energy_uq.csv`
- `outputs/uq_error_frame_table.csv`
- `outputs/correlation_summary.csv`
- `outputs/enrichment_summary.csv`
- `outputs/report.md`
- `outputs/figures/*.png`

Force-level DPOSE is not evaluated here. The current DP-EVA detached
last-layer feature implementation supports energy-level LLPR/DPOSE only.
