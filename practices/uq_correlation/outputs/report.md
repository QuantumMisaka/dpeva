# DPA4 Mini UQ-Error Correlation Report

- Frames: 27656
- Baseline model: ensemble member 0
- QbC committee: ensemble members 1..3
- Energy error: absolute per-atom prediction error from model 0
- Force errors: model-0 frame max and RMS force errors
- Force DPOSE is not evaluated because current detached-feature implementation only supports energy-level DPOSE/LLPR.

## Correlation Summary

| uq_metric | error_metric | pearson_r | pearson_p | spearman_rho | spearman_p | kendall_tau | kendall_p | n_frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uq_qbc_for | force_error_max | 0.716211 | 0 | 0.769377 | 0 | 0.576546 | 0 | 27656 |
| uq_qbc_for | force_error_rms | 0.666213 | 0 | 0.733844 | 0 | 0.543315 | 0 | 27656 |
| uq_rnd_for | force_error_max | 0.743308 | 0 | 0.780059 | 0 | 0.587334 | 0 | 27656 |
| uq_rnd_for | force_error_rms | 0.700178 | 0 | 0.736516 | 0 | 0.545871 | 0 | 27656 |
| uq_rnd_rescaled | force_error_max | 0.743308 | 0 | 0.780059 | 0 | 0.587334 | 0 | 27656 |
| uq_rnd_rescaled | force_error_rms | 0.700178 | 0 | 0.736516 | 0 | 0.545871 | 0 | 27656 |
| uq_llpr_energy_per_atom | energy_error_per_atom_abs | 0.132951 | 2.90401e-109 | 0.125284 | 3.79501e-97 | 0.0829399 | 4.42818e-95 | 27656 |
| uq_dpose_energy_ensemble_std_per_atom | energy_error_per_atom_abs | 0.117899 | 3.56679e-86 | 0.12387 | 5.42442e-95 | 0.081619 | 3.88859e-92 | 27656 |
| uq_llpr_energy_per_atom | force_error_max | -0.109795 | 6.4518e-75 | -0.397097 | 0 | -0.265895 | 0 | 27656 |
| uq_dpose_energy_ensemble_std_per_atom | force_error_max | -0.117634 | 8.57474e-86 | -0.387566 | 0 | -0.265166 | 0 | 27656 |

## Enrichment Summary

| uq_metric | error_metric | top_fraction | high_error_fraction | n_frames | n_selected | n_high_error | n_high_error_selected | high_error_precision | high_error_recall | enrichment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uq_qbc_for | force_error_max | 0.01 | 0.05 | 27656 | 277 | 1383 | 190 | 0.685921 | 0.137383 | 13.7164 |
| uq_qbc_for | force_error_max | 0.05 | 0.05 | 27656 | 1383 | 1383 | 747 | 0.54013 | 0.54013 | 10.801 |
| uq_qbc_for | force_error_max | 0.1 | 0.05 | 27656 | 2766 | 1383 | 1100 | 0.397686 | 0.795372 | 7.95257 |
| uq_qbc_for | force_error_rms | 0.01 | 0.05 | 27656 | 277 | 1383 | 187 | 0.67509 | 0.135213 | 13.4999 |
| uq_qbc_for | force_error_rms | 0.05 | 0.05 | 27656 | 1383 | 1383 | 763 | 0.551699 | 0.551699 | 11.0324 |
| uq_qbc_for | force_error_rms | 0.1 | 0.05 | 27656 | 2766 | 1383 | 1135 | 0.41034 | 0.82068 | 8.20561 |
| uq_rnd_for | force_error_max | 0.01 | 0.05 | 27656 | 277 | 1383 | 211 | 0.761733 | 0.152567 | 15.2325 |
| uq_rnd_for | force_error_max | 0.05 | 0.05 | 27656 | 1383 | 1383 | 775 | 0.560376 | 0.560376 | 11.2059 |
| uq_rnd_for | force_error_max | 0.1 | 0.05 | 27656 | 2766 | 1383 | 1083 | 0.39154 | 0.78308 | 7.82967 |
| uq_rnd_for | force_error_rms | 0.01 | 0.05 | 27656 | 277 | 1383 | 208 | 0.750903 | 0.150398 | 15.0159 |
| uq_rnd_for | force_error_rms | 0.05 | 0.05 | 27656 | 1383 | 1383 | 769 | 0.556038 | 0.556038 | 11.1191 |
| uq_rnd_for | force_error_rms | 0.1 | 0.05 | 27656 | 2766 | 1383 | 1113 | 0.402386 | 0.804772 | 8.04656 |
| uq_rnd_rescaled | force_error_max | 0.01 | 0.05 | 27656 | 277 | 1383 | 211 | 0.761733 | 0.152567 | 15.2325 |
| uq_rnd_rescaled | force_error_max | 0.05 | 0.05 | 27656 | 1383 | 1383 | 775 | 0.560376 | 0.560376 | 11.2059 |
| uq_rnd_rescaled | force_error_max | 0.1 | 0.05 | 27656 | 2766 | 1383 | 1083 | 0.39154 | 0.78308 | 7.82967 |
| uq_rnd_rescaled | force_error_rms | 0.01 | 0.05 | 27656 | 277 | 1383 | 208 | 0.750903 | 0.150398 | 15.0159 |
| uq_rnd_rescaled | force_error_rms | 0.05 | 0.05 | 27656 | 1383 | 1383 | 769 | 0.556038 | 0.556038 | 11.1191 |
| uq_rnd_rescaled | force_error_rms | 0.1 | 0.05 | 27656 | 2766 | 1383 | 1113 | 0.402386 | 0.804772 | 8.04656 |
| uq_llpr_energy_per_atom | energy_error_per_atom_abs | 0.01 | 0.05 | 27656 | 277 | 1383 | 71 | 0.256318 | 0.0513377 | 5.12561 |
| uq_llpr_energy_per_atom | energy_error_per_atom_abs | 0.05 | 0.05 | 27656 | 1383 | 1383 | 141 | 0.101952 | 0.101952 | 2.03875 |
| uq_llpr_energy_per_atom | energy_error_per_atom_abs | 0.1 | 0.05 | 27656 | 2766 | 1383 | 206 | 0.0744758 | 0.148952 | 1.4893 |
| uq_dpose_energy_ensemble_std_per_atom | energy_error_per_atom_abs | 0.01 | 0.05 | 27656 | 277 | 1383 | 60 | 0.216606 | 0.0433839 | 4.3315 |
| uq_dpose_energy_ensemble_std_per_atom | energy_error_per_atom_abs | 0.05 | 0.05 | 27656 | 1383 | 1383 | 118 | 0.0853218 | 0.0853218 | 1.70619 |
| uq_dpose_energy_ensemble_std_per_atom | energy_error_per_atom_abs | 0.1 | 0.05 | 27656 | 2766 | 1383 | 184 | 0.0665221 | 0.133044 | 1.33025 |
| uq_llpr_energy_per_atom | force_error_max | 0.01 | 0.05 | 27656 | 277 | 1383 | 49 | 0.176895 | 0.0354302 | 3.53739 |
| uq_llpr_energy_per_atom | force_error_max | 0.05 | 0.05 | 27656 | 1383 | 1383 | 127 | 0.0918294 | 0.0918294 | 1.83632 |
| uq_llpr_energy_per_atom | force_error_max | 0.1 | 0.05 | 27656 | 2766 | 1383 | 208 | 0.0751988 | 0.150398 | 1.50376 |
| uq_dpose_energy_ensemble_std_per_atom | force_error_max | 0.01 | 0.05 | 27656 | 277 | 1383 | 48 | 0.173285 | 0.0347072 | 3.4652 |
| uq_dpose_energy_ensemble_std_per_atom | force_error_max | 0.05 | 0.05 | 27656 | 1383 | 1383 | 118 | 0.0853218 | 0.0853218 | 1.70619 |
| uq_dpose_energy_ensemble_std_per_atom | force_error_max | 0.1 | 0.05 | 27656 | 2766 | 1383 | 222 | 0.0802603 | 0.160521 | 1.60497 |

## Figures

- `figures/uq_vs_error_scatter_uq_qbc_for_vs_force_error_max.png`
- `figures/uq_vs_error_hexbin_uq_qbc_for_vs_force_error_max.png`
- `figures/uncertainty_rank_error_curve_uq_qbc_for_vs_force_error_max.png`
- `figures/uq_vs_error_scatter_uq_qbc_for_vs_force_error_rms.png`
- `figures/uq_vs_error_hexbin_uq_qbc_for_vs_force_error_rms.png`
- `figures/uncertainty_rank_error_curve_uq_qbc_for_vs_force_error_rms.png`
- `figures/uq_vs_error_scatter_uq_rnd_for_vs_force_error_max.png`
- `figures/uq_vs_error_hexbin_uq_rnd_for_vs_force_error_max.png`
- `figures/uncertainty_rank_error_curve_uq_rnd_for_vs_force_error_max.png`
- `figures/uq_vs_error_scatter_uq_rnd_for_vs_force_error_rms.png`
- `figures/uq_vs_error_hexbin_uq_rnd_for_vs_force_error_rms.png`
- `figures/uncertainty_rank_error_curve_uq_rnd_for_vs_force_error_rms.png`
- `figures/uq_vs_error_scatter_uq_rnd_rescaled_vs_force_error_max.png`
- `figures/uq_vs_error_hexbin_uq_rnd_rescaled_vs_force_error_max.png`
- `figures/uncertainty_rank_error_curve_uq_rnd_rescaled_vs_force_error_max.png`
- `figures/uq_vs_error_scatter_uq_rnd_rescaled_vs_force_error_rms.png`
- `figures/uq_vs_error_hexbin_uq_rnd_rescaled_vs_force_error_rms.png`
- `figures/uncertainty_rank_error_curve_uq_rnd_rescaled_vs_force_error_rms.png`
- `figures/uq_vs_error_scatter_uq_llpr_energy_per_atom_vs_energy_error_per_atom_abs.png`
- `figures/uq_vs_error_hexbin_uq_llpr_energy_per_atom_vs_energy_error_per_atom_abs.png`
- `figures/uncertainty_rank_error_curve_uq_llpr_energy_per_atom_vs_energy_error_per_atom_abs.png`
- `figures/uq_vs_error_scatter_uq_dpose_energy_ensemble_std_per_atom_vs_energy_error_per_atom_abs.png`
- `figures/uq_vs_error_hexbin_uq_dpose_energy_ensemble_std_per_atom_vs_energy_error_per_atom_abs.png`
- `figures/uncertainty_rank_error_curve_uq_dpose_energy_ensemble_std_per_atom_vs_energy_error_per_atom_abs.png`
- `figures/uq_vs_error_scatter_uq_llpr_energy_per_atom_vs_force_error_max.png`
- `figures/uq_vs_error_hexbin_uq_llpr_energy_per_atom_vs_force_error_max.png`
- `figures/uncertainty_rank_error_curve_uq_llpr_energy_per_atom_vs_force_error_max.png`
- `figures/uq_vs_error_scatter_uq_dpose_energy_ensemble_std_per_atom_vs_force_error_max.png`
- `figures/uq_vs_error_hexbin_uq_dpose_energy_ensemble_std_per_atom_vs_force_error_max.png`
- `figures/uncertainty_rank_error_curve_uq_dpose_energy_ensemble_std_per_atom_vs_force_error_max.png`
