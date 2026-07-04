---
title: Dataset Audit Design
status: archived
audience: Historians / Developers / Maintainers / AI Agents
last-updated: 2026-07-04
owner: Docs Owner
---

# Dataset Audit Design

## Goal

Add a dataset and post-collect auditing capability to DP-EVA that helps users evaluate training sets, validation sets, sampled collect outputs, and sampled-vs-reference coverage before spending labeling or retraining budget.

The user-facing entry point is a short `dpeva audit <config.json>` command. Internally it reuses the existing analysis workflow stack so that `dpeva analysis <config.json>` with `mode="audit"` remains equivalent.

## Context

DP-EVA already has:

- `dpeva analysis` with `mode="model_test"` and `mode="dataset"`.
- `src/dpeva/analysis/dataset.py`, which computes dataset physical statistics such as energy, force, virial, pressure, cohesive energy, and element ratios.
- `src/dpeva/workflows/collect.py`, which writes `dataframe/final_df.csv`, collection logs, pool-level sampled statistics, and exported `dpdata`.
- `src/dpeva/io/collection.py`, which loads DPA descriptor features and converts atomic descriptors into normalized structural descriptors with `desc_stru_*` columns.
- Iter10 practice scripts that already proved the value of pool entropy, descriptor nearest-neighbor novelty, internal nearest-neighbor redundancy, composition coverage, UQ/filter log summaries, and exact frame overlap.

The new capability should turn those practice scripts into reusable DP-EVA functionality without forcing collect to run additional expensive diagnostics by default.

## Design Decision

Implement option 3 from the brainstorming phase:

- Keep analysis as the underlying module family.
- Add `mode="audit"` to `AnalysisConfig`.
- Add `dpeva audit config.json` as a CLI alias that loads the same config and dispatches to the same workflow path.
- Keep audit metrics post-hoc and diagnostic. They must not silently change collect selection behavior.

This preserves existing analysis architecture while making the command name clearer for users who want dataset inspection rather than model-test error analysis.

## Scope

The first implementation should support three use cases:

1. Audit a standalone training or validation dataset.
2. Audit one collect output relative to a reference training dataset.
3. Compare several collect outputs against the same reference and against each other.

The first implementation should not try to implement online active learning, automatic relabeling decisions, or automatic replacement of collect sampling criteria.

## Configuration Contract

Example:

```json
{
  "mode": "audit",
  "output_dir": "audit",
  "audit_reference": {
    "name": "train",
    "dataset_dir": "./sampled_dpdata",
    "desc_dir": "./desc_train_dpa4_air_embed",
    "desc_feature_kind": "descriptor"
  },
  "audit_targets": [
    {
      "name": "joint_r050_w025_15000",
      "collect_dir": "./dpa4-air-collect-joint-r050-w025-15000",
      "dataset_dir": "./dpa4-air-collect-joint-r050-w025-15000/dpdata/sampled_dpdata"
    }
  ],
  "audit_metrics": {
    "basic": true,
    "descriptor_nn": true,
    "composition": true,
    "overlap": true,
    "entropy": false,
    "advanced_entropy": false
  }
}
```

Required config behavior:

- `audit_reference` is optional for standalone dataset audits.
- `audit_targets` must contain at least one entry.
- Each target must provide either `dataset_dir`, `collect_dir`, `final_df_path`, or `desc_dir`.
- `collect_dir` implies default paths:
  - `collect_dir/dataframe/final_df.csv`
  - `collect_dir/collection.log`
  - `collect_dir/dpdata/sampled_dpdata`
- `desc_feature_kind` defaults to `descriptor`.
- `audit_metrics.basic` defaults to `true`.
- Expensive entropy metrics default to `false`.

## Metrics

### Basic Metrics

Always cheap and safe:

- selected/exported frame counts
- unique dataname count
- unique system count
- element categories
- atom count distribution
- pool counts
- normalized pool entropy
- optional UQ threshold and filter counts parsed from `collection.log`

Pool entropy is normalized Shannon entropy over pool-level sampled frame proportions. It measures source-pool balance, not full chemical diversity.

### Descriptor Metrics

Computed when descriptors are available:

- `nn_to_reference_*`: nearest-neighbor distance from each target frame to the reference descriptor set.
- `selected_internal_nn_*`: distance to the second nearest neighbor inside the target set.
- descriptor dimensions and loaded frame counts.
- optional distance histograms in a later plotting task.

The default distance is Euclidean in normalized structural descriptor space because this matches existing DP-EVA collect descriptor usage and the iter10 practice scripts.

### Composition Metrics

Computed from `type.raw` and `type_map.raw` through existing dpdata-compatible loaders:

- unique formulas
- frame fraction with formulas unseen in the reference
- composition-vector nearest-neighbor L1 distance to reference
- per-element atom ratios
- per-element frame/system presence

These are chemical-composition proxies. They do not distinguish geometry or adsorption motifs.

### Overlap Metrics

Computed across targets:

- exact frame overlap by `dataname`
- Jaccard index
- left/right overlap fraction
- left/right unique frame counts

The first implementation uses exact identity only. Structural RMSD or graph matching is out of scope.

### Entropy Metrics

Two entropy tiers are allowed:

1. Lightweight log-det feature entropy:
   - Compute covariance of descriptor vectors after optional PCA.
   - Report `slogdet(cov + ridge * I)`.
   - Report marginal gain from appending target descriptors to reference descriptors.
   - Default ridge is `1e-6`.

2. Advanced QUESTS-style entropy:
   - Compute KDE entropy and differential entropy in descriptor space.
   - Default disabled.
   - Requires explicit bandwidth configuration.
   - Should operate on a frame or atom sample cap to avoid accidental O(N^2) work.

The MVP implements the log-det metric and leaves QUESTS-style KDE behind the same interface with a clear not-enabled error if requested before full implementation.

## Literature Basis

The design draws from:

- Schwalbe-Koda et al., "Model-free estimation of completeness, uncertainties, and outliers in atomistic machine learning using information theory", Nature Communications 2025. This motivates atom-centered descriptor entropy, differential entropy, dataset completeness, novelty/overlap, and outlier detection.
- Local paper text at `test/ML-dataaset-entropy-sampling-pdf/full.md`, which provides the same QUESTS details in repository-local form.
- Wang, Rao, and Zhu, "Dataset-aware entropy-maximized active learning for machine-learned interatomic potentials", arXiv 2605.20384. This motivates global log-determinant dataset-aware filtering and treating entropy gain as a compactness/coverage criterion.
- Subramanyam and Perez, "Information-entropy-driven generation of material-agnostic datasets for machine-learning interatomic potentials", npj Computational Materials 2025. This motivates feature covariance entropy, broad feature-space coverage, and material-agnostic dataset generation.
- MAD and related atomistic dataset papers that use descriptor/latent-space projections, farthest-point landmarks, and coverage histograms to explain dataset representativeness.

## Architecture

New implementation units:

- `src/dpeva/analysis/audit.py`
  - Pure metric functions and small dataclasses.
  - No filesystem side effects except optional helpers that read logs/dataframes.
- `src/dpeva/analysis/audit_manager.py`
  - Loads reference/target artifacts.
  - Calls pure metrics.
  - Writes CSV, JSON, and Markdown report outputs.
- `src/dpeva/workflows/analysis.py`
  - Adds `mode="audit"` routing.
- `src/dpeva/cli.py`
  - Adds `audit` parser and handler.
- `src/dpeva/config.py`
  - Adds audit config models and expands `AnalysisConfig.mode`.
- `examples/recipes/analysis/config_audit.json`
  - Minimal working recipe.

The manager may reuse:

- `dpeva.io.dataset.load_systems`
- `dpeva.io.collection.CollectionIOManager.load_descriptors`
- `dpeva.constants.COL_DESC_PREFIX`

## Output Contract

`output_dir` should contain:

- `audit_metrics.csv`: one row per target.
- `audit_metrics.json`: structured metric payload.
- `audit_pool_counts.csv`: target-by-pool frame counts.
- `audit_overlap.csv`: pairwise overlap among targets when there are at least two targets.
- `audit_report.md`: human-readable explanation, metric tables, caveats, and artifact index.
- `analysis.log`: standard workflow log.

All outputs should be deterministic for fixed inputs.

## Failure Behavior

Failures should be explicit and local:

- Missing target artifacts should name the missing path and target name.
- Missing descriptors should skip descriptor metrics only if `descriptor_nn=false`; otherwise fail.
- Missing dataset composition should skip composition metrics only if `composition=false`; otherwise fail.
- `entropy=true` without descriptor data should fail.
- Advanced KDE entropy should require explicit `advanced_entropy=true` and `entropy_bandwidth`.

Plotting, when added later, should follow the existing analysis rule: plotting failures are warnings, metric computation failures are errors.

## Testing Strategy

Unit tests should cover:

- Pool name parsing and normalized entropy.
- Descriptor column sorting and nearest-neighbor statistics.
- Internal nearest-neighbor behavior with duplicate points.
- Composition formula extraction and L1 nearest-neighbor distance.
- Log parsing for collect UQ/filter summaries.
- Exact overlap tables.
- Log-det entropy numerical stability with ridge regularization.
- Config validation for `mode="audit"`.
- CLI alias dispatch for `dpeva audit`.

Workflow tests should use fake or tiny in-repo fixtures, not large iter10 artifacts.

## Non-Goals

- No automatic sampling recommendation in MVP.
- No structural RMSD or graph matching.
- No new descriptor generation.
- No dependency on specific DPA4 model internals.
- No mandatory advanced KDE entropy for daily use.
- No modification to collect selection behavior.

## Open Extension Points

Future work can add:

- QUESTS-style atom-centered KDE entropy using atomic descriptors.
- Approximate nearest-neighbor backends for large descriptor sets.
- PCA/UMAP/sketch-map plots.
- Farthest-point landmark coverage curves.
- Dataset compression reports and coreset export.
- Correlation between audit novelty metrics and downstream retraining error.
