---
title: HDF5 Last-Layer Collect Routing Review
status: archived
audience: Developers
last-updated: 2026-06-27
owner: Maintainers
---

# HDF5 last-layer collect routing review

- Status: archived
- Audience: Developers
- Last-Updated: 2026-06-27

## Purpose

This change closes a routing gap in the v0.8.0 DeepMD `dp embed` HDF5 adaptation.
The previous HDF5 support allowed `dpeva feature` to write `embedding.hdf5` and
allowed collect/LLPR paths to read HDF5 datasets, but the generic collect
descriptor input still defaulted to the HDF5 `descriptor` dataset.

For workflows where `dpeva feature` is run with
`feature_exporter="embed"` and `feature_kind="fitting_last_layer"`, the relevant
per-atom representation is stored in HDF5 `atomic_feature`. The reviewed changes
make that dataset selection explicit through `CollectionConfig.desc_feature_kind`
so collect can sample from last-layer features without overloading
`llpr_*_feature_dir` or pretending that last-layer features are ordinary
descriptors.

## Contract changes

- `CollectionConfig.desc_feature_kind` accepts `"descriptor"` and
  `"fitting_last_layer"`, defaulting to `"descriptor"` for compatibility.
- Collect reads HDF5 `descriptor` when `desc_feature_kind="descriptor"`.
- Collect reads HDF5 `atomic_feature` when
  `desc_feature_kind="fitting_last_layer"`.
- `desc_dir` and `training_desc_dir` use the same feature kind so joint sampling
  does not mix descriptor-space and last-layer-space features.
- Recursive HDF5 directory reads preserve pool identity by prefixing system names
  with the relative directory that contains each `embedding.hdf5`.
- Multi-pool `feature_exporter="embed"` writes one `embedding.hdf5` under each
  pool output directory instead of collapsing all pools into one root HDF5 file.
- A new recipe, `examples/recipes/collection/config_collect_hdf5_last_layer.json`,
  documents the minimal collect configuration for HDF5 last-layer sampling.

## Reviewed implementation scope

- `src/dpeva/config.py` defines the new collect field.
- `src/dpeva/workflows/collect.py` centralizes dataset selection and passes it
  into no-filter, filtered, joint, training descriptor, and 2-DIRECT atomic
  feature loading paths.
- `src/dpeva/io/collection.py` resolves HDF5 sources with optional file prefixes,
  applies prefixes to generated datanames, and parameterizes HDF5 dataset reads
  for structural descriptors, atomic features, and LLPR feature sums.
- `src/dpeva/feature/managers.py` mirrors the existing multi-pool `eval_desc`
  layout for the `embed` exporter.
- Unit tests cover config validation, HDF5 `atomic_feature` descriptor pooling,
  HDF5 atomic feature loading, nested pool prefixes, multi-pool embed command
  generation, and collect workflow routing.

## Review findings

No blocking correctness issue was identified in the reviewed diff.

One inherited shell-scripting caveat remains: multi-pool feature scripts build
`mkdir -p <pool_out>` lines directly, matching the existing `eval_desc` pattern.
The new `embed` branch does not introduce a different quoting model. A future
hardening pass can move this script assembly behind a shared shell-quoting helper,
but that is outside this routing change.

## Documentation governance

The contract is reflected in active user/developer documentation:

- `docs/guides/configuration.md` explains `desc_feature_kind` and the
  `descriptor` versus `atomic_feature` mapping.
- `docs/reference/validation.md` records the validation rule and recursive HDF5
  pool naming behavior.
- `examples/recipes/README.md` indexes the new HDF5 last-layer collect recipe.

This report archives the review and purpose of the change in the v0.8.0 document
set so the implementation remains traceable to the DeepMD `dp embed` HDF5
adaptation plan and validation report.
