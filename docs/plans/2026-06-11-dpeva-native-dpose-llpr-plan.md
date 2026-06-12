---
title: DP-EVA Native DPOSE/LLPR Plan
status: active
audience: Developers
last-updated: 2026-06-11
owner: Maintainers
---

# DP-EVA Native DPOSE/LLPR Plan

## Context

`test/lab-cosmo` contains metatrain LLPR and direct propagation of shallow
ensemble references. Those materials remain research references for DP-EVA; they
are not runtime dependencies in the first implementation stage.

The local `test/deepmd-kit` repository changes the earlier feasibility picture:
DeepMD-kit does not expose a `dp eval-lastlayer` CLI, but it does expose the
Python inference API `DeepPot.eval_fitting_last_layer(...)`. DP-EVA can therefore
build a native last-layer LLPR/DPOSE backend without forking deepmd-kit.

## First-Stage Scope

- Implement DP-EVA-native LLPR state construction from DeepMD fitting-net
  last-layer features.
- Support `feature_kind="fitting_last_layer"` in Python feature generation.
- Keep CLI feature generation limited to descriptors because deepmd-kit has no
  `dp eval-lastlayer` command.
- Provide collect-ready energy UQ columns:
  - `uq_llpr_energy_total`
  - `uq_llpr_energy_per_atom`
  - `uq_llpr_alpha`
  - `uq_llpr_calibrated`
- Route `collect` with `uq_backend="llpr"` through DeepMD last-layer feature
  uncertainty, using raw frame-summed last-layer features for LLPR covariance
  and existing structural descriptors for downstream DIRECT sampling.
- Keep force UQ on the existing QbC/RND path for this stage.
- Align LLPR calibration and covariance normalization with metatrain semantics
  for energy/system uncertainty. DPOSE ensembles require real last-layer weights;
  DP-EVA must not synthesize metatrain-like ensemble outputs from detached
  features alone.

## Implementation Boundary

DP-EVA owns the orchestration and lightweight LLPR numerics:

- `dpeva.feature.generator` extracts descriptor or fitting-last-layer features.
- `dpeva.uncertain.llpr` holds covariance, calibration, and shallow ensemble
  sampling logic.
- `dpeva.uncertain.dpose` defines the internal DPOSE ensemble and DeepMD
  PyTorch adapter contract used for differentiable force propagation.
- `dpeva.uncertain.manager` exposes a collect-ready LLPR analysis entry point.
- `dpeva.workflows.collect` supports `uq_backend="llpr"` with
  `llpr_train_feature_dir` and `llpr_candidate_feature_dir`.

Minimal collection configuration shape:

```yaml
uq_backend: llpr
uq_trust_mode: manual
uq_llpr_energy_trust_lo: 0.0
uq_llpr_energy_trust_hi: 0.2
llpr_train_feature_dir: path/to/train_last_layer
llpr_candidate_feature_dir: path/to/candidate_last_layer
llpr_targets: energy
```

Force-level DPOSE is only valid when a DeepMD PyTorch model path can expose a
differentiable energy graph. The public `DeepPot.eval_fitting_last_layer(...)`
API returns detached features and is therefore energy-LLPR only. If
`llpr_targets` requests force uncertainty without a differentiable adapter,
DP-EVA must fail fast instead of silently falling back to QbC/RND or detached
feature heuristics.

metatrain remains the reference for formulas, calibration options, and exported
`energy_uncertainty` / `energy_ensemble` semantics. It is not added to
`pyproject.toml` in this stage because metatrain DPA3 currently does not advertise
last-layer feature support in its test contract, and adding metatomic/metatensor
would materially change DP-EVA's dependency boundary.

## Force-Level Roadmap

Force-level DPOSE needs ensemble energy heads that remain differentiable with
respect to coordinates. A simple call to `eval_fitting_last_layer(...)` is
sufficient for energy-level LLPR features, but not enough to produce physically
meaningful force ensemble propagation.

### Current DP-EVA Status

As of 2026-06-11, DP-EVA has landed the energy-level and contract-level pieces,
but it has not landed a real DeepMD force-level DPOSE adapter.

- Implemented:
  - `dpeva.uncertain.llpr` provides metatrain-parity covariance, calibration,
    per-channel alpha, and energy/system uncertainty.
  - `dpeva.uncertain.dpose.DPOSEEnsemble` can sample shallow last-layer energy
    ensembles when real last-layer weights are supplied.
  - `DeepMDTorchDPOSEAdapter` defines the differentiable adapter contract and
    proves the autograd force propagation logic with a fake differentiable
    PyTorch model in unit tests.
  - Real GPU integration tests validate DPA3/DPA4 `DeepPot.eval(...)`,
    `DeepPot.eval_fitting_last_layer(...)`, detached last-layer energy LLPR, and
    LLPR runtime logging.
- Not implemented:
  - No production `DeepMDInternalTorchDPOSEAdapter` currently extracts
    differentiable DeepMD last-layer features from real DPA3/DPA4 internals.
  - No workflow path currently emits real force-level DPOSE columns from a
    DeepMD model.
  - Detached `.npy` last-layer feature arrays remain energy-only.
- Guardrail:
  - `UQManager.run_llpr_analysis(...)` raises immediately when
    `llpr_targets` is `force` or `energy_force`, because detached features
    cannot support force DPOSE.

### Energy-Level LLPR Maturity Review

The energy-level LLPR implementation is usable as a low-cost DeepMD last-layer
uncertainty backend, but it should be described as analytic energy LLPR rather
than as complete metatrain/DPOSE shallow-ensemble parity.

Implemented and aligned with the lab-cosmo/metatrain reference:

- DeepMD fitting last-layer features are used as the LLPR input, so the score is
  tied to the model's task-space representation rather than to a generic
  structural descriptor.
- The covariance, regularized Cholesky solve, and predictive standard-deviation
  computation match the metatrain LLPR core: accumulate `F.T @ F`, solve through
  the Cholesky factor, and return a standard deviation rather than a variance.
- The default system-energy feature normalization follows the metatrain intent
  of normalizing system targets by atom count before covariance accumulation.
- Post-hoc calibration supports the same method families used by metatrain:
  squared residuals, absolute residuals with the Gaussian `sqrt(pi / 2)` factor,
  and Gaussian CRPS-style scale fitting.
- Real GPU validation on the SAI 4V100 partition has exercised DPA3 and DPA4
  `DeepPot.eval(...)`, `DeepPot.eval_fitting_last_layer(...)`, detached
  energy-only LLPR scoring, and LLPR logging.

Remaining gaps before calling the energy-level path a complete DPOSE ensemble:

- The user-facing workflow does not yet extract real DeepMD last-layer weights
  automatically. Without those weights, DP-EVA can compute analytic LLPR
  uncertainty but cannot emit a metatrain-equivalent `energy_ensemble`.
- `DPOSEEnsemble` can sample around supplied real last-layer weights, but this
  is not yet connected to a production DeepMD weight extractor or collect output
  contract.
- DP-EVA does not train shallow ensemble weights with metatrain's recommended
  ensemble-specific proper scoring losses such as Gaussian NLL, Gaussian CRPS,
  or empirical CRPS.
- The CRPS calibration path should still be cross-checked numerically against
  metatrain's reference root solver before strict parity is claimed.
- Statistical usefulness has not yet been benchmarked on a labeled DP-EVA pool
  with reliability curves, uncertainty-error correlation, coverage, and active
  learning enrichment metrics.

Computationally, energy LLPR is efficient. It needs one DeepMD feature-extraction
pass plus dense linear algebra with feature dimension `d`: covariance is
`O(N * d^2)`, Cholesky is `O(d^3)`, prediction is `O(M * d^2)`, and memory is
`O(d^2)` for the covariance. The observed DPA3/DPA4 fixture feature size is
small enough that the linear algebra cost is minor compared with model
inference. Large candidate pools may still need batched feature loading or
memory-mapped arrays because raw `(frames, atoms, d)` feature arrays can dominate
I/O and storage.

Statistically, the current value is real but bounded. Uncalibrated
`alpha = 1` should be treated as a relative last-layer rigidity or OOD ranking
score, not as an absolute energy-error bar. With labeled calibration residuals
and validation, `uq_llpr_energy_per_atom` can become an interpretable
energy-error scale. Until force-level DPOSE is available, energy LLPR should
remain a companion signal beside force-oriented QbC/RND rather than replacing
them as the only active-learning gate.

### Engineering Assessment

Force-level DPOSE is feasible but should be treated as a medium-to-high
engineering task. The algorithmic core is small; the difficult part is
integrating with deepmd-kit's internal PyTorch inference backend without relying
on detached public outputs.

Recommended phasing:

1. Prototype a DPA3/DPA4-only internal adapter in isolation.
   - Estimated effort: 3-5 effective engineering days.
   - Scope: PyTorch `.pt` backend, OMat24-style head, GPU-only validation,
     single-frame or small-batch inference, energy-gradient force models only.
   - Success criterion: retain `coords.requires_grad=True` through ensemble
     energy and compute `force_ensemble = -grad(energy_member, coords)` on real
     DPA3/DPA4 models.
2. Harden the adapter for user-facing workflow integration.
   - Estimated effort: 1-2 weeks.
   - Scope: version guards, clear unsupported-model diagnostics, last-layer
     weight extraction, batching, logging, Slurm GPU regression tests, and
     collect output columns such as `force_uncertainty_max`.

Primary technical risks:

- Public `DeepPot.eval_fitting_last_layer(...)` returns detached numpy data and
  cannot be used for force DPOSE.
- deepmd-kit's PyTorch backend does expose internal torch models such as
  `self.dp.model["Default"]` and lower-level `forward_common_lower(...)`, but
  these are not stable public APIs.
- Current deepmd-kit fitting-last-layer hooks append detached `middle_output`
  tensors, so a force adapter must intercept or reconstruct the graph before the
  detach point.
- DPA4/SeZM has more internal paths than DPA3, including lower-level neighbor
  list, mapping, and fitting-net flow. The adapter must be guarded by model
  family and deepmd-kit version checks.
- Direct-force or denoising-force model paths are not equivalent to
  energy-gradient force DPOSE and must remain unsupported unless proven
  algorithmically valid.

Minimum acceptance criteria before exposing force-level DPOSE to users:

- Real DPA3 and DPA4 GPU tests pass with differentiable force propagation.
- Tests prove the coordinate graph is not detached before
  `torch.autograd.grad(...)`.
- Detached public API paths continue to fail fast for `force` and
  `energy_force`.
- Runtime logs report model path, head, backend, feature shape, ensemble size,
  and energy/force uncertainty ranges.
- The collect workflow only exposes force DPOSE columns after the real internal
  adapter is selected and validated.

The recommended follow-up is:

1. Validate energy LLPR on DPA4/SeZM and DPA3 model fixtures.
2. Compare `uq_llpr_energy_per_atom` against existing force QbC/RND on labeled
   pools.
3. Prototype differentiable shallow energy heads for force propagation in an
   isolated branch or upstream deepmd-kit adapter.
4. Only after validation, expose force-level DPOSE columns in collect.
