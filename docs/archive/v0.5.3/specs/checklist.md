---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# Checklist: Labeling Module

## Code Quality
- [x] All new Python files have docstrings and type hints.
- [x] No hardcoded paths (e.g., `/data/home/jiangh/...`) in the source code; use Config.
- [x] "Zen of Python" followed: Explicit config, Simple logic, Flat structure.

## Functional Verification
- [x] **Input Generation**:
    - [x] Correctly identifies Cluster, Layer, String, Bulk.
    - [x] Correctly sets K-points based on criteria.
    - [x] Correctly swaps axes for 1D/2D structures (Vacuum -> Z).
- [x] **Strategy**:
    - [x] Can modify `mixing_beta` in `INPUT` files.
    - [x] Returns correct modified file content.
- [x] **Post-Processing**:
    - [x] Correctly parses ABACUS outputs (Energy, Force, Virial).
    - [x] Correctly identifies outliers based on thresholds.
    - [x] Correctly exports to `deepmd/npy`.

## Test Data Verification (`test/fp-setting`) [COMPLETED]
- [x] Generated `INPUT` matches reference in `abacus-inputs` (ignoring trivial formatting differences).
- [x] Generated `STRU` matches reference (lattice, positions, magnetism).
- [x] Processed output statistics match reference in `conveged_data_view`.
