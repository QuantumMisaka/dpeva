---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# DPEVA Release Notes - v0.5.2

## Labeling Workflow Enhancements (2026-03-07)

### New Features
*   **Hierarchical Statistics**: Implemented Global -> Dataset -> Structure Type granular statistics tracking for labeling tasks.
*   **Metadata Injection**: `AbacusGenerator` now writes `task_meta.json` (Dataset/Type info) to each task directory, ensuring task lineage is preserved even after packing.
*   **Directory Restoration**: Converged tasks are now moved to `CONVERGED/[dataset]/[type]/[task]` instead of a flat list, restoring the logical hierarchy.
*   **OMP Threads Injection**: Automatically injects `OMP_NUM_THREADS` into generated runner scripts based on configuration.

### Bug Fixes
*   **Double Counting Fix**: Resolved an issue where `process_results` double-counted failed tasks by excluding `INPUT` files in `OUT.*` backup directories.
*   **Collection Fix**: Fixed `RuntimeError: Object must be System or MultiSystems!` in `dpdata.MultiSystems` collection logic.
*   **Output Directory**: Fixed hardcoded output directory bug; now correctly uses user-configured `output_dir`.

### Refactoring
*   **Structure Analysis Decoupling**: Moved structure analysis logic from `AbacusGenerator` to `StructureAnalyzer` (`dpeva.labeling.structure`), improving modularity and testability.
*   **Logging Optimization**: Reduced Slurm monitoring log frequency to once every 10 minutes to reduce noise.

### Verification
*   Passed full end-to-end test on `fp-setting-2` dataset.
*   Verified correct generation of `task_meta.json`.
*   Verified correct directory restoration in `CONVERGED`.
*   Verified correct statistics reporting.
