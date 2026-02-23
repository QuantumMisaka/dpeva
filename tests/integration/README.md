# Integration Tests

This directory contains end-to-end integration tests for DP-EVA workflows.

## Prerequisites

- **DeepMD-kit**: Must be installed and available in `PATH` (`dp` command).
- **Slurm** (Optional): Required for `test_slurm_multidatapool_e2e.py` if running with `backend=slurm`.

## Data Setup

Integration tests require specific data. The data is managed in `tests/integration/data`.
If the data is missing, run the setup script:

```bash
python tests/integration/setup_data.py
```

This script will attempt to populate the data directory from available sources or generate mock data.

## Running Tests

By default, Slurm tests are skipped. To enable them:

```bash
export DPEVA_RUN_SLURM_ITEST=1
pytest tests/integration
```

Local backend tests run by default if dependencies are met.
