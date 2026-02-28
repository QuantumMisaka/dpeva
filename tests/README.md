# DP-EVA Tests

This directory contains the test suite for DP-EVA.

## Structure

- `unit/`: Unit tests for individual components. Fast and isolated.
- `integration/`: End-to-end integration tests. May require external dependencies (Slurm, DeepMD-kit) and data.

## Running Tests

To run all tests:
```bash
pytest tests
```

To run only unit tests:
```bash
pytest tests/unit
```

To run integration tests:
```bash
pytest tests/integration
```

## Environment Variables

- `DPEVA_INTERNAL_BACKEND`: Overrides the backend in config (e.g. `local` or `slurm`).
- `DPEVA_RUN_SLURM_ITEST`: Set to `1` to enable Slurm integration tests (requires `sbatch`).
