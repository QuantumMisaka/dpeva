# Integration Tests

This suite verifies CLI, workflow and subprocess handoffs. User-facing runnable
configurations live in [examples/recipes/quickstart](../../examples/recipes/quickstart/README.md).

## Fixtures and prerequisites

Small datasets and the DPA-3.1-3M model are tracked in
[data/](data/README.md); a normal checkout already has them. Restore accidentally
missing fixture files from Git rather than regenerating them from an author's
private workspace. Temporary synthetic inputs belong in the tests' `tmp_path`.
DPA4/DeepMD contract fixtures remain external and use their documented overrides.

Prepare the Python runtime and developer dependencies using the
[installation guide](../../docs/guides/installation.md). Tests that execute real
DeepMD need the matching PyTorch backend and `dp` on `PATH`. They may perform
training and must run on appropriate compute resources.

## Run

```bash
python scripts/run_gate.py integration
```

The local multi-datapool test runs real DeepMD training and requires
`DPEVA_RUN_LOCAL_ITEST=1`. Its Slurm variant requires `DPEVA_RUN_SLURM_ITEST=1`
and working `sbatch`/`squeue`. The real LLPR/DPOSE test additionally uses its own opt-in flag
and models, described in [data/README.md](data/README.md). Do not enable these on
a login node or merely to run a fast unit regression.

When replacing a one-off acceptance script, keep tests that cross real CLI or
subprocess boundaries here; move isolated algorithm/path checks to `unit/`.
