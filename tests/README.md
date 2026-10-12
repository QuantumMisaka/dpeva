# DP-EVA Tests

Tests protect reusable behavior; user workflows live in
[examples/recipes](../examples/recipes/README.md). Start with the
[quickstart recipe](../examples/recipes/quickstart/README.md) to try DP-EVA.

## Test layers

- `unit/`: deterministic algorithms, configuration/path handling, file formats,
  and failure boundaries. Temporary files are isolated with `tmp_path`.
- `integration/`: workflow/CLI handoffs and subprocess boundaries. External
  GPU/Slurm cases are explicitly enabled; ordinary tests use local fixtures.
- `contract/deepmd/`: real upstream CLI/artifact contracts. These require the
  documented runtime and external model fixtures, and do not run by default.

## Run

After preparing the runtime described in the [installation guide](../docs/guides/installation.md),
install developer tools with `python -m pip install -e '.[dev]'`. From the repository root:

```bash
python scripts/run_gate.py unit
python scripts/run_gate.py integration
```

For a focused change, run its owning tests directly, for example:

```bash
python -m pytest tests/unit/utils/test_config_paths.py -q
```

The gate manifest in `scripts/gates.toml` owns commands, coverage and profiles.
See [integration prerequisites](integration/README.md) before enabling costly
runtime tests. Hardware evidence remains opt-in, not a prerequisite for every
contributor. Consolidate temporary probes into the owning tests; do not retain
a second smoke runner or tests that merely assert a historical report's text.
