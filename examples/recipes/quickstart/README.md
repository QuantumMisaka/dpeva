# Local quickstart recipe

This directory contains the single local `feature → train → infer → collect`
example used by the [quickstart guide](../../../docs/guides/quickstart.md).
It reads the pretrained model and tiny datasets in `tests/integration/data/`.
The included `input.json` limits training to 30 steps. The fixture is for
learning the workflow, not evaluating scientific performance.

From the repository root, after installing DP-EVA and a DeepMD `dp` command
with the PyTorch backend:

```bash
dpeva feature examples/recipes/quickstart/feature.json
dpeva train examples/recipes/quickstart/train.json
dpeva infer examples/recipes/quickstart/infer.json
dpeva collect examples/recipes/quickstart/collect.json
```

The JSON files resolve input and output paths relative to this directory. If
you run from elsewhere, pass absolute paths to the four JSON files. All
generated files land in `work/`, which Git ignores.
