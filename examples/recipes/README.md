# DP-EVA Recipes

This directory contains example recipes for running different DP-EVA workflows.

## Sampling Workflows

### 1. Standard DIRECT Sampling
Located in `sampling_direct/`.
Uses standard structural clustering to select representative frames.

**Usage:**
```bash
cd sampling_direct
python run_direct.py config.json
```

### 2. 2-Step DIRECT Sampling (2-DIRECT)
Located in `sampling_2direct/`.
Uses a two-step clustering approach (Structural -> Atomic) to optimize for labeling costs by selecting frames with representative atomic environments but fewer total atoms.

**Usage:**
```bash
cd sampling_2direct
python run_2direct.py config.json
```

## Configuration
Each recipe folder contains a `config.json` file. You should modify the paths (`testdata_dir`, `desc_dir`) to point to your actual data locations.
