---
alwaysApply: false
description: 
---
# DP-EVA - Deep Potential EVolution Accelerator

DP-EVA is an active learning framework designed for efficient fine-tuning of DPA large atomistic model. It integrates uncertainty quantification (UQ), representative sampling (DIRECT), and automated DFT labeling workflows to minimize data annotation costs while maximizing model performance via fully unraveling the pre-trained knowledge.

## Working Effectively

### Bootstrap and Install

- **Install**: `pip install -e .[dev]` -- Installs package in editable mode with dev dependencies.
- **External Dependencies**:
  - `dp`: DeepMD-kit (Required for Train/Infer/Feature).
  - `abacus`: ABACUS (Required for Labeling).
  - `sbatch`: Slurm (Optional, for HPC submission).
- **Verification**: `dpeva --help` -- Should return exit code 0.

### Test Repository

- **Unit Tests**: `pytest tests/unit` -- Fast, mocked tests. **Run this first.**
- **Integration Tests**: `pytest tests/integration` -- Slower, requires real environment.
- **Single Test**: `pytest tests/unit/utils/test_config_paths.py` -- Run specific test file.

### Lint and Format

- **Check**: `ruff check .`
- **Format**: `ruff format .`
- **Rule**: Always run linting/formatting before committing.

### Build Documentation

- **Build**: `cd docs && make html`
- **Preview**: Open `docs/build/html/index.html`

## Workflow Scenarios

All workflow scenarios and CLI calling for core functionality can be found via documents, like `dp train`, `dp infer`, `dp collect`, etc. All json files example for each workflow should be in `examples/recipes/`.

## Critical Notes

- **Configuration**: DP-EVA uses strict Pydantic validation and integated variable management. See `src/dpeva/config.py` for definitive schema.
- **Pathing**: Use absolute paths in configs to avoid ambiguity.
- **Data**: Most workflows expect `dpdata` compatible formats.

## Agent Documentation Protocol

**All Agents MUST adhere to the following documentation lifecycle rules:**

1. **Code Review Phase**:
   - Create review reports in `docs/reports/` with naming `YYYY-MM-DD-Code-Review-<Topic>.md`.
2. **Feature Planning Phase**:
   - Create implementation plans/specs in `docs/plans/` with naming `YYYY-MM-DD-<Feature>-Plan.md`.
   - Include `status: active` and `audience: developers` in front matter.
3. **Task Completion & Archiving**:
   - Upon feature completion or issue resolution, verify if the plan/report should be archived.
   - Move completed plans to `docs/archive/vX.Y.Z/plans/`.
   - Move completed reports to `docs/archive/vX.Y.Z/reports/`.
   - Update the index file `docs/archive/vX.Y.Z/README.md` immediately.
4. **Sphinx Indexing Requirement**:
   - When adding/moving/deleting `.md` files, YOU MUST check and update the corresponding `.rst` files in `docs/source/`.
   - Ensure `toctree` directives do not reference non-existent files.
   - Run `make html` (if available) or verify paths manually to prevent broken builds.

