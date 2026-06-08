# DP-EVA - Deep Potential EVolution Accelerator

DP-EVA is an active learning framework designed for efficient fine-tuning of DPA universial machine learning potentials. It integrates Uncertainty Quantification (UQ), Representative Sampling (DIRECT), and Automated Labeling (DFT) workflows.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

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

## Validation Scenarios

**Use these scenarios to verify core functionality manually.**

### 1. Training Workflow
- **Command**: `dpeva train examples/recipes/training/config_train.json`
- **Verify**: Check `work_dir` for `0/model.ckpt.pt` and `0/lcurve.out`.

### 2. Inference Workflow
- **Command**: `dpeva infer examples/recipes/inference/config_infer.json`
- **Verify**: Check for `results_*.out` files in the output directory.

### 3. Collection (Sampling)
- **Command**: `dpeva collect examples/recipes/collection/config_collect_normal.json`
- **Verify**: Check for `sampled_data` directory, `collection.log` and other output information.

### 4. Labeling (DFT)
- **Command**: `dpeva label examples/recipes/labeling/config_cpu.json`
- **Verify**: Check for `inputs/` generation and `outputs/cleaned/` data.

### 5. Analysis
- **Command**: `dpeva analysis examples/recipes/analysis/config_analysis.json`
- **Verify**: Check for plots and data stat in log file.

## Repository Structure

```text
dpeva/
├── src/dpeva/              # Source code
│   ├── config.py           # Configuration models (Pydantic)
│   ├── cli.py              # Entry point
│   └── ...
├── tests/                  # Test suite
├── docs/                   # Sphinx documentation
├── examples/recipes/       # Sample configurations
├── .trae/skills/           # Agent development skills
│   ├── audit-codebase      # Code quality auditing
│   ├── code-review-skill   # PR review helper
│   └── release-helper      # Release automation
└── AGENTS.md               # This file
```

## Critical Notes

- **Configuration**: DP-EVA uses strict Pydantic validation. See `src/dpeva/config.py` for definitive schema.
- **Pathing**: Use absolute paths in configs to avoid ambiguity.
- **Data**: Most workflows expect `dpdata` compatible formats.

## Agent Documentation Protocol

**All Agents MUST adhere to the following documentation lifecycle rules:**

1.  **Code Review Phase**:
    - Create review reports in `docs/reports/` with naming `YYYY-MM-DD-Code-Review-<Topic>.md`.
    - Do NOT create reports in root or `docs/archive` directly.

2.  **Feature Planning Phase**:
    - Create implementation plans/specs in `docs/plans/` with naming `YYYY-MM-DD-<Feature>-Plan.md`.
    - Include `status: active` and `audience: developers` in front matter.

3.  **Task Completion & Archiving**:
    - Upon feature completion or issue resolution, verify if the plan/report should be archived.
    - Move completed plans to `docs/archive/vX.Y.Z/plans/`.
    - Move completed reports to `docs/archive/vX.Y.Z/reports/`.
    - Update the index file `docs/archive/vX.Y.Z/README.md` immediately.

4.  **Sphinx Indexing Requirement**:
    - When adding/moving/deleting `.md` files, YOU MUST check and update the corresponding `.rst` files in `docs/source/`.
    - Ensure `toctree` directives do not reference non-existent files.
    - Run `make html` (if available) or verify paths manually to prevent broken builds.

5.  **Strict Prohibition**:
    - NEVER create documentation files in the project root.
    - NEVER leave orphan files without `README.md` indexing.

## Project Contract

- 配置以 `src/dpeva/config.py` 与 Reference 为准
- 主链路以 `src/dpeva/cli.py -> src/dpeva/workflows/*.py` 为准
- 工作流可独立启动，但应共享底层模块，不在 workflow 层复制逻辑
- 对外契约变化必须在同一 PR 同步更新文档与 `examples/recipes/`
- 详细工程契约、文档生命周期与质量门禁统一进入 `docs/guides/developer-guide.md`

## Developing Environment

- 项目测试机器：SAI超算，可调用`sai-user-guide` skill 获取更多信息。
- 项目测试环境：`dpeva-dev` conda env.
