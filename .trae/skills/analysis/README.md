# Analysis Skill

This skill helps users configure and run the DP-EVA Analysis Workflow.

## Capabilities

1.  **Configure Analysis**: Generate `AnalysisConfig` JSON based on user requirements.
2.  **Explain Modes**: Clarify differences between `dataset` mode (statistics) and `model_test` mode (parity plots).
3.  **Troubleshoot**: Help resolve common errors like missing `type_map` or invalid `result_dir`.

## Usage Examples

### 1. Dataset Mode (Statistics)

**User**: "I want to analyze the distribution of energy and forces in my dataset."
**Agent**: Suggests `mode="dataset"` configuration.

```json
{
    "mode": "dataset",
    "result_dir": "path/to/dataset",
    "type_map": ["Fe", "C"],
    "output_dir": "analysis_output"
}
```

### 2. Model Test Mode (Parity Plots)

**User**: "I want to compare my model predictions against ground truth."
**Agent**: Suggests `mode="model_test"` configuration.

```json
{
    "mode": "model_test",
    "result_dir": "path/to/inference_results",
    "type_map": ["Fe", "C"],
    "output_dir": "analysis_output"
}
```

## Critical Checks

- Ensure `type_map` matches the system.
- Ensure `result_dir` contains valid `*.npy` or `results.*.out` files depending on mode.
