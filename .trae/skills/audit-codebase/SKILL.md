---
name: "audit-codebase"
description: "Executes project-specific static analysis to detect hardcoded paths, magic numbers, missing docstrings, and commented code. Invoke before committing changes or when asked to check code quality."
---

# DP-EVA Code Audit

This skill runs a custom static analysis tool tailored for the DP-EVA project. It checks for:
1.  **Hardcoded Absolute Paths**: Critical for HPC environment portability.
2.  **Missing Docstrings**: Enforces documentation on public APIs.
3.  **Magic Numbers**: Detects unnamed numeric literals.
4.  **Commented Code**: Identifies potential dead code.

## Usage

To run the audit on the default source directory (`src/dpeva`):

```bash
python scripts/audit.py
```

To run in **strict mode** (fails on warnings):

```bash
python scripts/audit.py --strict
```

To audit a specific directory:

```bash
python scripts/audit.py <directory>
```

## Interpretation

- **🔴 ERROR**: Blocking violation (e.g., hardcoded path). Must be fixed before merge.
- **🟡 WARNING**: High-priority issue (e.g., missing docstring on public class). Should be fixed.
- **🔵 INFO**: Code smell (e.g., magic number, commented code). Recommendation for cleanup.
