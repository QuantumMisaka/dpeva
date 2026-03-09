---
name: "docs-governance"
description: "Establishes a robust documentation governance system. Invoke when setting up docs, auditing structure, enforcing standards, or integrating doc automation."
---

# Documentation Governance System

This skill provides a comprehensive framework for managing software documentation as code (Docs-as-Code). It is distilled from best practices in large-scale engineering projects.

## When to Use
- **Setup**: Initializing a new `docs/` directory structure.
- **Audit**: Checking existing docs for structural integrity, broken links, or missing metadata.
- **Policy**: Establishing maintenance rules, RACI matrices, and contribution guidelines.
- **Automation**: Integrating documentation checks into CI/CD pipelines.

## 1. Standard Directory Structure

A healthy documentation system should be organized by **User Intent** (Function) and **Lifecycle** (Cadence).

```text
docs/
├── guides/         # [High Freq] How-to guides & tutorials (User-centric)
├── reference/      # [Med Freq]  API/Config references (SSOT, often auto-generated)
├── architecture/   # [Low Freq]  Design docs & decision records (ADR)
├── policy/         # [Low Freq]  Contribution & maintenance policies
├── governance/     # [Meta]      Audit reports, tooling, & inventory
├── plans/          # [Active]    Current feature specs & execution plans
├── archive/        # [Frozen]    Historical docs (vX.Y/plans, vX.Y/specs)
└── README.md       # [Entry]     Navigation index & audience routing
```

## 2. Governance Policies

### 2.1 Metadata Standard (Front Matter)
All active Markdown files MUST include YAML front matter for machine readability.

```yaml
---
title: <Document Title>
status: active      # active | draft | deprecated | archived
audience: developers # developers | users | maintainers
last-updated: YYYY-MM-DD
owner: <Team/Role>
---
```

### 2.2 RACI Matrix (Responsibility)
Define clear ownership for each document type.

| Type | Responsible (Writer) | Accountable (Approver) |
| :--- | :--- | :--- |
| **Guides** | Feature Dev | Tech Lead |
| **Reference** | Code Owner | Architect |
| **Architecture** | Architect | Project Lead |
| **Policy** | Docs Owner | Project Lead |

## 3. Automation & Quality Gates

### 3.1 Governance Script (`scripts/doc_check.py`)
Use this script logic to enforce structure and metadata.

```python
import os
import re
import yaml
from pathlib import Path

def check_governance(docs_root="docs"):
    errors = []
    # 1. Structure Check: Ensure README.md exists in every active dir
    for root, dirs, files in os.walk(docs_root):
        if "archive" in root or "_static" in root: continue
        if "README.md" not in files:
            errors.append(f"Missing README.md in {root}")

    # 2. Metadata Check: Ensure Front Matter exists
    for root, _, files in os.walk(docs_root):
        for f in files:
            if f.endswith(".md") and "archive" not in root:
                path = os.path.join(root, f)
                with open(path) as file:
                    if not file.read().startswith("---"):
                        errors.append(f"Missing Front Matter in {path}")
    
    return errors
```

### 3.2 CI Integration (`.github/workflows/doc-lint.yml`)

```yaml
name: Doc Governance
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check Structure & Metadata
        run: python3 scripts/doc_check.py
      - name: Check Freshness
        run: python3 scripts/check_docs_freshness.py --days 90
```

## 4. Best Practices

1.  **Single Source of Truth (SSOT)**: Never duplicate field definitions. Use `docs/reference` as the authority and link to it from `guides`.
2. **Archive Strategy**: Do not delete old plans/specs. Move them to `docs/archive/vX.Y/` to preserve historical context without cluttering the active view.
3. **Docs-Code Sync**: Treat documentation changes as code changes. Require docs updates in the same PR as feature changes.
4. **Review Process**: Every docs PR must be reviewed by the assigned Owner (defined in RACI).

## 5. Core Documents Breakdown

### 5.1 Developer Guide (`docs/guides/developer-guide.md`)
- **Purpose**: The primary entry point for contributors.
- **Content**: Architecture overview, development workflow, testing standards, and contribution guidelines.
- **Role**: Serves as the "How-to-Dev" manual.

### 5.2 Guides Directory (`docs/guides/`)
- **Purpose**: Task-oriented documentation for users and developers.
- **Content**:
    - `installation.md`: Environment setup.
    - `quickstart.md`: "Hello World" example.
    - `cli.md`: Command-line interface reference.
    - `configuration.md`: Configuration file guide.
    - `slurm.md`: HPC job submission guide.
    - `troubleshooting.md`: Common issues and fixes.
- **Role**: The "User Manual" for the system.

### 5.3 User Entry Points
- **README.md**: The root entry point, directing different audiences (Users, Developers, Researchers) to their respective guides.
- **docs/README.md**: The documentation index, explaining the structure and navigation.

## 6. Execution Checklist

- [ ] **Scaffold**: Create directory tree.
- [ ] **Policy**: Write `docs/policy/{maintenance,contributing}.md`.
- [ ] **Tooling**: Implement `scripts/doc_check.py`.
- [ ] **CI**: Configure GitHub Actions.
- [ ] **Audit**: Run initial scan and fix debts.
