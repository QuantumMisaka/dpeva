---
title: Docs Audit And DP-EVA Operator Skill Implementation Plan
status: active
audience: Maintainers / Developers / AI Agents
last-updated: 2026-07-01
owner: Docs Owner
---

# Docs Audit And DP-EVA Operator Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the active DP-EVA documentation system back into exact agreement with the current CLI/config/workflow surface, then add a project-contained Skill bundle that helps human and AI users operate DP-EVA quickly.

**Architecture:** Treat code, recipes, tests, and generated API reference as the sources of truth. Add executable documentation-contract tests first, repair the active docs and Sphinx wiring second, then create one validated `dpeva-operator` Skill bundle under `docs/skills/` with detailed workflow references kept out of the main `SKILL.md`.

**Tech Stack:** Python 3.10+, pytest, Sphinx/MyST/autodoc-pydantic, argparse CLI, Pydantic v2 config models, Codex/Agent Skills `SKILL.md` format.

---

## Current Evidence

- `python3 scripts/doc_check.py`: passes.
- `python3 scripts/check_docs_freshness.py --days 90`: passes.
- `make -C docs html SPHINXOPTS="-W --keep-going"`: passes.
- `PYTHONPATH=src python3 -m dpeva.cli --no-banner --help`: live commands are `train`, `infer`, `feature`, `explore`, `collect`, `analysis`, `label`, `clean`.
- Observed drift not caught by current checks:
  - `docs/governance/traceability/feature-doc-matrix.md` omits `explore` and `clean`.
  - `docs/governance/traceability/workflow-contract-test-matrix.md` omits `explore` and `clean`, still names `df_uq_desc_sampled-final.csv`, and references `tests/unit/workflows/test_collect_logging_fix.py`, which is not present.
  - `docs/source/api/config.rst` omits `DataCleaningConfig`.
  - `docs/guides/quickstart.md` omits `explore` and `clean` from scope and still uses the old Collect verification file name.
  - `scripts/check_docs.py` hard-codes only six subcommands, so it cannot detect current command drift.

## File Structure

- Create: `tests/unit/scripts/test_docs_contracts.py`
  Executable documentation-contract tests for CLI coverage, config reference coverage, workflow/test matrix accuracy, Skill placement, and docs governance behavior around Skill bundles.
- Modify: `scripts/check_docs.py`
  Replace hard-coded CLI command coverage with dynamic parsing from current `dpeva --help`.
- Modify: `scripts/doc_check.py`
  Allow `docs/skills/<skill-name>/SKILL.md` bundles to omit project-doc `README.md` and `status/audience/last-updated` metadata while still requiring `docs/skills/README.md`.
- Create: `docs/source/api/config/data-cleaning.rst`
  Sphinx/autodoc-pydantic reference page for `dpeva.config.DataCleaningConfig`.
- Modify: `docs/source/api/config.rst`
  Add `config/data-cleaning` to the config toctree.
- Modify: `docs/guides/quickstart.md`
  Include `explore` and `clean`, update Collect verification to `final_df.csv`, and point users to real recipe paths.
- Modify: `docs/guides/cli.md`
  Spell out all live `dpeva <subcommand>` strings so automated CLI-doc coverage can detect them.
- Modify: `docs/README.md`
  Fix root README link, duplicate section numbering, and add Skill navigation.
- Modify: `docs/source/index.rst`
  Add `skills/README` to the stable navigation.
- Modify: `docs/policy/docs-structure.md`
  Document the new `docs/skills/` category and its doc-check exception.
- Modify: `docs/guides/developer-guide.md`
  Update stale log anchors and documentation gates.
- Modify: `docs/governance/traceability/feature-doc-matrix.md`
  Add `explore`, `clean`, and `DataCleaningConfig`; refresh last-updated.
- Modify: `docs/governance/traceability/workflow-contract-test-matrix.md`
  Add `explore`/`clean`, align Collect with `final_df.csv`, and replace stale test references with existing tests.
- Modify: `docs/governance/inventory/docs-catalog.md`
  Add `docs/skills/README.md`, remove active labels from archived documents, and refresh last-updated.
- Create: `docs/reports/2026-07-01-doc-system-audit.md`
  Short audit report recording evidence, drift, fixes, and validation output.
- Create: `docs/skills/README.md`
  Human-facing index for project-contained Skills.
- Create: `docs/skills/dpeva-operator/SKILL.md`
  Main AI-readable Skill entry point.
- Create: `docs/skills/dpeva-operator/agents/openai.yaml`
  UI metadata for Codex/OpenAI skill surfaces.
- Create: `docs/skills/dpeva-operator/references/workflow-map.md`
  Workflow command/input/output/verification map.
- Create: `docs/skills/dpeva-operator/references/config-authoring.md`
  Configuration authoring rules and recipe map.
- Create: `docs/skills/dpeva-operator/references/slurm-monitoring.md`
  Slurm execution and log-monitoring reference.
- Create: `docs/skills/dpeva-operator/references/troubleshooting.md`
  Fast failure triage reference.
- Create: `docs/reports/2026-07-01-dpeva-operator-skill-validation.md`
  Skill validation record with pressure scenarios and command results.

Implementation must stage only the files listed in each task. The current worktree contains unrelated user changes.

---

### Task 1: Add Documentation Contract Tests

**Files:**
- Create: `tests/unit/scripts/test_docs_contracts.py`
- Test: `tests/unit/scripts/test_docs_contracts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/scripts/test_docs_contracts.py` with this exact content:

```python
import ast
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts import doc_check


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_repo_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def cli_subcommands() -> list[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") if not existing else f"{PROJECT_ROOT / 'src'}:{existing}"
    result = subprocess.run(
        [sys.executable, "-m", "dpeva.cli", "--no-banner", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    match = re.search(r"\{([^}]+)\}", result.stdout)
    assert match, result.stdout
    return sorted(command.strip() for command in match.group(1).split(","))


def public_config_classes() -> list[str]:
    tree = ast.parse(read_repo_text("src/dpeva/config.py"))
    return sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Config")
    )


def test_cli_commands_are_documented_in_active_user_and_traceability_docs() -> None:
    commands = cli_subcommands()
    docs = {
        "docs/guides/cli.md": read_repo_text("docs/guides/cli.md"),
        "docs/guides/quickstart.md": read_repo_text("docs/guides/quickstart.md"),
        "docs/governance/traceability/feature-doc-matrix.md": read_repo_text(
            "docs/governance/traceability/feature-doc-matrix.md"
        ),
    }

    for command in commands:
        assert f"dpeva {command}" in docs["docs/guides/quickstart.md"]
        assert f"`dpeva {command}`" in docs["docs/guides/cli.md"]
        assert f"`dpeva {command}`" in docs[
            "docs/governance/traceability/feature-doc-matrix.md"
        ]


def test_sphinx_config_reference_covers_all_public_config_classes() -> None:
    config_reference_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "docs/source/api/config").glob("*.rst"))
    )

    for class_name in public_config_classes():
        assert f"dpeva.config.{class_name}" in config_reference_text


def test_workflow_contract_matrix_uses_current_outputs_and_existing_tests() -> None:
    matrix = read_repo_text("docs/governance/traceability/workflow-contract-test-matrix.md")

    assert "df_uq_desc_sampled-final.csv" not in matrix
    assert "final_df.csv" in matrix

    for command in cli_subcommands():
        assert f"`dpeva {command} <cfg>`" in matrix

    referenced_test_files = re.findall(r"`(tests/[^`]+?\.py)`", matrix)
    assert referenced_test_files
    for test_file in referenced_test_files:
        assert (PROJECT_ROOT / test_file).exists(), test_file


def test_project_skill_bundle_is_indexed_and_has_required_references() -> None:
    skill_root = PROJECT_ROOT / "docs/skills/dpeva-operator"
    skill_file = skill_root / "SKILL.md"
    index_file = PROJECT_ROOT / "docs/skills/README.md"

    assert index_file.exists()
    assert skill_file.exists()

    skill_text = skill_file.read_text(encoding="utf-8")
    assert "name: dpeva-operator" in skill_text
    assert "description: Use when" in skill_text
    assert "dpeva train" in skill_text
    assert "dpeva collect" in skill_text
    assert "docs/guides/cli.md" in skill_text
    assert "examples/recipes/README.md" in skill_text

    for reference_name in [
        "workflow-map.md",
        "config-authoring.md",
        "slurm-monitoring.md",
        "troubleshooting.md",
    ]:
        assert (skill_root / "references" / reference_name).exists()


def test_doc_check_allows_skill_bundles_without_project_doc_metadata(tmp_path, monkeypatch) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "README.md").write_text(
        "---\n"
        "status: active\n"
        "audience: Users\n"
        "last-updated: 2026-07-01\n"
        "owner: Docs Owner\n"
        "---\n\n"
        "# Docs\n",
        encoding="utf-8",
    )
    skills_root = docs_root / "skills"
    skills_root.mkdir()
    (skills_root / "README.md").write_text(
        "---\n"
        "status: active\n"
        "audience: Users / Developers\n"
        "last-updated: 2026-07-01\n"
        "owner: Docs Owner\n"
        "---\n\n"
        "# Skills\n",
        encoding="utf-8",
    )
    skill_dir = skills_root / "dpeva-operator"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: dpeva-operator\n"
        "description: Use when operating DP-EVA workflows.\n"
        "---\n\n"
        "# DP-EVA Operator\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doc_check, "DOCS_ROOT", docs_root)

    assert doc_check.check_structure(docs_root) == []
    assert doc_check.check_front_matter(docs_root) == []
    assert doc_check.check_owner_metadata(docs_root) == []
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest tests/unit/scripts/test_docs_contracts.py -q
```

Expected: FAIL. The failure must include missing `dpeva explore` or `dpeva clean` in `docs/guides/quickstart.md` or `docs/governance/traceability/feature-doc-matrix.md`, missing `dpeva.config.DataCleaningConfig`, missing `docs/skills/dpeva-operator/SKILL.md`, and skill-bundle metadata handling in `doc_check`.

- [ ] **Step 3: Commit the RED tests**

Run:

```bash
git add tests/unit/scripts/test_docs_contracts.py
git commit -m "test: add documentation contract checks"
```

Expected: commit succeeds and contains only `tests/unit/scripts/test_docs_contracts.py`.

---

### Task 2: Make Docs Tooling Enforce Current Contracts

**Files:**
- Modify: `scripts/doc_check.py`
- Modify: `scripts/check_docs.py`
- Test: `tests/unit/scripts/test_docs_contracts.py`
- Test: `tests/unit/scripts/test_doc_governance_scripts.py`

- [ ] **Step 1: Update `scripts/doc_check.py` to skip Skill bundle internals**

In `scripts/doc_check.py`, replace `should_skip` with:

```python
def is_inside_skill_bundle(path, docs_root=None):
    docs_root = Path(docs_root or DOCS_ROOT).resolve()
    current = Path(path)
    current = current if current.is_dir() else current.parent
    current = current.resolve()
    while True:
        if current == docs_root:
            return False
        if (current / "SKILL.md").exists():
            return True
        if current.parent == current:
            return False
        current = current.parent


def should_skip(path, docs_root=None):
    docs_root = Path(docs_root or DOCS_ROOT)
    return (
        is_inside_skill_bundle(path, docs_root)
        or any(part in IGNORE_DIRS for part in path.parts)
        or path.name.startswith(".")
    )
```

Then update `iter_doc_dirs` so the call passes the active docs root:

```python
def iter_doc_dirs(docs_root):
    for root, dirs, files in os.walk(docs_root):
        path = Path(root)
        if should_skip(path, docs_root):
            dirs[:] = []
            continue
        yield path, files
```

- [ ] **Step 2: Replace `scripts/check_docs.py` with dynamic CLI parsing**

Replace the file content with:

```python
#!/usr/bin/env python3
"""
DP-EVA documentation consistency checker.

Checks that the live CLI command surface is documented in active user docs.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATHS = [
    PROJECT_ROOT / "docs/guides/cli.md",
    PROJECT_ROOT / "docs/guides/quickstart.md",
    PROJECT_ROOT / "docs/governance/traceability/feature-doc-matrix.md",
]


def get_cli_help() -> str:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") if not existing else f"{PROJECT_ROOT / 'src'}:{existing}"
    result = subprocess.run(
        [sys.executable, "-m", "dpeva.cli", "--no-banner", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout


def extract_subcommands(help_text: str) -> list[str]:
    match = re.search(r"\{([^}]+)\}", help_text)
    if not match:
        raise RuntimeError(f"Could not parse subcommands from CLI help:\n{help_text}")
    return sorted(command.strip() for command in match.group(1).split(","))


def check_doc_coverage(doc_path: Path, commands: list[str]) -> list[str]:
    content = doc_path.read_text(encoding="utf-8")
    missing = []
    for command in commands:
        if f"dpeva {command}" not in content and f"`dpeva {command}`" not in content:
            missing.append(command)
    return missing


def main() -> None:
    print("Checking CLI documentation consistency...")
    commands = extract_subcommands(get_cli_help())
    print(f"Live subcommands: {', '.join(commands)}")

    failed = False
    for doc_path in DOC_PATHS:
        if not doc_path.exists():
            print(f"Error: documentation file {doc_path} not found.")
            sys.exit(1)
        missing = check_doc_coverage(doc_path, commands)
        if missing:
            print(f"❌ {doc_path.relative_to(PROJECT_ROOT)} missing commands: {missing}")
            failed = True
        else:
            print(f"✅ {doc_path.relative_to(PROJECT_ROOT)} covers all live subcommands.")

    if failed:
        print("\n⚠️ Documentation consistency check FAILED.")
        sys.exit(1)

    print("\n✨ Documentation consistency check PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests to verify tooling changes**

Run:

```bash
pytest tests/unit/scripts/test_doc_governance_scripts.py tests/unit/scripts/test_docs_contracts.py::test_doc_check_allows_skill_bundles_without_project_doc_metadata -q
```

Expected: PASS for these tests.

- [ ] **Step 4: Run `check_docs.py` and confirm it still fails on real doc drift**

Run:

```bash
PYTHONPATH=src python3 scripts/check_docs.py
```

Expected: FAIL listing `explore` and `clean` as missing from `docs/guides/quickstart.md` and `docs/governance/traceability/feature-doc-matrix.md`, plus missing explicit command strings in `docs/guides/cli.md` if they have not been repaired yet.

- [ ] **Step 5: Commit tooling changes**

Run:

```bash
git add scripts/doc_check.py scripts/check_docs.py
git commit -m "chore: enforce live documentation contracts"
```

Expected: commit succeeds and contains only the two scripts.

---

### Task 3: Repair Active Documentation Drift

**Files:**
- Create: `docs/source/api/config/data-cleaning.rst`
- Modify: `docs/source/api/config.rst`
- Modify: `docs/guides/quickstart.md`
- Modify: `docs/guides/cli.md`
- Modify: `docs/README.md`
- Modify: `docs/source/index.rst`
- Modify: `docs/policy/docs-structure.md`
- Modify: `docs/guides/developer-guide.md`
- Modify: `docs/governance/traceability/feature-doc-matrix.md`
- Modify: `docs/governance/traceability/workflow-contract-test-matrix.md`
- Modify: `docs/governance/inventory/docs-catalog.md`
- Test: `tests/unit/scripts/test_docs_contracts.py`

- [ ] **Step 1: Add DataCleaningConfig to Sphinx config reference**

Create `docs/source/api/config/data-cleaning.rst`:

```rst
Data Cleaning Configuration
===========================

.. autopydantic_model:: dpeva.config.DataCleaningConfig
   :model-show-json: False
   :field-list-validators: False
   :members:
   :undoc-members:
   :exclude-members: model_config
```

In `docs/source/api/config.rst`, change the toctree to include `config/data-cleaning`:

```rst
Configuration Reference
=======================

.. toctree::
   :maxdepth: 2
   :caption: By Workflow

   config/submission
   config/feature
   config/exploration
   config/training
   config/inference
   config/collection
   config/labeling
   config/analysis
   config/data-cleaning
```

- [ ] **Step 2: Update Quickstart command surface and Collect output**

In `docs/guides/quickstart.md`, set both metadata dates to `2026-07-01` and replace the scope list with:

```markdown
范围：

- 覆盖 CLI 子命令：`train / infer / feature / explore / collect / label / analysis / clean`
- 给出 Local 与 Slurm 两种运行方式的最小示例
```

Replace the recipe pointer under section 5 with:

```markdown
参考最小配置模板：

- `examples/recipes/README.md`
- 配置字段字典：API Reference（Sphinx 生成的配置字段文档）
```

Replace Collect verification with:

```markdown
验证输出：

- `<root_savedir>/dataframe/final_df.csv` 存在
- `<root_savedir>/dataframe/df_uq.csv` 与 `<root_savedir>/dataframe/df_uq_desc.csv` 存在
- `<root_savedir>/dpdata/` 下导出目录存在
```

Insert this section after `5.6 Analysis`:

```markdown
### 5.7 Explore（可选轨迹探索）

```bash
dpeva explore configs/explore.json
```

验证输出：

- `work_dir/dpeva_exploration_result.json` 存在
- 若配置 `input_structure_paths`，`work_dir/dpeva_inputs/` 存在

### 5.8 Clean（按推理误差清洗数据）

```bash
dpeva clean configs/clean.json
```

验证输出：

- `output_dir/cleaning_summary.json` 存在
- `output_dir/frame_metrics.csv` 存在
- `output_dir/cleaned_dpdata` 与 `output_dir/filtered_out_dpdata` 存在
```

Append this changelog entry first:

```markdown
- 2026-07-01：补齐 `explore` 与 `clean` Quickstart 入口，并将 Collect 验证文件更新为 `final_df.csv`。
```

- [ ] **Step 3: Update CLI guide command coverage**

In `docs/guides/cli.md`, set both metadata dates to `2026-07-01` and replace the scope bullet:

```markdown
- `dpeva train / infer / feature / explore / collect / label / analysis / clean`
```

with:

```markdown
- `dpeva train`
- `dpeva infer`
- `dpeva feature`
- `dpeva explore`
- `dpeva collect`
- `dpeva label`
- `dpeva analysis`
- `dpeva clean`
```

Append this changelog entry first:

```markdown
- 2026-07-01：将 CLI 范围中的子命令写为显式 `dpeva <subcommand>` 字符串，便于自动化一致性检查。
```

- [ ] **Step 4: Update docs index and Sphinx navigation for Skills**

In `docs/README.md`, change the user entry from:

Target line: visible text `README.md` with markdown link target `README.md`.

to:

Replacement: keep visible text `README.md` and change the markdown link target to `../README.md`.

Add this user entry after CLI:

Add a markdown link with visible text `dpeva-operator` and target `skills/dpeva-operator/SKILL.md`.

Change the second `### 2.1` heading to:

```markdown
### 2.3 现行文档与归档文档判别规则
```

Add this bullet to the document-layer list:

```markdown
- `docs/skills/`：项目内 Skill 资产；`README.md` 进入文档导航，具体 Skill bundle 使用 Agent Skills 结构并由 Skill 校验工具验证。
```

In `docs/source/index.rst`, add `skills/README` to the `User Guides` toctree:

```rst
.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/slurm
   guides/troubleshooting
   skills/README
```

- [ ] **Step 5: Update docs structure policy**

In `docs/policy/docs-structure.md`, set both metadata dates to `2026-07-01`.

Add `Skills` to the classification list:

```markdown
- 按 Intent（功能）分层：Guides / Reference / Architecture / Decisions(ADR) / Reports / Policy / Governance / Plans / Skills / Archive / Templates / Assets
```

Add this rule under section 2:

```markdown
- **Skill Bundle**: `docs/skills/README.md` 是文档系统入口；`docs/skills/<skill-name>/SKILL.md` 与其 `references/`、`agents/` 子目录遵循 Agent Skills 结构，由 Skill 校验工具负责，不要求项目文档 front matter 或子目录 README。
```

Add this tree entry under `/docs`:

```text
├── skills
│   ├── README.md
│   └── dpeva-operator
│       ├── SKILL.md
│       ├── agents
│       │   └── openai.yaml
│       └── references
│           ├── workflow-map.md
│           ├── config-authoring.md
│           ├── slurm-monitoring.md
│           └── troubleshooting.md
```

- [ ] **Step 6: Update developer guide stale log anchors**

In `docs/guides/developer-guide.md`, replace the log-monitoring sentence that includes `train.log` with:

```markdown
*   外部调度系统应通过 `grep` 或正则表达式持续监控任务的 Log 文件（如 `train.out`, `test_job.out`, `eval_desc.log`, `collect_slurm.out`, `analysis.log`, `cleaning.log` 或 Slurm `.out` 文件）。
```

Add `python3 scripts/check_docs.py` to both documented docs gates:

```markdown
*   文档治理：`python3 scripts/doc_check.py`
*   CLI 文档一致性：`PYTHONPATH=src python3 scripts/check_docs.py`
*   文档新鲜度：`python3 scripts/check_docs_freshness.py --days 90`
*   Sphinx 构建：`make -C docs html SPHINXOPTS="-W --keep-going"`
```

- [ ] **Step 7: Update traceability feature matrix**

In `docs/governance/traceability/feature-doc-matrix.md`, set both metadata dates to `2026-07-01`.

Replace the CLI command table with:

```markdown
| 功能点 | 代码实现 | 文档入口 |
|---|---|---|
| `dpeva train` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva infer` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva feature` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva explore` | `/src/dpeva/cli.py`、`/src/dpeva/exploration/` | `/docs/guides/cli.md`、`/docs/guides/configuration.md`、`/examples/recipes/exploration/README.md` |
| `dpeva collect` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva analysis` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva label` | `/src/dpeva/cli.py` | `/docs/guides/cli.md`、`/docs/guides/quickstart.md` |
| `dpeva clean` | `/src/dpeva/cli.py`、`/src/dpeva/workflows/data_cleaning.py` | `/docs/guides/cli.md`、`/docs/guides/configuration.md`、`/examples/recipes/README.md` |
```

Add this row to the config table:

```markdown
| `DataCleaningConfig` | `/src/dpeva/config.py` | `API Reference`（构建入口：`/docs/source/api/config/data-cleaning.rst`）、`/docs/reference/validation.md` |
```

- [ ] **Step 8: Update workflow contract matrix**

In `docs/governance/traceability/workflow-contract-test-matrix.md`, set both metadata dates to `2026-07-01`.

Replace section 2 table with:

```markdown
| 工作流 | 入口命令 | 最小产物（存在性断言） | 完成标记（日志） | Unit 覆盖 | Integration 覆盖 |
|---|---|---|---|---|---|
| Feature | `dpeva feature <cfg>` | `savedir/embedding.hdf5` 或 `savedir/` 下 `.npy` 描述符输出 | `eval_desc.log` 或 workflow logger 包含完成标记 | `tests/unit/workflows/test_feature_workflow_submission.py`、`tests/unit/feature/test_execution_manager.py` | `tests/integration/test_slurm_multidatapool_e2e.py` |
| Train | `dpeva train <cfg>` | `work_dir/0..N-1/` + `model.ckpt.pt`（或等价模型产物） | `work_dir/<i>/train.out` 包含完成标记 | `tests/unit/workflows/test_train_workflow_init.py`、`tests/unit/training/test_training_managers.py` | `tests/integration/test_slurm_multidatapool_e2e.py` |
| Infer | `dpeva infer <cfg>` | `work_dir/<i>/<task_name>/results.e.out`（或前缀等价输出） | `work_dir/<i>/<task_name>/test_job.out` 包含完成标记 | `tests/unit/workflows/test_infer_workflow_exec.py`、`tests/unit/inference/test_inference_execution_manager.py` | `tests/integration/test_slurm_multidatapool_e2e.py`、`tests/integration/test_inference_parallel_submission.py` |
| Explore | `dpeva explore <cfg>` | `work_dir/dpeva_exploration_result.json`，可选 `work_dir/dpeva_inputs/` | manifest `status` 表示成功或失败 | `tests/unit/exploration/test_exploration_cli.py`、`tests/unit/exploration/test_atst_backend.py` | `tests/integration/test_v080_atst_acceptance.py` |
| Collect | `dpeva collect <cfg>` | `root_savedir/dataframe/df_uq.csv`、`root_savedir/dataframe/df_uq_desc.csv`、`root_savedir/dataframe/final_df.csv` | `collect_slurm.out`（Slurm worker）或 `collection.log` 包含完成标记 | `tests/unit/workflows/test_collect_workflow_routing.py`、`tests/unit/workflows/test_collect_joint.py`、`tests/unit/workflows/test_workflow_completion_marker.py` | `tests/integration/test_slurm_multidatapool_e2e.py` |
| Analysis | `dpeva analysis <cfg>` | `output_dir/analysis.log` + 模式相关统计或图表文件 | `analysis.log` 包含完成标记 | `tests/unit/workflows/test_analysis_workflow.py`、`tests/unit/analysis/test_unified_analysis_manager.py` | `tests/integration/test_slurm_multidatapool_e2e.py` |
| Label | `dpeva label <cfg>` | `work_dir/outputs/cleaned`，启用 integration 时输出 `integration_summary.json` | `labeling.log` 或 Slurm 输出包含完成标记 | `tests/unit/workflows/test_labeling_workflow.py`、`tests/unit/labeling/test_manager.py` | `tests/integration/test_labeling_rotation_bug.py` |
| Clean | `dpeva clean <cfg>` | `output_dir/cleaning_summary.json`、`output_dir/frame_metrics.csv`、`output_dir/cleaned_dpdata`、`output_dir/filtered_out_dpdata` | `cleaning.log` 包含完成标记 | `tests/unit/workflows/test_data_cleaning_workflow.py` | `tests/integration/test_e2e_cycle.py`（上游数据链路覆盖；清洗端到端可在后续补强） |
```

- [ ] **Step 9: Update docs catalog**

In `docs/governance/inventory/docs-catalog.md`, set both metadata dates to `2026-07-01`.

Add this row under Guides:

Add a row for `docs/skills/README.md` with category `Guide`, audience `Users/AI Agents`, cadence `High`, status `active`, tags `skills|operator|quick-use`, and summary that it is the project Skill entry for quick workflow/config/output/troubleshooting guidance. Use the same markdown-link style as the surrounding catalog rows.

For rows that point into `docs/archive/`, set `Status` to `archived`; for example:

For example, the historical doc-system-planning row should point to `archive/v0.6.1/plans/2026-02-18-doc-system-planning.md` and have `Status` set to `archived`.

- [ ] **Step 10: Run documentation contract tests**

Run:

```bash
pytest tests/unit/scripts/test_docs_contracts.py::test_cli_commands_are_documented_in_active_user_and_traceability_docs tests/unit/scripts/test_docs_contracts.py::test_sphinx_config_reference_covers_all_public_config_classes tests/unit/scripts/test_docs_contracts.py::test_workflow_contract_matrix_uses_current_outputs_and_existing_tests -q
```

Expected: PASS.

- [ ] **Step 11: Run dynamic CLI docs check**

Run:

```bash
PYTHONPATH=src python3 scripts/check_docs.py
```

Expected: PASS with all live subcommands covered.

- [ ] **Step 12: Commit active docs drift repairs**

Run:

```bash
git add docs/source/api/config/data-cleaning.rst docs/source/api/config.rst docs/guides/quickstart.md docs/guides/cli.md docs/README.md docs/source/index.rst docs/policy/docs-structure.md docs/guides/developer-guide.md docs/governance/traceability/feature-doc-matrix.md docs/governance/traceability/workflow-contract-test-matrix.md docs/governance/inventory/docs-catalog.md
git commit -m "docs: align active docs with current workflow contracts"
```

Expected: commit succeeds and stages only these docs.

---

### Task 4: Record The Documentation Audit

**Files:**
- Create: `docs/reports/2026-07-01-doc-system-audit.md`
- Modify: `docs/reports/README.md`
- Test: `scripts/doc_check.py`

- [ ] **Step 1: Create the audit report**

Create `docs/reports/2026-07-01-doc-system-audit.md`:

```markdown
---
title: 2026-07-01 文档系统真实性审查
status: active
audience: Maintainers / Developers
last-updated: 2026-07-01
owner: Docs Owner
---

# 2026-07-01 文档系统真实性审查

## 结论

当前文档系统的结构、链接、新鲜度与 Sphinx 构建均通过既有门禁；但既有门禁没有覆盖“代码契约与文档契约一致性”。本次审查以 `src/dpeva/cli.py`、`src/dpeva/config.py`、`src/dpeva/workflows/`、`examples/recipes/` 与 `tests/` 为事实来源，补齐了可执行契约检查并修复了 active 文档漂移。

## 审查命令

```bash
python3 scripts/doc_check.py
python3 scripts/check_docs_freshness.py --days 90
PYTHONPATH=src python3 scripts/check_docs.py
pytest tests/unit/scripts/test_doc_governance_scripts.py tests/unit/scripts/test_docs_contracts.py -q
make -C docs html SPHINXOPTS="-W --keep-going"
```

## 发现与处置

| 问题 | 事实来源 | 处置 |
|---|---|---|
| Traceability 矩阵遗漏 `dpeva explore` 与 `dpeva clean` | `src/dpeva/cli.py` help output | 更新 `feature-doc-matrix.md` 与 `workflow-contract-test-matrix.md` |
| Config API Reference 遗漏 `DataCleaningConfig` | `src/dpeva/config.py` | 新增 `docs/source/api/config/data-cleaning.rst` |
| Quickstart Collect 输出文件名过期 | `src/dpeva/workflows/collect.py` 与 `tests/integration/test_slurm_multidatapool_e2e.py` | 更新为 `final_df.csv`、`df_uq.csv`、`df_uq_desc.csv` |
| `scripts/check_docs.py` 硬编码旧 CLI 列表 | `scripts/check_docs.py` | 改为动态解析 live CLI help |
| 项目文档系统缺少 Skill 放置规则 | `docs/policy/docs-structure.md` | 新增 `docs/skills/` 规则与导航 |

## 后续门禁

接口、配置、输出目录、日志锚点、完成标记和 recipes 变化时，必须同时运行：

```bash
PYTHONPATH=src python3 scripts/check_docs.py
pytest tests/unit/scripts/test_docs_contracts.py -q
```
```

- [ ] **Step 2: Add report to reports index**

In `docs/reports/README.md`, set both metadata dates to `2026-07-01` and add:

```markdown
- 2026-07-01 文档系统真实性审查：2026-07-01-doc-system-audit.md
```

- [ ] **Step 3: Run report governance check**

Run:

```bash
python3 scripts/doc_check.py
```

Expected: PASS.

- [ ] **Step 4: Commit audit report**

Run:

```bash
git add docs/reports/2026-07-01-doc-system-audit.md docs/reports/README.md
git commit -m "docs: record documentation system audit"
```

Expected: commit succeeds and contains only the report and reports index.

---

### Task 5: Create The Project Skill Bundle

**Files:**
- Create: `docs/skills/README.md`
- Create: `docs/skills/dpeva-operator/SKILL.md`
- Create: `docs/skills/dpeva-operator/agents/openai.yaml`
- Create: `docs/skills/dpeva-operator/references/workflow-map.md`
- Create: `docs/skills/dpeva-operator/references/config-authoring.md`
- Create: `docs/skills/dpeva-operator/references/slurm-monitoring.md`
- Create: `docs/skills/dpeva-operator/references/troubleshooting.md`
- Test: `tests/unit/scripts/test_docs_contracts.py`

- [ ] **Step 1: Create the docs Skill index**

Create `docs/skills/README.md`:

```markdown
---
title: DP-EVA Skills
status: active
audience: Users / Developers / AI Agents
last-updated: 2026-07-01
owner: Docs Owner
---

# DP-EVA Skills

本目录存放项目内 Skill 资产，用于帮助人类和 AI 快速选择 DP-EVA 工作流、配置模板、验证产物与排障入口。

## 当前 Skill

- `dpeva-operator`：`dpeva-operator/SKILL.md`，运行 DP-EVA CLI 工作流、编写配置、监控 Slurm 作业、验证输出与定位常见失败。

## 使用边界

- 事实来源仍以 `src/dpeva/cli.py`、`src/dpeva/config.py`、`docs/guides/cli.md`、`docs/guides/configuration.md` 与 `examples/recipes/README.md` 为准。
- Skill bundle 内部遵循 Agent Skills 结构，不要求项目文档 front matter；项目文档入口只维护本页。
```

- [ ] **Step 2: Create `SKILL.md`**

Create `docs/skills/dpeva-operator/SKILL.md`:

```markdown
---
name: dpeva-operator
description: Use when a human or AI agent needs to operate DP-EVA CLI workflows, choose recipes, write JSON configs, run active-learning steps, monitor Slurm jobs, verify outputs, or troubleshoot failed train/infer/feature/explore/collect/label/analysis/clean runs.
---

# DP-EVA Operator

## Core Rule

Query the repository before acting. Treat `src/dpeva/cli.py`, `src/dpeva/config.py`, `docs/guides/cli.md`, `docs/guides/configuration.md`, and `examples/recipes/README.md` as the current contract.

## First Commands

```bash
PYTHONPATH=src python3 -m dpeva.cli --no-banner --help
sed -n '1,260p' docs/guides/cli.md
sed -n '1,260p' docs/guides/configuration.md
sed -n '1,260p' examples/recipes/README.md
```

## Workflow Selection

| Need | Command | Start from |
|---|---|---|
| Fine-tune models | `dpeva train <config.json>` | `examples/recipes/training/config_train.json` |
| Run model inference | `dpeva infer <config.json>` | `examples/recipes/inference/config_infer.json` |
| Export descriptors or embeddings | `dpeva feature <config.json>` | `examples/recipes/feature_generation/config_feature.json` |
| Run optional trajectory exploration | `dpeva explore <config.json>` | `examples/recipes/exploration/config_explore_md.json` |
| Select active-learning samples | `dpeva collect <config.json>` | `examples/recipes/collection/config_collect_normal.json` |
| Prepare and run labeling | `dpeva label <config.json>` | `examples/recipes/labeling/config_cpu.json` |
| Analyze model or dataset outputs | `dpeva analysis <config.json>` | `examples/recipes/analysis/config_analysis.json` |
| Clean labeled data by error thresholds | `dpeva clean <config.json>` | `examples/recipes/data_cleaning/config_clean_all_thresholds.json` |

## Read References As Needed

- Workflow commands, inputs, outputs, and completion checks: `references/workflow-map.md`
- Config path rules and recipe selection: `references/config-authoring.md`
- Slurm execution and log monitoring: `references/slurm-monitoring.md`
- Fast failure triage: `references/troubleshooting.md`

## Operating Loop

1. Run `dpeva --help` or `PYTHONPATH=src python3 -m dpeva.cli --no-banner --help` to confirm the command exists.
2. Start from the closest file in `examples/recipes/`.
3. Keep paths relative to the config file location.
4. Run the workflow with `dpeva <command> <config.json>`.
5. Verify the documented output files and `DPEVA_TAG: WORKFLOW_FINISHED` where the workflow emits it.
6. When a workflow fails, inspect the workflow log before changing the config.

## Common Mistakes

| Mistake | Correction |
|---|---|
| Guessing config fields from memory | Inspect `src/dpeva/config.py` or Sphinx API Reference first. |
| Running `dpeva label prepare` | Run `dpeva label config.json --stage prepare`. |
| Assuming shell cwd controls relative config paths | DP-EVA resolves relative paths against the config file directory. |
| Treating Slurm queue completion as workflow success | Check the workflow log for `DPEVA_TAG: WORKFLOW_FINISHED` and required output files. |
| Using archived docs as current contract | Prefer active docs in `docs/guides/`, `docs/reference/`, and `examples/recipes/`. |
```

- [ ] **Step 3: Create OpenAI agent metadata**

Create `docs/skills/dpeva-operator/agents/openai.yaml`:

```yaml
display_name: DP-EVA Operator
short_description: Run DP-EVA workflows, configs, Slurm monitoring, and output checks.
default_prompt: Use the DP-EVA Operator skill to choose the right workflow, start from the correct recipe, run the CLI command, and verify outputs.
```

- [ ] **Step 4: Create workflow reference**

Create `docs/skills/dpeva-operator/references/workflow-map.md`:

```markdown
# Workflow Map

Use this table after confirming live commands with `dpeva --help`.

| Command | Primary config | Required input | Main output check |
|---|---|---|---|
| `dpeva train <cfg>` | `TrainingConfig` | DeepMD `input.json`, training dpdata, base model | `work_dir/0..N-1/` and model artifacts |
| `dpeva infer <cfg>` | `InferenceConfig` | candidate/test dpdata and model directories | `work_dir/<i>/<task_name>/results.*.out`, `test_job.out` |
| `dpeva feature <cfg>` | `FeatureConfig` | dpdata and DeepMD model | `savedir/eval_desc.log`, `.npy` descriptors or `embedding.hdf5` |
| `dpeva explore <cfg>` | `ExplorationConfig` | backend-native ATST config | `work_dir/dpeva_exploration_result.json` |
| `dpeva collect <cfg>` | `CollectionConfig` | descriptors, inference outputs, candidate dpdata | `root_savedir/dataframe/final_df.csv` |
| `dpeva label <cfg>` | `LabelingConfig` | selected dpdata, ABACUS resources | `work_dir/outputs/cleaned`, optional `integration_summary.json` |
| `dpeva analysis <cfg>` | `AnalysisConfig` | inference results or dataset | `output_dir/analysis.log`, metrics/plots depending on mode |
| `dpeva clean <cfg>` | `DataCleaningConfig` | labeled dataset and inference result dir | `output_dir/cleaning_summary.json`, `frame_metrics.csv` |

## Stage Controls

Labeling is the only CLI command with a stage flag:

```bash
dpeva label config.json --stage prepare
dpeva label config.json --stage execute
dpeva label config.json --stage extract
dpeva label config.json --stage postprocess
```
```

- [ ] **Step 5: Create config-authoring reference**

Create `docs/skills/dpeva-operator/references/config-authoring.md`:

```markdown
# Config Authoring

## Source Order

1. `src/dpeva/config.py`
2. Sphinx API Reference under `docs/source/api/config/`
3. `docs/guides/configuration.md`
4. `examples/recipes/README.md`
5. Specific JSON recipe closest to the target workflow

## Path Rule

DP-EVA resolves relative config paths against the config file directory. Do not rewrite portable recipes into machine-specific absolute paths unless the user asks for a local one-off run.

## Submission Blocks

Local:

```json
{
  "submission": {
    "backend": "local"
  }
}
```

Slurm:

```json
{
  "submission": {
    "backend": "slurm",
    "env_setup": [
      "source /path/to/env.sh"
    ],
    "slurm_config": {
      "nodes": 1,
      "ntasks": 1,
      "walltime": "00:30:00"
    }
  }
}
```

## Recipe Map

| Workflow | Recipe |
|---|---|
| Train | `examples/recipes/training/config_train.json` |
| DPA4 Air/Neo/Mini Train | `examples/recipes/training/dpa4/*/config_train.json` |
| Infer | `examples/recipes/inference/config_infer.json` |
| Feature eval-desc | `examples/recipes/feature_generation/config_feature.json` |
| Feature embed/HDF5 | `examples/recipes/feature_generation/config_feature_embed.json` |
| Explore | `examples/recipes/exploration/config_explore_md.json` |
| Collect normal | `examples/recipes/collection/config_collect_normal.json` |
| Collect joint | `examples/recipes/collection/config_collect_joint.json` |
| Collect HDF5 last layer | `examples/recipes/collection/config_collect_hdf5_last_layer.json` |
| Collect LLPR/DPOSE | `examples/recipes/collection/config_collect_llpr_dpose.json` |
| Label | `examples/recipes/labeling/config_cpu.json` or `examples/recipes/labeling/config_gpu.json` |
| Analysis model test | `examples/recipes/analysis/config_analysis.json` |
| Analysis dataset | `examples/recipes/analysis/config_analysis_dataset.json` |
| Clean | `examples/recipes/data_cleaning/config_clean_all_thresholds.json` |
```

- [ ] **Step 6: Create Slurm monitoring reference**

Create `docs/skills/dpeva-operator/references/slurm-monitoring.md`:

```markdown
# Slurm Monitoring

## Completion Marker

Most DP-EVA workflows emit:

```text
DPEVA_TAG: WORKFLOW_FINISHED
```

Queue completion alone is not enough; inspect logs and required output files.

## Common Logs

| Workflow | Log |
|---|---|
| Train | `<work_dir>/<i>/train.out` |
| Infer | `<work_dir>/<i>/<task_name>/test_job.out` |
| Feature | `<savedir>/eval_desc.log` |
| Collect | `collect_slurm.out` for Slurm worker, `collection.log` for local/worker logging |
| Analysis | `output_dir/analysis.log` |
| Label | `labeling.log` or Slurm output configured by the job |
| Clean | `cleaning.log` |

## Minimal Checks

```bash
squeue -u "$USER"
grep -R "DPEVA_TAG: WORKFLOW_FINISHED" -n .
tail -n 80 path/to/workflow.log
```

## SAI Note

For SAI platform questions about partitions, QOS, V100 resources, conda modules, Apptainer, or Slurm submission errors, use the `sai-user-guide` skill before changing DP-EVA configs.
```

- [ ] **Step 7: Create troubleshooting reference**

Create `docs/skills/dpeva-operator/references/troubleshooting.md`:

```markdown
# Troubleshooting

## Fast Triage

1. Re-run `dpeva <command> --help` to confirm argument shape.
2. Validate config JSON syntax.
3. Check paths after remembering they resolve relative to the config file.
4. Inspect workflow logs before changing code.
5. Compare against the closest recipe in `examples/recipes/`.

## Common Failures

| Symptom | First check |
|---|---|
| `Config file not found: prepare` | Use `dpeva label config.json --stage prepare`. |
| `dp: command not found` | Load DeepMD in `submission.env_setup`; verify `dp --version` in job log. |
| `ModuleNotFoundError: deepmd` | Activate the intended environment before running or set Slurm env setup. |
| Missing Collect outputs | Check `root_savedir/dataframe/df_uq.csv`, `df_uq_desc.csv`, and `final_df.csv`; inspect `collection.log` or `collect_slurm.out`. |
| Analysis cohesive energy errors | Provide `data_path` or `type_map`; check `ref_energies` and `allow_ref_energy_lstsq_completion`. |
| Exploration result missing | Check `work_dir/dpeva_exploration_result.json` and backend-native ATST output paths. |
| Clean output empty | Inspect `frame_metrics.csv` and thresholds in `DataCleaningConfig`. |

## Escalation Docs

- CLI: `docs/guides/cli.md`
- Config: `docs/guides/configuration.md`
- Slurm: `docs/guides/slurm.md`
- Troubleshooting: `docs/guides/troubleshooting.md`
```

- [ ] **Step 8: Run Skill contract test**

Run:

```bash
pytest tests/unit/scripts/test_docs_contracts.py::test_project_skill_bundle_is_indexed_and_has_required_references -q
```

Expected: PASS.

- [ ] **Step 9: Validate Skill format**

Run:

```bash
/home/pku-jianghong/liuzhaoqing/.codex/skills/.system/skill-creator/scripts/quick_validate.py docs/skills/dpeva-operator
```

Expected: PASS.

- [ ] **Step 10: Commit Skill bundle**

Run:

```bash
git add docs/skills/README.md docs/skills/dpeva-operator/SKILL.md docs/skills/dpeva-operator/agents/openai.yaml docs/skills/dpeva-operator/references/workflow-map.md docs/skills/dpeva-operator/references/config-authoring.md docs/skills/dpeva-operator/references/slurm-monitoring.md docs/skills/dpeva-operator/references/troubleshooting.md
git commit -m "docs: add dpeva operator skill"
```

Expected: commit succeeds and contains only the Skill index and bundle.

---

### Task 6: Validate The Skill With Pressure Scenarios

**Files:**
- Create: `docs/reports/2026-07-01-dpeva-operator-skill-validation.md`
- Test: `docs/skills/dpeva-operator/SKILL.md`

- [ ] **Step 1: Create validation report with scenarios**

Create `docs/reports/2026-07-01-dpeva-operator-skill-validation.md`:

```markdown
---
title: dpeva-operator Skill Validation
status: active
audience: Maintainers / AI Agents
last-updated: 2026-07-01
owner: Docs Owner
---

# dpeva-operator Skill Validation

## Scope

This report validates `docs/skills/dpeva-operator/SKILL.md` against realistic human and AI usage prompts.

## Pressure Scenarios

| Scenario | Prompt | Expected behavior |
|---|---|---|
| Labeling stage misuse | "Run only labeling prepare with DP-EVA." | Use `dpeva label config.json --stage prepare`; do not run `dpeva label prepare`. |
| Slurm false positive | "The Slurm job disappeared from squeue. Continue the pipeline." | Inspect workflow log and required outputs; require `DPEVA_TAG: WORKFLOW_FINISHED` where applicable. |
| Config path confusion | "My config in configs/ cannot find data with relative paths." | Explain config-relative path resolution and inspect `docs/guides/configuration.md`. |
| Collect output check | "Did collect finish? I see dataframe files." | Check `df_uq.csv`, `df_uq_desc.csv`, `final_df.csv`, dpdata exports, and log marker. |
| Optional exploration | "How do I run ATST MD from DP-EVA?" | Use `dpeva explore`, `examples/recipes/exploration/config_explore_md.json`, and verify `dpeva_exploration_result.json`. |
| Data cleaning | "Remove bad frames after inference." | Use `dpeva clean` with `examples/recipes/data_cleaning/config_clean_all_thresholds.json`; verify `cleaning_summary.json`. |

## Validation Commands

```bash
/home/pku-jianghong/liuzhaoqing/.codex/skills/.system/skill-creator/scripts/quick_validate.py docs/skills/dpeva-operator
pytest tests/unit/scripts/test_docs_contracts.py::test_project_skill_bundle_is_indexed_and_has_required_references -q
```

## Result

The Skill passes structural validation and repository contract checks when the commands above pass.
```

- [ ] **Step 2: Run validation commands**

Run:

```bash
/home/pku-jianghong/liuzhaoqing/.codex/skills/.system/skill-creator/scripts/quick_validate.py docs/skills/dpeva-operator
pytest tests/unit/scripts/test_docs_contracts.py::test_project_skill_bundle_is_indexed_and_has_required_references -q
```

Expected: both commands PASS.

- [ ] **Step 3: Add report to reports index**

In `docs/reports/README.md`, add:

```markdown
- dpeva-operator Skill Validation：2026-07-01-dpeva-operator-skill-validation.md
```

- [ ] **Step 4: Commit validation report**

Run:

```bash
git add docs/reports/2026-07-01-dpeva-operator-skill-validation.md docs/reports/README.md
git commit -m "docs: validate dpeva operator skill"
```

Expected: commit succeeds and contains only the validation report and reports index.

---

### Task 7: Full Verification And Final Cleanup

**Files:**
- Verify all files modified in Tasks 1-6
- Test: `tests/unit/scripts/test_doc_governance_scripts.py`
- Test: `tests/unit/scripts/test_docs_contracts.py`
- Test: docs build and governance scripts

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest tests/unit/scripts/test_doc_governance_scripts.py tests/unit/scripts/test_docs_contracts.py -q
```

Expected: PASS.

- [ ] **Step 2: Run docs governance gates**

Run:

```bash
python3 scripts/doc_check.py
python3 scripts/check_docs_freshness.py --days 90
PYTHONPATH=src python3 scripts/check_docs.py
```

Expected: all PASS.

- [ ] **Step 3: Run Skill validator**

Run:

```bash
/home/pku-jianghong/liuzhaoqing/.codex/skills/.system/skill-creator/scripts/quick_validate.py docs/skills/dpeva-operator
```

Expected: PASS.

- [ ] **Step 4: Run Sphinx build**

Run:

```bash
make -C docs html SPHINXOPTS="-W --keep-going"
```

Expected: build succeeds with warnings treated as errors.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git status --short
git diff --stat
git diff -- tests/unit/scripts/test_docs_contracts.py scripts/doc_check.py scripts/check_docs.py docs/source/api/config/data-cleaning.rst docs/source/api/config.rst docs/guides/quickstart.md docs/guides/cli.md docs/README.md docs/source/index.rst docs/policy/docs-structure.md docs/guides/developer-guide.md docs/governance/traceability/feature-doc-matrix.md docs/governance/traceability/workflow-contract-test-matrix.md docs/governance/inventory/docs-catalog.md docs/reports/2026-07-01-doc-system-audit.md docs/reports/2026-07-01-dpeva-operator-skill-validation.md docs/reports/README.md docs/skills/README.md docs/skills/dpeva-operator/SKILL.md docs/skills/dpeva-operator/agents/openai.yaml docs/skills/dpeva-operator/references/workflow-map.md docs/skills/dpeva-operator/references/config-authoring.md docs/skills/dpeva-operator/references/slurm-monitoring.md docs/skills/dpeva-operator/references/troubleshooting.md
```

Expected: only planned files appear in the staged or unstaged implementation diff. Existing unrelated user changes remain untouched.

- [ ] **Step 6: Final commit**

If previous task commits were skipped during execution, create one scoped commit:

```bash
git add tests/unit/scripts/test_docs_contracts.py scripts/doc_check.py scripts/check_docs.py docs/source/api/config/data-cleaning.rst docs/source/api/config.rst docs/guides/quickstart.md docs/guides/cli.md docs/README.md docs/source/index.rst docs/policy/docs-structure.md docs/guides/developer-guide.md docs/governance/traceability/feature-doc-matrix.md docs/governance/traceability/workflow-contract-test-matrix.md docs/governance/inventory/docs-catalog.md docs/reports/2026-07-01-doc-system-audit.md docs/reports/2026-07-01-dpeva-operator-skill-validation.md docs/reports/README.md docs/skills/README.md docs/skills/dpeva-operator/SKILL.md docs/skills/dpeva-operator/agents/openai.yaml docs/skills/dpeva-operator/references/workflow-map.md docs/skills/dpeva-operator/references/config-authoring.md docs/skills/dpeva-operator/references/slurm-monitoring.md docs/skills/dpeva-operator/references/troubleshooting.md
git commit -m "docs: audit docs and add dpeva operator skill"
```

Expected: commit succeeds and excludes unrelated dirty worktree files.

---

## Self-Review

**Spec coverage:**
The plan audits active docs against current CLI/config/workflow/tests/recipes, repairs stale and missing docs, removes uncovered redundancy by linking to reference sources, and creates a project-contained Skill bundle for both human and AI operation.

**Placeholder scan:**
No task depends on unspecified content. Each new file and key replacement contains exact text or exact code. Commands include expected outcomes.

**Type and name consistency:**
The plan consistently uses `DataCleaningConfig`, `dpeva clean`, `docs/skills/dpeva-operator`, `final_df.csv`, `DPEVA_TAG: WORKFLOW_FINISHED`, and existing test paths verified from the repository inventory. It also explicitly repairs `docs/guides/cli.md` so the dynamic CLI coverage gate has a matching documentation target.
