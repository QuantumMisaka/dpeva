---
title: Governance Alignment Implementation Plan
status: proposed
audience: Developers / AI Agents
last-updated: 2026-09-04
owner: Project Maintainer
---

# Governance Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local, CI, documentation, and release gates traceable from one small manifest, then enforce evidence/owner/retirement rules without increasing the number of checks.

**Spec:** `docs/superpowers/specs/2026-09-04-project-governance-and-deepmd-3-2-design.html` (`#development`, `#testing` §13.4, `#docs`, `#requirements` R13–R14, `#rollout` §15.8–15.9)

**Architecture:** Replace duplicated command definitions with a small TOML gate manifest and a Python runner used by both `gate.sh` and GitHub Actions. Keep traceability as checked Markdown/JSON paths, and implement one quarterly audit that reports stale or ownerless rules; it never edits policy automatically.

**Tech Stack:** Python 3.10+, `tomllib` on 3.11+ with the existing `tomli` dev fallback on 3.10, Bash, GitHub Actions, pytest, existing docs checks.

## Global Constraints

- Start final governance alignment after Plans A/B and after Plans C/D have created any gates that must be centralized; do not make Plan E a prerequisite for scientific bug fixes.
- Every new gate must replace duplicate invocation or detect a distinct failure class; gate count is not a success metric (`#rollout` §15.9).
- A green local/CI gate proves only its declared layer; SAI/scientific claims remain separate.
- Governance rules require basis, owner, enforcement path, and review/retirement date; no self-asserted evidence.
- Final merge of governance mechanism changes requires the repository's independent governance review, with cross-family must-attempt semantics.

---

### Task 1: Define one executable gate manifest

**Files:**
- Create: `scripts/gates.toml`
- Create: `scripts/run_gate.py`
- Create: `tests/unit/scripts/test_run_gate.py`

**Test strategy:**
- Behavior boundary: named gates execute argv without shell interpolation, stop on failure, and expose profiles for local, PR, docs, DeepMD contract, and release.
- Existing suite to extend: none; current `gate.sh` hardcodes commands.
- New test file justification: parsing, profile expansion, and subprocess failure propagation form a new reusable boundary.
- Temporary probes: none.

**Interfaces:**
- Consumes: `scripts/gates.toml` and an injected subprocess runner.
- Produces: `Gate`, `load_manifest(path)`, `resolve_profile(name)`, `run_gate(name, runner=subprocess.run) -> int`, and CLI `python scripts/run_gate.py NAME`.

- [ ] **Step 1: Write failing manifest/runner tests**

```python
def test_pr_profile_has_unique_ordered_gates() -> None:
    manifest = load_manifest(Path("scripts/gates.toml"))
    assert resolve_profile(manifest, "pr") == ["lint", "unit", "audit", "explore_import", "explore_cli", "atst_cli"]


def test_runner_uses_argv_and_stops_on_failure(tmp_path) -> None:
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 3 if argv[-1] == "unit" else 0)
    result = run_names(["lint", "unit", "audit"], manifest=fake_manifest(), runner=fake_run)
    assert result == 3
    assert calls == [["tool", "lint"], ["tool", "unit"]]
```

Run: `pytest tests/unit/scripts/test_run_gate.py -q`

Expected: collection fails because `scripts.run_gate` does not exist.

- [ ] **Step 2: Create the minimal manifest**

```toml
schema_version = "1.0"

[gates.lint]
argv = ["ruff", "check", "src", "tests", "scripts"]
layer = "static"
owner = "Project Maintainer"
basis = "developer-guide §1.6.3"

[gates.unit]
argv = ["pytest", "tests/unit", "--cov=src/dpeva", "--cov-branch", "--cov-report=term", "--cov-report=json:build/coverage/coverage-unit.json", "--cov-fail-under=80"]
layer = "unit"
owner = "Workflow Owner"
basis = "SPEC R13 and existing python-quality CI"

[gates.audit]
argv = ["python", "scripts/audit.py", "src/dpeva"]
layer = "static"
owner = "Project Maintainer"
basis = "existing python-quality CI"

[gates.docs]
argv = ["python", "scripts/doc_check.py"]
layer = "docs"
owner = "Docs Owner"
basis = "developer-guide §1.6.3"

[gates.docs_freshness]
argv = ["python", "scripts/check_docs_freshness.py", "--days", "90"]
layer = "docs"
owner = "Docs Owner"
basis = "developer-guide §1.6.3"

[gates.docs_build]
argv = ["make", "-C", "docs", "html", "SPHINXOPTS=-W --keep-going"]
layer = "docs"
owner = "Docs Owner"
basis = "existing docs-check CI"

[gates.docs_artifacts]
argv = ["python", "-c", "from pathlib import Path; assert Path('docs/build/html/index.html').is_file(); assert Path('docs/build/html/api/config.html').is_file()"]
layer = "docs"
owner = "Docs Owner"
basis = "existing docs-check CI"

[gates.docs_linkcheck]
argv = ["make", "-C", "docs", "linkcheck", "SPHINXOPTS=-W --keep-going"]
layer = "docs"
owner = "Docs Owner"
basis = "existing docs-check pull-request job"

[gates.integration]
argv = ["pytest", "tests/integration", "-q"]
layer = "integration"
owner = "Workflow Owner"
basis = "Plan A Phase 0 acceptance"

[gates.deepmd_contract]
argv = ["pytest", "-m", "deepmd_contract", "tests/contract/deepmd", "-q"]
layer = "upstream-contract"
owner = "Compatibility Owner"
basis = "Plan D CPU contract"

[gates.qualification_collect]
argv = ["python", "scripts/validation/collect_deepmd_32_qualification.py", "--job-ref", "build/deepmd-qualification/latest.json", "--require-complete"]
layer = "sai-evidence"
owner = "Compatibility Owner"
basis = "Plan D SAI qualification"

[gates.explore_import]
argv = ["python", "-c", "import dpeva.exploration"]
layer = "optional-extra"
owner = "Exploration Owner"
basis = "existing explore-extra-smoke CI"

[gates.explore_cli]
argv = ["dpeva", "explore", "--help"]
layer = "optional-extra"
owner = "Exploration Owner"
basis = "existing explore-extra-smoke CI"

[gates.atst_cli]
argv = ["atst", "--help"]
layer = "optional-extra"
owner = "Exploration Owner"
basis = "existing explore-extra-smoke CI"

[profiles]
local = ["lint", "unit", "audit"]
pr = ["lint", "unit", "audit", "explore_import", "explore_cli", "atst_cli"]
docs = ["docs", "docs_freshness", "docs_build", "docs_artifacts"]
docs_pr = ["docs", "docs_freshness", "docs_build", "docs_artifacts", "docs_linkcheck"]
integration = ["integration"]
deepmd_release = ["deepmd_contract", "qualification_collect"]
release = ["lint", "unit", "audit", "explore_import", "explore_cli", "atst_cli", "integration", "docs", "docs_freshness", "docs_build", "docs_artifacts", "docs_linkcheck"]
```

`local` is the core developer profile because `[dev]` deliberately excludes the exploration extra. `pr` is the union exercised by hosted jobs after their job-specific installs; the manifest owns the commands even when CI keeps separate jobs. `integration` is explicit and opt-in during normal edits but joins `release`. `deepmd_release` stays separate because it requires exact upstream and SAI evidence; run it only for DeepMD support claims. These entries centralize checks already required by CI/Plans A/D; they do not authorize new checks.

- [ ] **Step 3: Implement safe profile execution**

Import `tomllib`, falling back to `tomli as tomllib` on Python 3.10; validate required keys, unique gate names, non-empty argv lists, and profile references. Execute each gate with `subprocess.run(gate.argv, check=False)` and return the first non-zero code. `--list` prints name/layer/owner/basis without running commands.

```python
@dataclass(frozen=True)
class Gate:
    name: str
    argv: tuple[str, ...]
    layer: str
    owner: str
    basis: str


@dataclass(frozen=True)
class Manifest:
    gates: dict[str, Gate]
    profiles: dict[str, tuple[str, ...]]


def run_names(names: list[str], manifest: Manifest, runner=subprocess.run) -> int:
    for name in names:
        gate = manifest.gates[name]
        result = runner(list(gate.argv), check=False)
        if result.returncode:
            return result.returncode
    return 0
```

- [ ] **Step 4: Run runner tests and commit**

Run: `pytest tests/unit/scripts/test_run_gate.py -q && python scripts/run_gate.py --list`

Expected: tests pass; listed gates match the TOML exactly.

```bash
git add scripts/gates.toml scripts/run_gate.py tests/unit/scripts/test_run_gate.py
git commit -m "feat: centralize quality gate definitions"
```

### Task 2: Make local and CI entry points consume the manifest

**Files:**
- Modify: `scripts/gate.sh`
- Modify: `.github/workflows/python-quality.yml`
- Modify: `.github/workflows/docs-check.yml`
- Modify: `.github/workflows/doc-lint.yml`
- Modify: `tests/unit/scripts/test_run_gate.py`

**Test strategy:**
- Behavior boundary: local and CI invoke manifest gate names; no second copy of a gate's argv remains in workflow YAML or shell.
- Existing suite to extend: gate-runner tests.
- New test file justification: none.
- Temporary probes: none.

**Interfaces:**
- Consumes: `python scripts/run_gate.py`.
- Produces: local `scripts/gate.sh` wrapper and CI jobs that call exact gate names.

- [ ] **Step 1: Add failing alignment tests**

```python
def test_local_gate_delegates_to_manifest() -> None:
    text = Path("scripts/gate.sh").read_text()
    assert 'python scripts/run_gate.py local "$@"' in text
    assert "ruff check" not in text
    assert "pytest tests/unit" not in text


def test_python_quality_jobs_use_gate_names() -> None:
    text = Path(".github/workflows/python-quality.yml").read_text()
    assert "python scripts/run_gate.py lint" in text
    assert "python scripts/run_gate.py unit" in text
    assert "python scripts/run_gate.py audit" in text
    assert "python scripts/run_gate.py explore_import" in text
    assert "python scripts/run_gate.py explore_cli" in text
    assert "python scripts/run_gate.py atst_cli" in text


def test_docs_jobs_use_gate_names() -> None:
    build = Path(".github/workflows/docs-check.yml").read_text()
    lint = Path(".github/workflows/doc-lint.yml").read_text()
    assert "python scripts/run_gate.py docs_build" in build
    assert "python scripts/run_gate.py docs_artifacts" in build
    assert "python scripts/run_gate.py docs_linkcheck" in build
    assert "python scripts/run_gate.py docs" in lint
    assert "python scripts/run_gate.py docs_freshness" in lint
```

Run: `pytest tests/unit/scripts/test_run_gate.py -q`

Expected: FAIL because commands are duplicated.

- [ ] **Step 2: Reduce the local wrapper**

```bash
#!/bin/bash
set -Eeuo pipefail
python scripts/run_gate.py local "$@"
```

Preserve `--strict` only if it maps to a declared `audit_strict` gate; otherwise remove it from both implementation and docs in the same commit.

- [ ] **Step 3: Replace CI run blocks with gate names**

Keep installation, caching, permissions, concurrency, and artifact-upload steps. Replace only command blocks. The unit job creates `build/coverage/` before invoking its gate; the gate command remains the sole owner of pytest/coverage arguments. The docs build job calls `docs_build`, `docs_artifacts`, and the pull-request-only `docs_linkcheck` gate. Do not merge unrelated jobs: separate jobs preserve failure attribution and concurrency.

The replacement run blocks are:

```yaml
- name: Run declared gate
  run: python scripts/run_gate.py lint
```

Use the same block with `unit`, `audit`, `explore_import`, `explore_cli`, `atst_cli`, `docs`, `docs_freshness`, `docs_build`, `docs_artifacts`, and `docs_linkcheck` in their current owning jobs; preserve the existing pull-request condition on linkcheck.

- [ ] **Step 4: Verify alignment**

Run: `pytest tests/unit/scripts/test_run_gate.py -q && bash -n scripts/gate.sh && python scripts/run_gate.py --list`

Expected: all commands exit `0`; no command duplication assertion fails.

- [ ] **Step 5: Commit aligned entry points**

```bash
git add scripts/gate.sh scripts/gates.toml scripts/run_gate.py .github/workflows/python-quality.yml .github/workflows/docs-check.yml .github/workflows/doc-lint.yml tests/unit/scripts/test_run_gate.py
git commit -m "ci: align local and hosted quality gates"
```

### Task 3: Add checked capability-to-evidence traceability

**Files:**
- Create: `docs/governance/traceability/capability-evidence.json`
- Create: `scripts/check_traceability.py`
- Create: `tests/unit/scripts/test_traceability.py`
- Modify: `docs/governance/traceability/feature-doc-matrix.md`
- Modify: `docs/governance/traceability/workflow-contract-test-matrix.md`
- Modify: `scripts/gates.toml`

**Test strategy:**
- Behavior boundary: each public capability maps to existing code, test, recipe/doc, owner, and latest evidence path; checks verify paths/schema, not source strings.
- Existing suite to extend: none; current Markdown matrices are manual.
- New test file justification: machine validation of paths and required fields is a new boundary.
- Temporary probes: none.

**Interfaces:**
- Consumes: repository-relative paths and capability IDs.
- Produces: `check_traceability.py` and gate `traceability`.

- [ ] **Step 1: Write failing traceability tests**

```python
def test_every_entry_has_existing_paths() -> None:
    failures = validate_traceability(Path("docs/governance/traceability/capability-evidence.json"), repo_root=Path.cwd())
    assert failures == []


def test_validation_does_not_require_capability_text_in_source(tmp_path) -> None:
    code = tmp_path / "example.py"
    code.write_text("content intentionally unrelated to capability IDs")
    registry = write_registry(tmp_path, code_paths=["example.py"])
    assert validate_traceability(registry, repo_root=tmp_path) == []
```

Run: `pytest tests/unit/scripts/test_traceability.py -q`

Expected: collection fails because the checker and JSON are absent.

- [ ] **Step 2: Create a small traceability schema**

Each JSON entry has exactly:

```json
{
  "capability_id": "workflow.feature",
  "code_paths": ["src/dpeva/workflows/feature.py"],
  "test_paths": ["tests/unit/workflows/test_feature_workflow_env.py"],
  "documentation_paths": ["docs/guides/cli.md", "examples/recipes/feature_generation/config_feature.json"],
  "owner": "Workflow Owner",
  "evidence_path": "docs/reports/2026-09-04-run-contract-pilot-report.md"
}
```

Include existing public workflows plus new run contract, lineage/eval-card, and DeepMD capability lane only after their owning plan has landed.

- [ ] **Step 3: Implement path/schema validation and add the gate**

The checker loads the JSON list, rejects duplicate IDs/absolute paths/missing owners, and checks each path exists. It never infers behavior from file text. Add:

```toml
[gates.traceability]
argv = ["python", "scripts/check_traceability.py"]
layer = "docs"
owner = "Docs Owner"
basis = "SPEC R13"
```

The validator's path boundary is:

```python
def validate_traceability(path: Path, repo_root: Path) -> list[str]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        capability_id = entry["capability_id"]
        if capability_id in seen:
            failures.append(f"duplicate capability_id: {capability_id}")
        seen.add(capability_id)
        if not entry.get("owner"):
            failures.append(f"missing owner: {capability_id}")
        for field in ("code_paths", "test_paths", "documentation_paths"):
            for value in entry[field]:
                candidate = Path(value)
                if candidate.is_absolute() or not (repo_root / candidate).exists():
                    failures.append(f"invalid {field}: {capability_id}: {value}")
    return failures
```

Append `traceability` to docs and release profiles, not the unit profile.

- [ ] **Step 4: Run and commit traceability checks**

Run: `pytest tests/unit/scripts/test_traceability.py -q && python scripts/check_traceability.py`

Expected: both commands exit `0` and no nonexistent path is accepted.

```bash
git add docs/governance/traceability/capability-evidence.json docs/governance/traceability/feature-doc-matrix.md docs/governance/traceability/workflow-contract-test-matrix.md scripts/check_traceability.py scripts/gates.toml tests/unit/scripts/test_traceability.py
git commit -m "docs: make capability traceability checkable"
```

### Task 4: Add a report-only use-it-or-lose-it audit

**Files:**
- Create: `docs/governance/rules.json`
- Create: `scripts/audit_governance_rules.py`
- Create: `tests/unit/scripts/test_governance_rule_audit.py`
- Modify: `docs/policy/maintenance.md`
- Create: `.github/workflows/governance-audit.yml`

**Test strategy:**
- Behavior boundary: ownerless, basis-free, unenforced, or overdue rules are reported; the audit never edits or deletes governance assets.
- Existing suite to extend: none.
- New test file justification: R14 has no executable owner/basis/retirement check today.
- Temporary probes: none.

**Interfaces:**
- Consumes: a JSON registry with rule ID, owner, basis, enforcement paths, last review, and review interval.
- Produces: human/JSON audit output and a quarterly scheduled artifact.

- [ ] **Step 1: Write failing audit tests**

```python
def test_overdue_rule_is_reported(tmp_path) -> None:
    registry = tmp_path / "rules.json"
    registry.write_text(json.dumps([{
        "rule_id": "R-test", "owner": "Docs Owner", "basis": "incident-1",
        "enforcement_paths": ["scripts/gate.sh"], "last_reviewed": "2025-01-01",
        "review_interval_days": 90
    }]))
    findings = audit_rules(registry, repo_root=Path.cwd(), today=date(2026, 9, 4))
    assert findings[0]["reason"] == "review-overdue"


def test_audit_is_report_only(tmp_path) -> None:
    before = Path("docs/governance/rules.json").read_bytes()
    audit_rules(Path("docs/governance/rules.json"), Path.cwd(), date.today())
    assert Path("docs/governance/rules.json").read_bytes() == before
```

Run: `pytest tests/unit/scripts/test_governance_rule_audit.py -q`

Expected: collection fails because the audit API is absent.

- [ ] **Step 2: Register only active, evidenced rules**

Seed at most eight entries: the enforcement mechanisms introduced by this rollout (stop/go checkpoints, shared gates, evidence traceability, and governance final review) plus any existing rule they replace. Include only rules with concrete enforcement paths. A ninth entry requires deleting/merging an obsolete entry or documenting a genuinely distinct failure class. Do not copy the fourteen SPEC requirements or general prose principles into a second policy registry.

Use this exact record shape for each entry:

```json
{
  "rule_id": "GATE-SINGLE-SOURCE",
  "owner": "Project Maintainer",
  "basis": "SPEC R13",
  "enforcement_paths": ["scripts/gates.toml", "scripts/run_gate.py"],
  "last_reviewed": "2026-09-04",
  "review_interval_days": 90
}
```

- [ ] **Step 3: Implement report-only auditing**

The script validates required fields, verifies enforcement paths, computes overdue review dates, and emits JSON with `active`, `findings`, and `reviewed_at`. Default exit is `0` with findings so the quarterly job reports rather than blocks. `--strict` exits `1` and is used only at release review.

```python
def audit_rules(registry: Path, repo_root: Path, today: date) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for rule in json.loads(registry.read_text(encoding="utf-8")):
        deadline = date.fromisoformat(rule["last_reviewed"]) + timedelta(days=rule["review_interval_days"])
        missing = [path for path in rule["enforcement_paths"] if not (repo_root / path).exists()]
        if missing:
            findings.append({"rule_id": rule["rule_id"], "reason": "missing-enforcement", "detail": ",".join(missing)})
        elif today > deadline:
            findings.append({"rule_id": rule["rule_id"], "reason": "review-overdue", "detail": deadline.isoformat()})
    return findings
```

- [ ] **Step 4: Schedule quarterly reporting**

The workflow runs on manual dispatch and cron `0 3 1 1,4,7,10 *`, uploads the JSON report, and never creates issues, commits, or deletes files automatically.

```yaml
on:
  workflow_dispatch:
  schedule:
    - cron: "0 3 1 1,4,7,10 *"
jobs:
  audit:
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/audit_governance_rules.py --format json --output governance-audit.json
      - uses: actions/upload-artifact@v4
        with:
          name: governance-audit
          path: governance-audit.json
```

- [ ] **Step 5: Run and commit the audit**

Run: `pytest tests/unit/scripts/test_governance_rule_audit.py -q && python scripts/audit_governance_rules.py --format json`

Expected: test suite passes; command exits `0` and lists any current findings without modifying the worktree.

```bash
git add docs/governance/rules.json docs/policy/maintenance.md scripts/audit_governance_rules.py tests/unit/scripts/test_governance_rule_audit.py .github/workflows/governance-audit.yml
git commit -m "feat: audit governance rules for retirement"
```

### Task 5: Close documentation, release, and independent review gates

**Files:**
- Modify: `docs/guides/developer-guide.md`
- Modify: `docs/guides/docs-governance-quickstart.md`
- Modify: `docs/policy/contributing.md`
- Modify: `docs/policy/quality.md`
- Modify: `docs/policy/maintenance.md`
- Modify: `docs/governance/README.md`
- Modify: `docs/superpowers/specs/2026-09-04-project-governance-and-deepmd-3-2-design.html`
- Create: `docs/reports/2026-09-04-project-governance-final-review.md`

**Test strategy:**
- Behavior boundary: stable docs describe the implemented thin governance path once, release gates are reproducible, and final review records its actual reviewer boundary.
- Existing suite to extend: repository documentation checks.
- New test file justification: none.
- Temporary probes: reviewer launcher artifacts stay outside the repository except the final report.

**Interfaces:**
- Consumes: Plans A–D acceptance reports, gate manifest, traceability registry, and governance audit.
- Produces: release-ready documentation and independent governance review report.

- [ ] **Step 1: Remove superseded duplicate instructions**

Keep `AGENTS.md` as the minimal pointer; explain stable workflow only in developer/policy guides. Replace old “completion marker alone advances orchestration” text and any duplicated gate argv with links to the run contract and gate manifest.

Use these canonical statements and link to them elsewhere instead of copying commands:

```markdown
The executable gate catalog is `scripts/gates.toml`; `scripts/run_gate.py` is the only command dispatcher used by local and hosted entry points.

A workflow is complete only when its process/job succeeded, its declared artifacts were verified, and its run manifest reached `finished`. A completion marker alone is insufficient.
```

- [ ] **Step 2: Run the complete local release profile**

Run: `python scripts/run_gate.py release && git diff --check`

Expected: all commands exit `0`; the release profile includes the warning-as-error Sphinx build and link check.

- [ ] **Step 3: Run DeepMD and SAI evidence checks only when claims require them**

Run: `python scripts/run_gate.py deepmd_release`

Expected: exit `0` whenever the release claims DeepMD support; every capability marked supported has exact passing evidence. Experimental/blocked/unsupported capabilities do not block an unrelated software release, for which this profile is not invoked.

- [ ] **Step 4: Attempt the required independent governance review**

Read `~/.agents/GOVERNANCE_REVIEW.md` immediately before review and invoke its current non-recursive launcher. Record reviewer family, reviewed commit/diff, findings, and disposition. If the cross-family backend is unavailable, mark `cross-family review incomplete; follow-up required`; do not label the governance gate passed.

Run: `sed -n '1,260p' ~/.agents/GOVERNANCE_REVIEW.md`

Expected: the current launcher and evidence contract are read immediately before invoking the exact command specified there; the final report records the launcher command and exit status without credentials.

- [ ] **Step 5: Apply accepted findings and rerun affected gates**

Every accepted finding names its changed files and exact rerun command in the final review report. Rejected findings include evidence and rationale; self-review is not acceptance evidence.

Run: `python scripts/run_gate.py release && python scripts/run_gate.py deepmd_release && git diff --check`

Expected: all gates affected by accepted findings exit `0`; if cross-family review is incomplete, the report retains that incomplete disposition even when these local checks pass.

- [ ] **Step 6: Commit governance closure**

```bash
git add docs/guides/developer-guide.md docs/guides/docs-governance-quickstart.md docs/policy/contributing.md docs/policy/quality.md docs/policy/maintenance.md docs/governance/README.md docs/superpowers/specs/2026-09-04-project-governance-and-deepmd-3-2-design.html docs/reports/2026-09-04-project-governance-final-review.md
git commit -m "docs: close project governance rollout"
```

Expected: the final report truthfully states whether the cross-family governance gate passed or remains incomplete.
