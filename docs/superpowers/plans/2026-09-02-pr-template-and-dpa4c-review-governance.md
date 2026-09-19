---
title: English PR Template and DPA4C Review Governance Implementation Plan
status: record
audience: Developers / AI Agents
last-updated: 2026-09-02
owner: Project Maintainer
---

# English PR Template and DPA4C Review Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the GitHub PR template with a concise English template and submit an evidence-backed `REQUEST_CHANGES` review for PR #9.

**Spec:** none - requirements supplied directly (user confirmed English template; separate per-PR evidence from major-delivery lifecycle; accept DeepMD-kit 3.2.0 and DPA4C as supported upstream consumption).

**Architecture:** `.github/PULL_REQUEST_TEMPLATE.md` remains the sole GitHub contributor form. It records impact, conditional follow-up, reproducible verification, and risks. A PR review is an external action separate from the template branch.

**Tech Stack:** Markdown, GitHub CLI, project documentation checks.

## Global Constraints

- GitHub-facing template and review prose are English.
- The two pre-existing untracked user plans remain untouched.
- Do not push the template branch or submit the review without the explicitly approved external action.

---

### Task 1: Replace the GitHub PR template with an English evidence form

**Files:**
- Modify: `.github/PULL_REQUEST_TEMPLATE.md`
- Modify: `docs/guides/docs-governance-quickstart.md`
- Modify: `docs/policy/contributing.md`
- Modify: `docs/policy/quality.md`

**Test strategy:**
- Behavior boundary: each PR gets English prompts for impact, conditional documentation/dependency synchronization, verification evidence, and risk; plan/report archival is no longer unconditional boilerplate.
- Existing suite to extend: `scripts/doc_check.py`.
- New test file justification: none; this changes documentation policy only.
- Temporary probes: none.

**Interfaces:**
- Consumes: GitHub automatic template injection and existing documentation policy.
- Produces: four exact headings: `## Change and impact`, `## Required follow-up`, `## Verification`, and `## Risk, migration, and rollback`.

- [ ] **Step 1: Replace the template with the exact English body**

```markdown
## Change and impact

- Summary:
- Public contract impact: None / CLI / configuration / outputs / dependency

## Required follow-up

- [ ] No public-contract change.
- [ ] Updated documentation and recipes (list paths):
- [ ] Updated dependency or runtime-environment guidance (if applicable):

## Verification

- Commands run:
- Results:
- Not covered and why:

## Risk, migration, and rollback

- Compatibility impact:
- Migration steps:
- Rollback:
```

- [ ] **Step 2: Align governance wording**

Change the three governance documents so that substantial architectural, migration, or release work follows the documentation lifecycle, but an ordinary PR needs only the four evidence sections. Retain mandatory documentation synchronization for public contracts and retain the three documentation build/check commands.

- [ ] **Step 3: Run documentation checks**

Run: `python3 scripts/doc_check.py && python3 scripts/check_docs_freshness.py --days 90 && make -C docs html SPHINXOPTS="-W --keep-going"`

Expected: each command exits `0`; Sphinx has no warnings.

- [ ] **Step 4: Review the diff**

Run: `git diff --check && git diff -- .github/PULL_REQUEST_TEMPLATE.md docs/guides/docs-governance-quickstart.md docs/policy/contributing.md docs/policy/quality.md`

Expected: no whitespace errors; no policy still requires an archival plan/report checkbox in every PR.

- [ ] **Step 5: Commit**

Run: `git add .github/PULL_REQUEST_TEMPLATE.md docs/guides/docs-governance-quickstart.md docs/policy/contributing.md docs/policy/quality.md && git commit -m "docs: simplify English pull request template"`

Expected: one commit containing only the template and its matching governance policy updates.

### Task 2: Submit the scoped `REQUEST_CHANGES` review for PR #9

**Files:**
- Modify: no repository files.

**Test strategy:**
- Behavior boundary: PR #9 receives actionable changes required by the accepted DeepMD-kit 3.2/DPA4C baseline, without rejecting shared `pt-expt` support itself.
- Existing suite to extend: none; GitHub review state is the observable result.
- New test file justification: none; this is an external review action.
- Temporary probes: a temporary review Markdown file outside the repository, removed after submission.

**Interfaces:**
- Consumes: GitHub PR #9, DeepMD-kit v3.2.0 primary docs/source, and DP-EVA's centralized config contract.
- Produces: a GitHub review with state `REQUEST_CHANGES`.

- [ ] **Step 1: Recheck PR head and CI immediately before submission**

Run: `gh pr view 9 --repo QuantumMisaka/dpeva --json headRefOid,files,url && gh pr checks 9 --repo QuantumMisaka/dpeva`

Expected: review the captured head; if it differs materially from `4d9b848ebbcdd179ab054bc97a15e62a34369448`, re-review before submission.

- [ ] **Step 2: Submit a review requesting these changes**

1. Complete the DeepMD-kit 3.2 baseline: dependency range, `MIN_DEEPMD_VERSION`, affected recipes/integration fixtures, and SAI runtime guidance.
2. Add centralized FeatureConfig validation: allow missing head for single-task CLI extraction; reject current unvalidated `pt-expt + embed` and `pt-expt + python` combinations; document it in `docs/reference/validation.md`.
3. Test the shared backend contract across existing train/freeze, infer/test, and feature command paths; add a real DPA4C `eval-desc -> DIRECT loader` smoke test with declared fixture/skip semantics.
4. Update the PR body to the new template and accurately state CI status and any coverage gaps.

Run: `gh pr review 9 --repo QuantumMisaka/dpeva --request-changes --body-file <temporary-review-file>`

Expected: GitHub succeeds and the review state is `CHANGES_REQUESTED`.

- [ ] **Step 3: Verify the posted review**

Run: `gh pr view 9 --repo QuantumMisaka/dpeva --json url,reviews`

Expected: output contains the newly submitted review; do not modify the PR branch.

---

## Execution record (2026-09-03 回写)

- **Task 1 完成**：模板分支已 rebase 至 `4820d24`，push 后经 **PR #10** 合并（CI doc-lint/build-docs 通过，Sphinx 由 CI `build-docs` 覆盖）。`main` = `b637ca0`。
- **Task 2 完成（2026-09-02）**：PR #9 收到两轮 `REQUEST_CHANGES`（广谱→收窄），贡献者补 `0ad5538`（拒绝 pt-expt+embed）后于 9/2 15:02 **合入**。
- **遗留项（转 v2.2 任务跟踪器）**：PR#9 第一轮审阅中未落实的 DeepMD-kit 3.2.0 基线项（依赖范围 / `MIN_DEEPMD_VERSION` / recipes 与 SAI runtime 指引仍写 3.1.2）→ 见 `../../../../docs/tasks/ft2dp-v2.2.md` P4。
- 关联记录：`STORAGE_RESCUE_20260903.md`（工作区根）、`docs/superpowers/plans/2026-07-10-*` 与 `2026-07-11-*` 计划的状态回写。
- **当前状态来源**：本计划的原始 checkbox 仅记录当时设计；后续状态、验收和未决项以工作区 `../../../../docs/tasks/ft2dp-v2.2.md` 为准。
