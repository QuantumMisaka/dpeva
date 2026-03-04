---
name: "release-helper"
description: "Automates version bumping and documentation synchronization for DP-EVA releases. Invoke when preparing a new release or bumping versions."
---

# DP-EVA Release Helper

This skill automates the release process by:
1.  **Fetching Changes**: Retrieves git commit history since the last version tag.
2.  **Summarizing Updates**: Combines git logs with current session changes to generate a release summary.
3.  **Updating Documentation**: Appends the new release note to `docs/guides/developer-guide.md` following the "Current Era" maintenance policy.
4.  **Bumping Version**: Updates `src/dpeva/__init__.py` and `README.md` metadata.

## Usage

### 1. Prepare Release Information
First, analyze the changes to be included in this release:
- Check git logs since the last tag: `git log --oneline $(git describe --tags --abbrev=0)..HEAD`
- Summarize the key changes (Features, Fixes, Refactors, Docs).

### 2. Execute Release Script
Run the release helper script to bump the version and update files.

To perform a **Patch Release** (e.g., 0.4.5 -> 0.4.6):
```bash
python scripts/release_helper.py patch
```

To perform a **Minor Release** (e.g., 0.4.5 -> 0.5.0):
```bash
python scripts/release_helper.py minor
```

### 3. Update Developer Guide
**CRITICAL**: You must manually update `docs/guides/developer-guide.md` after running the script. The script bumps the version number in the header, but you need to append the detailed release note.

**Action**:
1.  Open `docs/guides/developer-guide.md`.
2.  Locate the `## 6. 版本修订记录` -> `### 6.2 版本历史` -> `#### Current Era (v0.4.x)` 章节。
3.  **Insert** the new version entry at the **TOP** of the list (below the header).
4.  Format the entry as follows:
    ```markdown
    *   **vX.Y.Z** (YYYY-MM-DD):
        *   **[类型]** 变更描述...
        *   **[类型]** 变更描述...
    ```
5.  Ensure you follow the "Append-only" policy defined in section 6.1.

## Post-Execution Steps

1.  Verify that `src/dpeva/__init__.py` and `README.md` have the correct new version.
2.  Verify that `docs/guides/developer-guide.md` contains the new version entry and is correctly formatted.
3.  Commit the changes: `git commit -am "chore: release vX.Y.Z"`
4.  Tag the release: `git tag vX.Y.Z`
