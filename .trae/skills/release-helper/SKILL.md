---
name: "release-helper"
description: "Automates version bumping and documentation synchronization for DP-EVA releases. Invoke when preparing a new release or bumping versions."
---

# DP-EVA Release Helper

This skill automates the release process by:
1.  **Bumping Version**: Updates `src/dpeva/__init__.py` and documentation (e.g., `developer-guide.md`).
2.  **Generating Changelog**: Drafts a new entry in `CHANGELOG.md` based on recent git commits.
3.  **Updating Metadata**: Ensures version consistency across the codebase.

## Usage

To verify the release changes without modifying files (Dry Run):

```bash
python tools/release_helper.py patch --dry-run
```

To perform a **Patch Release** (e.g., 0.4.5 -> 0.4.6):

```bash
python tools/release_helper.py patch
```

To perform a **Minor Release** (e.g., 0.4.5 -> 0.5.0):

```bash
python tools/release_helper.py minor
```

To perform a **Major Release** (e.g., 0.4.5 -> 1.0.0):

```bash
python tools/release_helper.py major
```

## Post-Execution Steps

1.  Review the updated `CHANGELOG.md` and edit the content if necessary.
2.  **Update README Version**: Run `python scripts/update_version.py` to synchronize the version badge in `README.md`.
3.  Commit the changes: `git commit -am "chore: release vX.Y.Z"`
4.  Tag the release: `git tag vX.Y.Z`
