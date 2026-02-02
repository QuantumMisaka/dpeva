---
name: "update-dev-docs"
description: "Updates the DP-EVA Developer Guide with recent code changes. Invoke when core code (runner, src, tests) is modified or user requests doc updates."
---

# DP-EVA Documentation Updater

This skill assists in keeping the `DP-EVA_Project_Developer_Guide.md` synchronized with the codebase.

## When to Invoke
*   When the user explicitly asks to "update the developer guide" or "document these changes".
*   After significant modifications to `runner/`, `src/dpeva/`, or `test/`.
*   When new configuration parameters or workflow behaviors are introduced.

## Execution Protocol
1.  **Read Documentation**: `read .trae/documents/DP-EVA_Project_Developer_Guide.md`
2.  **Analyze Changes**: Summarize the technical updates (Features, Optimizations, Fixes) based on the recent conversation or code edits.
3.  **Determine Version**:
    *   **Patch (x.x.N+1)**: Bug fixes, minor optimizations, doc updates.
    *   **Minor (x.N+1.0)**: New features, significant refactoring.
4.  **Draft Content**:
    *   **Revision History**: Add a bulleted entry with `[Type] Description`. Format: `* **vX.Y.Z** (YYYY-MM-DD):`
    *   **Body Updates**: Locate relevant sections (e.g., "4.4 Collect", "3.3 UQ") and update text/config examples if changed.
5.  **Apply Update**: Use `SearchReplace` to insert the new content into the markdown file. You SHOULD use Chinese in the document.
