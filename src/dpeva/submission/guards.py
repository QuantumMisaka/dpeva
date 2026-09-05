from dpeva.constants import WORKFLOW_FINISHED_TAG


def guarded_command(
    command: str,
    artifact_checks: list[str],
    freshness_globs: list[str] | None = None,
) -> str:
    """Render fail-closed artifact and current-attempt checks around a command.

    ``freshness_globs`` are shell words (quoted literal prefix plus an optional
    wildcard suffix). Each word is an independent required output group.
    Bash arrays keep matching paths whitespace- and newline-safe.
    """

    globs = freshness_globs or []
    lines: list[str] = []
    if globs:
        lines.extend(
            [
                "shopt -s nullglob globstar",
                "declare -A dpeva_attempt_output_baseline=()",
                "dpeva_output_identity() {",
                "    stat -L --printf='%d:%i:%s:%y:%z' -- \"$1\"",
                "}",
                "dpeva_snapshot_output() {",
                "    local artifact=$1 identity",
                "    if ! identity=$(dpeva_output_identity \"$artifact\"); then return 1; fi",
                "    dpeva_attempt_output_baseline[\"$artifact\"]=$identity",
                "}",
                "dpeva_require_fresh_output() {",
                "    local artifact identity previous found=1",
                "    for artifact in \"$@\"; do",
                "        if [[ ! -f \"$artifact\" || ! -s \"$artifact\" ]]; then continue; fi",
                "        if ! identity=$(dpeva_output_identity \"$artifact\"); then return 1; fi",
                "        if [[ -z ${dpeva_attempt_output_baseline[$artifact]+present} ]]; then",
                "            found=0",
                "        else",
                "            previous=${dpeva_attempt_output_baseline[$artifact]}",
                "            if [[ \"$identity\" != \"$previous\" ]]; then found=0; fi",
                "        fi",
                "    done",
                "    return \"$found\"",
                "}",
            ]
        )
        for index, pattern in enumerate(globs):
            lines.extend(
                [
                    f"dpeva_attempt_group_{index}=({pattern})",
                    f"for dpeva_artifact in \"${{dpeva_attempt_group_{index}[@]}}\"; do",
                    "    if [[ -f \"$dpeva_artifact\" ]]; then dpeva_snapshot_output \"$dpeva_artifact\"; fi",
                    "done",
                ]
            )

    lines.extend([command, *artifact_checks])
    for index, pattern in enumerate(globs):
        lines.extend(
            [
                f"dpeva_attempt_group_{index}=({pattern})",
                f"dpeva_require_fresh_output \"${{dpeva_attempt_group_{index}[@]}}\"",
            ]
        )
    lines.append(f'printf "%s\\n" "{WORKFLOW_FINISHED_TAG}"')
    return "\n".join(lines) + "\n"
