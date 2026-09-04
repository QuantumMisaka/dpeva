from dpeva.constants import WORKFLOW_FINISHED_TAG


def guarded_command(command: str, artifact_checks: list[str]) -> str:
    """Render a command followed by required checks and its success marker."""
    lines = [command, *artifact_checks, f'printf "%s\\n" "{WORKFLOW_FINISHED_TAG}"']
    return "\n".join(lines) + "\n"
