import subprocess

from dpeva.submission.guards import guarded_command
from dpeva.submission.templates import JobConfig, TemplateEngine


def test_failed_command_does_not_emit_finished(tmp_path) -> None:
    script = tmp_path / "fail.sh"
    script.write_text("#!/bin/bash\nset -Eeuo pipefail\n" + guarded_command("false", []))
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "DPEVA_TAG: WORKFLOW_FINISHED" not in result.stdout


def test_missing_artifact_does_not_emit_finished(tmp_path) -> None:
    check = f"test -s {tmp_path / 'missing.out'}"
    result = subprocess.run(
        ["bash", "-c", "set -Eeuo pipefail\n" + guarded_command("true", [check])],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "DPEVA_TAG: WORKFLOW_FINISHED" not in result.stdout


def test_slurm_directives_precede_fail_closed_shell_setup() -> None:
    script = TemplateEngine.from_default("slurm").render(JobConfig(command="true"))
    lines = script.splitlines()
    last_directive = max(i for i, line in enumerate(lines) if line.startswith("#SBATCH"))
    strict_setup = lines.index("set -Eeuo pipefail")
    assert last_directive < strict_setup
