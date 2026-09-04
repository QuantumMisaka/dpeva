import subprocess

from dpeva.submission.guards import guarded_command


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
