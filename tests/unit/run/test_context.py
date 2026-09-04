import hashlib
import json
import re
import subprocess
import sys

import pytest
from pydantic import ValidationError

from dpeva.run import RunState
from dpeva.run.context import RunContext, RunOptions


def test_existing_run_id_is_not_overwritten(tmp_path) -> None:
    options = RunOptions(run_id="feature-fixed")
    original = RunContext.create(tmp_path, "feature", options, {"x": 1}, {"x": 1})

    with pytest.raises(FileExistsError):
        RunContext.create(tmp_path, "feature", options, {"x": 2}, {"x": 2})

    assert json.loads((original.run_dir / "config.original.json").read_text()) == {"x": 1}


def test_generated_run_id_is_unique_and_safe(tmp_path) -> None:
    first = RunContext.create(tmp_path, "feature", RunOptions(), {}, {})
    second = RunContext.create(tmp_path, "feature", RunOptions(), {}, {})

    assert first.run_id != second.run_id
    assert re.fullmatch(r"feature-\d{8}T\d{6}Z-[0-9a-f]{6}", first.run_id)
    assert first.run_dir.parent == tmp_path / ".dpeva" / "runs"


@pytest.mark.parametrize("run_id", ["../escape", "nested/id", "", ".", "..", "bad id"])
def test_run_id_cannot_escape_run_root(tmp_path, run_id) -> None:
    with pytest.raises(ValueError):
        RunContext.create(tmp_path, "feature", RunOptions(run_id=run_id), {}, {})


def test_options_reject_ambiguous_resume_and_force() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RunOptions(run_id="run", resume=True, force=True)
    with pytest.raises(ValueError, match="run-id"):
        RunOptions(resume=True)
    with pytest.raises(ValueError, match="reason"):
        RunOptions(run_id="run", force=True)


def test_resume_increments_attempt_and_records_event(tmp_path) -> None:
    initial = RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {"x": 1}, {"x": 1})
    initial.recorder.transition(RunState.VALIDATED)
    initial.recorder.transition(RunState.RUNNING)

    resumed = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="run", resume=True),
        {"x": 2},
        {"x": 2},
    )

    assert resumed.attempt_id == 2
    assert resumed.recorder.manifest.status is RunState.RUNNING
    event = resumed.recorder.manifest.events[-1]
    assert event.kind == "resume"
    assert event.attempt_id == 2
    assert json.loads((resumed.run_dir / "config.original.json").read_text()) == {"x": 1}


def test_resume_rejects_terminal_run(tmp_path) -> None:
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {}, {})
    context.recorder.transition(RunState.VALIDATED)
    context.recorder.transition(RunState.RUNNING)
    context.recorder.transition(RunState.FINISHED)

    with pytest.raises(ValueError, match="terminal"):
        RunContext.create(
            tmp_path,
            "feature",
            RunOptions(run_id="run", resume=True),
            {},
            {},
        )


def test_force_archives_previous_manifest_and_records_attempt(tmp_path) -> None:
    initial = RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {"x": 1}, {"x": 1})
    initial.recorder.transition(RunState.VALIDATED)
    initial.recorder.transition(RunState.RUNNING)
    previous = (initial.run_dir / "run.json").read_text()

    forced = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="run", force=True, reason="rerun after corrected input"),
        {"x": 2},
        {"x": 2},
    )

    archive = forced.run_dir / "attempts" / "attempt-0001.json"
    assert archive.read_text() == previous
    assert forced.attempt_id == 2
    assert forced.recorder.manifest.status is RunState.CREATED
    event = forced.recorder.manifest.events[-1]
    assert event.kind == "force"
    assert event.attempt_id == 2
    assert event.reason == "rerun after corrected input"
    assert json.loads((forced.run_dir / "config.resolved.json").read_text()) == {"x": 1}
    assert json.loads((forced.run_dir / "config.original.attempt-0002.json").read_text()) == {"x": 2}
    assert json.loads((forced.run_dir / "config.original.json").read_text()) == {"x": 1}


def test_force_rejects_malformed_existing_manifest_without_overwrite(tmp_path) -> None:
    run_dir = tmp_path / ".dpeva" / "runs" / "run"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "run.json"
    manifest.write_text('{"run_id":"run","workflow":"feature","mystery":1}\n')

    with pytest.raises(ValidationError):
        RunContext.create(
            tmp_path,
            "feature",
            RunOptions(run_id="run", force=True, reason="retry"),
            {},
            {},
        )
    assert manifest.read_text() == '{"run_id":"run","workflow":"feature","mystery":1}\n'
    assert not (run_dir / "attempts").exists()


def test_register_verified_artifacts_records_relative_path_and_streaming_hash(tmp_path) -> None:
    artifact = tmp_path / "outputs" / "features.npy"
    artifact.parent.mkdir()
    payload = b"scientific evidence\n"
    artifact.write_bytes(payload)
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {}, {})

    context.register_verified_artifacts("feature", [artifact])

    record = context.recorder.manifest.artifacts[-1]
    assert record.kind == "feature"
    assert record.path == "outputs/features.npy"
    assert record.producer_run == "run"
    assert record.status == "verified"
    assert record.checksum == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("path_kind", ["outside", "directory", "empty"])
def test_register_verified_artifacts_rejects_unverifiable_paths(tmp_path, path_kind) -> None:
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {}, {})
    if path_kind == "outside":
        path = tmp_path.parent / "outside.bin"
        path.write_bytes(b"outside")
    elif path_kind == "directory":
        path = tmp_path / "directory"
        path.mkdir()
    else:
        path = tmp_path / "empty.bin"
        path.touch()

    with pytest.raises(ValueError):
        context.register_verified_artifacts("feature", [path])

    assert context.recorder.manifest.artifacts == []


def test_contained_symlink_is_verified_against_its_resolved_target(tmp_path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"inside")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {}, {})

    context.register_verified_artifacts("feature", [link])

    record = context.recorder.manifest.artifacts[-1]
    assert record.path == "target.bin"
    assert record.checksum == hashlib.sha256(b"inside").hexdigest()


def test_escape_symlink_is_rejected(tmp_path) -> None:
    outside = tmp_path.parent / "outside-target.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "escape.bin"
    link.symlink_to(outside)
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {}, {})

    with pytest.raises(ValueError, match="outside"):
        context.register_verified_artifacts("feature", [link])


def test_batch_artifact_persistence_failure_registers_nothing(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {}, {})

    def fail_replace(source, destination):
        if str(destination).endswith("run.json"):
            raise OSError("injected publication failure")
        return original_replace(source, destination)

    import dpeva.run.recorder as recorder_module

    original_replace = recorder_module.os.replace
    monkeypatch.setattr(recorder_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication"):
        context.register_verified_artifacts("feature", [first, second])

    assert context.recorder.manifest.artifacts == []
    assert json.loads((context.run_dir / "run.json").read_text())["artifacts"] == []


def test_concurrent_force_allocates_unique_attempts(tmp_path) -> None:
    RunContext.create(tmp_path, "feature", RunOptions(run_id="run"), {}, {})
    worker = (
        "from dpeva.run.context import RunContext, RunOptions; "
        "print(RunContext.create(__import__('sys').argv[1], 'feature', "
        "RunOptions(run_id='run', force=True, reason='concurrent retry'), "
        "{'worker': True}, {'worker': True}).attempt_id)"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=30) for process in processes]
    assert all(process.returncode == 0 for process in processes), results
    attempts = [int(stdout.strip()) for stdout, _ in results]

    assert sorted(attempts) == [2, 3]
    payload = json.loads((tmp_path / ".dpeva/runs/run/run.json").read_text())
    assert [event["attempt_id"] for event in payload["events"]] == [3]
    assert payload["events"][0]["reason"] == "concurrent retry"
    assert (tmp_path / ".dpeva/runs/run/attempts/attempt-0001.json").exists()
    assert (tmp_path / ".dpeva/runs/run/attempts/attempt-0002.json").exists()


def test_non_json_config_fails_closed_without_removing_run(tmp_path) -> None:
    with pytest.raises(TypeError):
        RunContext.create(
            tmp_path,
            "feature",
            RunOptions(run_id="bad-config"),
            {"unsupported": {"set"}},
            {},
        )

    run_dir = tmp_path / ".dpeva" / "runs" / "bad-config"
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "CONFIG"
    assert "set" in payload["failure"]["message"]
