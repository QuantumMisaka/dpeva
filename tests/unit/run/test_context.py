import hashlib
import json
import re

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
    assert json.loads((forced.run_dir / "config.resolved.json").read_text()) == {"x": 2}


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
