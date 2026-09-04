import json

import pytest
from pydantic import ValidationError

from dpeva.run import InvalidStateTransition, RunEventKind, RunState
from dpeva.run.models import ArtifactRecord, FailureRecord, JobRecord, RunEvent, RunManifest
from dpeva.run.recorder import StatusRecorder


def test_failed_run_is_persisted_atomically(tmp_path) -> None:
    path = tmp_path / "run.json"
    recorder = StatusRecorder.create(path=path, run_id="feature-fixed", workflow="feature")
    recorder.transition(RunState.VALIDATED)
    recorder.transition(RunState.RUNNING)
    recorder.fail(category="EXECUTION", message="dp exited 2")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "failed"
    assert payload["failure"] == {"category": "EXECUTION", "message": "dp exited 2"}
    assert [event["state"] for event in payload["events"]] == ["validated", "running", "failed"]
    assert not (tmp_path / "run.json.tmp").exists()


def test_partial_run_preserves_failure_evidence(tmp_path) -> None:
    recorder = StatusRecorder.create(tmp_path / "run.json", "infer-1", "infer")
    recorder.transition(RunState.VALIDATED)
    recorder.transition(RunState.RUNNING)

    recorder.partial(category="EXECUTION", message="one child failed")

    assert recorder.manifest.status is RunState.PARTIAL
    assert recorder.manifest.failure == FailureRecord(category="EXECUTION", message="one child failed")
    loaded = StatusRecorder.load(tmp_path / "run.json")
    assert loaded.manifest.status is RunState.PARTIAL
    assert loaded.manifest.events[-1].state is RunState.PARTIAL


def test_invalid_transition_is_rejected_without_persisting(tmp_path) -> None:
    path = tmp_path / "run.json"
    recorder = StatusRecorder.create(path, "run-1", "infer")
    original = path.read_text(encoding="utf-8")

    with pytest.raises(InvalidStateTransition):
        recorder.transition(RunState.FINISHED)

    assert path.read_text(encoding="utf-8") == original
    assert recorder.manifest.status is RunState.CREATED
    assert recorder.manifest.events == []


def test_recovery_event_is_serialized_as_event_kind(tmp_path) -> None:
    recorder = StatusRecorder.create(tmp_path / "run.json", "run-1", "infer")
    recorder.transition(RunState.VALIDATED)
    recorder.transition(RunState.RUNNING)
    recorder.fail(category="EXECUTION", message="first attempt failed")
    recorder.transition(RunState.RUNNING, event=RunEventKind.RECOVERY)

    payload = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["events"][-1]["kind"] == "recovery"
    assert payload["events"][-1]["state"] == "running"


def test_manifest_view_isolated_from_recorder_state(tmp_path) -> None:
    path = tmp_path / "run.json"
    recorder = StatusRecorder.create(path, "run-1", "infer")
    view = recorder.manifest
    view.status = RunState.FINISHED
    view.events.append(RunEvent(state=RunState.FINISHED))
    recorder.save()

    assert recorder.manifest.status is RunState.CREATED
    assert recorder.manifest.events == []
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "created"


@pytest.mark.parametrize(
    "manifest",
    [
        {"run_id": "x", "workflow": "infer", "status": "failed"},
        {
            "run_id": "x",
            "workflow": "infer",
            "status": "running",
            "failure": {"category": "EXECUTION", "message": "stale"},
        },
    ],
)
def test_manifest_failure_matches_current_state(manifest) -> None:
    with pytest.raises(ValidationError):
        RunManifest.model_validate(manifest)


def test_loading_malformed_manifest_fails_closed(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"run_id": "x", "workflow": "infer", "mystery": 1}\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        StatusRecorder.load(path)


def test_partial_transition_requires_failure_evidence(tmp_path) -> None:
    recorder = StatusRecorder.create(tmp_path / "run.json", "run-1", "infer")
    recorder.transition(RunState.VALIDATED)
    recorder.transition(RunState.RUNNING)

    with pytest.raises(ValidationError):
        recorder.transition(RunState.PARTIAL)

    assert recorder.manifest.status is RunState.RUNNING
    assert recorder.manifest.failure is None


def test_recovery_clears_current_failure_but_keeps_event_history(tmp_path) -> None:
    recorder = StatusRecorder.create(tmp_path / "run.json", "run-1", "infer")
    recorder.transition(RunState.VALIDATED)
    recorder.transition(RunState.RUNNING)
    recorder.fail(category="EXECUTION", message="first attempt failed")
    recorder.transition(RunState.RUNNING, event=RunEventKind.RECOVERY)

    assert recorder.manifest.status is RunState.RUNNING
    assert recorder.manifest.failure is None
    assert [event.state for event in recorder.manifest.events] == [
        RunState.VALIDATED,
        RunState.RUNNING,
        RunState.FAILED,
        RunState.RUNNING,
    ]


def test_explicit_event_is_persisted_without_changing_state(tmp_path) -> None:
    recorder = StatusRecorder.create(tmp_path / "run.json", "run-1", "infer")
    recorder.record_event(kind="force")

    assert recorder.manifest.status is RunState.CREATED
    assert recorder.manifest.events[-1].kind == "force"
    assert recorder.manifest.events[-1].state is RunState.CREATED


def test_replace_failure_does_not_publish_or_leave_temporary_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "run.json"
    recorder = StatusRecorder.create(path, "run-1", "infer")
    original = path.read_text(encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr("dpeva.run.recorder.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        recorder.transition(RunState.VALIDATED)

    assert recorder.manifest.status is RunState.CREATED
    assert path.read_text(encoding="utf-8") == original
    assert not (tmp_path / "run.json.tmp").exists()


def test_serialization_failure_does_not_publish_or_leave_temporary_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "run.json"
    recorder = StatusRecorder.create(path, "run-1", "infer")
    original = path.read_text(encoding="utf-8")

    def fail_serialization(*args, **kwargs):
        raise TypeError("serialization failed")

    monkeypatch.setattr(RunManifest, "model_dump_json", fail_serialization)
    with pytest.raises(TypeError, match="serialization failed"):
        recorder.transition(RunState.VALIDATED)

    assert recorder.manifest.status is RunState.CREATED
    assert path.read_text(encoding="utf-8") == original
    assert not (tmp_path / "run.json.tmp").exists()


def test_file_fsync_failure_does_not_publish_or_leave_temporary_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "run.json"
    recorder = StatusRecorder.create(path, "run-1", "infer")
    original = path.read_text(encoding="utf-8")

    monkeypatch.setattr("dpeva.run.recorder.os.fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        recorder.transition(RunState.VALIDATED)

    assert recorder.manifest.status is RunState.CREATED
    assert path.read_text(encoding="utf-8") == original
    assert not (tmp_path / "run.json.tmp").exists()


def test_parent_directory_fsync_failure_does_not_publish_state(tmp_path, monkeypatch) -> None:
    path = tmp_path / "run.json"
    recorder = StatusRecorder.create(path, "run-1", "infer")
    original = path.read_text(encoding="utf-8")
    calls = 0
    real_fsync = __import__("os").fsync

    def fail_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr("dpeva.run.recorder.os.fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        recorder.transition(RunState.VALIDATED)

    assert recorder.manifest.status is RunState.CREATED
    assert path.read_text(encoding="utf-8") != original
    assert not (tmp_path / "run.json.tmp").exists()


def test_persistence_fsyncs_parent_directory(tmp_path, monkeypatch) -> None:
    fsync_calls = []
    original_fsync = __import__("os").fsync

    def observe_fsync(fd):
        fsync_calls.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr("dpeva.run.recorder.os.fsync", observe_fsync)
    StatusRecorder.create(tmp_path / "run.json", "run-1", "infer")

    assert len(fsync_calls) == 2


def test_manifest_is_closed_and_nested_records_are_json_serializable() -> None:
    with pytest.raises(ValidationError):
        RunManifest.model_validate(
            {"run_id": "x", "workflow": "infer", "status": "created", "mystery": 1}
        )
    with pytest.raises(ValidationError):
        RunEvent.model_validate({"state": "created", "mystery": 1})

    manifest = RunManifest(
        run_id="run-1",
        workflow="infer",
        jobs=[JobRecord(name="model-0", backend="local", status=RunState.FINISHED)],
        artifacts=[
            ArtifactRecord(
                kind="descriptor",
                path="descriptor.npy",
                producer_run="run-1",
                status="verified",
            )
        ],
    )
    payload = json.loads(manifest.model_dump_json())
    assert payload["status"] == "created"
    assert payload["jobs"][0]["status"] == "finished"
