import errno
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
    assert recorder.manifest.events[2].failure == FailureRecord(
        category="EXECUTION", message="first attempt failed"
    )
    loaded = StatusRecorder.load(tmp_path / "run.json")
    assert loaded.manifest.events[2].failure == FailureRecord(
        category="EXECUTION", message="first attempt failed"
    )
    assert [event.state for event in recorder.manifest.events] == [
        RunState.VALIDATED,
        RunState.RUNNING,
        RunState.FAILED,
        RunState.RUNNING,
    ]


@pytest.mark.parametrize(
    ("status", "event_kind"),
    [("failed", RunEventKind.RECOVERY), ("partial", RunEventKind.RESUME)],
)
def test_loaded_legacy_terminal_event_is_enriched_before_recovery(
    tmp_path, status, event_kind
) -> None:
    path = tmp_path / f"legacy-{status}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": f"legacy-{status}",
                "workflow": "infer",
                "status": status,
                "events": [{"state": status, "kind": "transition", "attempt_id": 1}],
                "failure": {"category": "EXECUTION", "message": f"legacy {status}"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recorder = StatusRecorder.load(path)
    assert recorder.manifest.events[-1].failure is None
    recorder.transition(RunState.RUNNING, event=event_kind)

    loaded = StatusRecorder.load(path)
    assert loaded.manifest.failure is None
    assert loaded.manifest.events[0].failure == FailureRecord(
        category="EXECUTION", message=f"legacy {status}"
    )
    assert json.loads(path.read_text(encoding="utf-8"))["events"][0]["failure"] == {
        "category": "EXECUTION",
        "message": f"legacy {status}",
    }


def test_legacy_recovery_enriches_transition_not_force_event(tmp_path) -> None:
    path = tmp_path / "legacy-force.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "legacy-force",
                "workflow": "infer",
                "status": "failed",
                "events": [
                    {"state": "failed", "kind": "transition", "attempt_id": 1},
                    {"state": "failed", "kind": "force", "attempt_id": 1},
                ],
                "failure": {"category": "EXECUTION", "message": "legacy failure"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recorder = StatusRecorder.load(path)
    recorder.transition(RunState.RUNNING, event=RunEventKind.RECOVERY)
    events = recorder.manifest.events

    assert events[0].failure == FailureRecord(category="EXECUTION", message="legacy failure")
    assert events[1].failure is None


def test_legacy_recovery_without_transition_adds_unambiguous_terminal_event(tmp_path) -> None:
    path = tmp_path / "legacy-no-transition.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "legacy-no-transition",
                "workflow": "infer",
                "status": "failed",
                "events": [{"state": "failed", "kind": "force", "attempt_id": 1}],
                "failure": {"category": "EXECUTION", "message": "legacy failure"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recorder = StatusRecorder.load(path)
    recorder.transition(RunState.RUNNING, event=RunEventKind.RECOVERY)
    events = recorder.manifest.events

    assert events[0].kind == "force"
    assert events[0].failure is None
    assert events[1].kind == "transition"
    assert events[1].state is RunState.FAILED
    assert events[1].failure == FailureRecord(category="EXECUTION", message="legacy failure")


@pytest.mark.parametrize("status", [RunState.FAILED, RunState.PARTIAL])
def test_record_event_on_current_failure_attaches_failure_evidence(tmp_path, status) -> None:
    recorder = StatusRecorder.create(tmp_path / f"{status.value}.json", "run-1", "infer")
    recorder.transition(RunState.VALIDATED)
    recorder.transition(RunState.RUNNING)
    if status is RunState.FAILED:
        recorder.fail(category="EXECUTION", message="child failed")
    else:
        recorder.partial(category="EXECUTION", message="child incomplete")

    recorder.record_event(kind="force")

    assert recorder.manifest.events[-1].state is status
    assert recorder.manifest.events[-1].failure == recorder.manifest.failure


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

    assert recorder.manifest.status is RunState.VALIDATED
    assert path.read_text(encoding="utf-8") != original
    assert not (tmp_path / "run.json.tmp").exists()

    recorder.transition(RunState.RUNNING)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [event["state"] for event in payload["events"]] == ["validated", "running"]


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.ENOENT, errno.EMFILE])
def test_directory_open_errors_are_not_silenced(tmp_path, monkeypatch, error_number) -> None:
    recorder = StatusRecorder.create(tmp_path / "run.json", "run-1", "infer")

    monkeypatch.setattr(
        "dpeva.run.recorder.os.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(error_number, "directory open failed")),
    )
    with pytest.raises(OSError) as exc_info:
        recorder.transition(RunState.VALIDATED)
    assert exc_info.value.errno == error_number
    assert recorder.manifest.status is RunState.VALIDATED


def test_known_unsupported_directory_open_is_safe(tmp_path, monkeypatch) -> None:
    recorder = StatusRecorder.create(tmp_path / "run.json", "run-1", "infer")
    monkeypatch.setattr(
        "dpeva.run.recorder.os.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(errno.EINVAL, "unsupported")),
    )

    recorder.transition(RunState.VALIDATED)
    assert recorder.manifest.status is RunState.VALIDATED


def test_known_unsupported_directory_fsync_is_safe(tmp_path, monkeypatch) -> None:
    recorder = StatusRecorder.create(tmp_path / "run.json", "run-1", "infer")
    calls = 0
    real_fsync = __import__("os").fsync

    def unsupported_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EINVAL, "unsupported")
        return real_fsync(fd)

    monkeypatch.setattr("dpeva.run.recorder.os.fsync", unsupported_directory_fsync)
    recorder.transition(RunState.VALIDATED)

    assert recorder.manifest.status is RunState.VALIDATED


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
    with pytest.raises(ValidationError):
        RunEvent.model_validate(
            {
                "state": "running",
                "failure": {"category": "EXECUTION", "message": "invalid"},
            }
        )

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
