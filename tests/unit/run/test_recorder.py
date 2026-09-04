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
