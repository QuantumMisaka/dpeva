from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import dpeva.cli as cli
from dpeva.config_migration import MigrationResult, MigrationWarning, migrate_legacy_config
from dpeva.run.context import RunContext, RunOptions, input_identity, source_identity
from dpeva.run.doctor import DoctorCheck, build_doctor_report
from dpeva.run.status import RunState
from dpeva.workflows.feature import FeatureWorkflow


def test_migration_result_keeps_exact_raw_input_and_schema_metadata() -> None:
    raw = {"backend": "local", "data_path": "data"}
    result = MigrationResult(
        normalized={"submission": {"backend": "local"}, "data_path": "data"},
        warnings=(MigrationWarning(field="backend", replacement="submission.backend"),),
        original=raw,
        input_schema_version="1.0",
    )

    assert result.original == raw
    assert result.input_schema_version == "1.0"


def test_cli_config_loader_reads_source_once(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}
    raw = {"data_path": "data", "backend": "local"}

    def load_once(_path):
        calls["count"] += 1
        return raw

    monkeypatch.setattr(cli, "load_json_config", load_once)
    monkeypatch.setattr(cli, "resolve_config_paths", lambda mapping, _path: mapping)

    result = cli.load_and_resolve_config(str(tmp_path / "config.json"))

    assert calls["count"] == 1
    assert result.original == raw
    assert result.normalized["submission"]["backend"] == "local"


def test_schema_version_is_consumed_and_unsupported_version_rejected() -> None:
    result = migrate_legacy_config({"schema_version": "1.0", "data_path": "data"})
    assert result.input_schema_version == "1.0"
    assert "schema_version" not in result.normalized

    with pytest.raises(ValueError, match="schema_version"):
        migrate_legacy_config({"schema_version": "2.0", "data_path": "data"})


def test_context_persists_config_metadata_reference_and_payload(tmp_path: Path) -> None:
    context = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="metadata"),
        {"backend": "local"},
        {"submission": {"backend": "local"}},
        config_metadata={
            "schema_version": "1.0",
            "input_schema_version": "1.0",
            "migration_warnings": [],
        },
    )

    manifest = json.loads((context.run_dir / "run.json").read_text())
    assert manifest["config"]["metadata"] == "config.metadata.json"
    assert "environment" not in manifest
    assert json.loads((context.run_dir / "config.metadata.json").read_text())["schema_version"] == "1.0"


def test_context_always_writes_empty_warning_metadata(tmp_path: Path) -> None:
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="default-metadata"), {}, {})
    metadata_ref = context.recorder.manifest.config["metadata"]
    metadata = json.loads((context.run_dir / metadata_ref).read_text(encoding="utf-8"))
    assert metadata == {
        "schema_version": "1.0",
        "input_schema_version": "1.0",
        "migration_warnings": [],
    }


def test_force_archives_config_metadata_reference(tmp_path: Path) -> None:
    initial = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="metadata-force"),
        {"x": 1},
        {"x": 1},
        config_metadata={"schema_version": "1.0", "input_schema_version": "1.0"},
    )
    previous = json.loads((initial.run_dir / "run.json").read_text())
    forced = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="metadata-force", force=True, reason="metadata retry"),
        {"x": 2},
        {"x": 2},
        config_metadata={"schema_version": "1.0", "input_schema_version": "1.0"},
    )

    archive = json.loads((forced.run_dir / "attempts/attempt-0001.json").read_text())
    assert archive["config"]["metadata"] == previous["config"]["metadata"]
    assert (forced.run_dir / forced.recorder.manifest.config["metadata"]).is_file()


def test_resume_submitted_run_rejects_before_new_context(tmp_path: Path) -> None:
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="submitted"), {}, {})
    context.recorder.transition(RunState.VALIDATED)
    context.recorder.transition(RunState.SUBMITTED)

    with pytest.raises(ValueError, match="submitted.*scheduler"):
        RunContext.create(
            tmp_path,
            "feature",
            RunOptions(run_id="submitted", resume=True),
            {},
            {},
        )


def test_submitted_resume_short_circuits_before_evidence_factories(tmp_path: Path) -> None:
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="submitted-factory"), {}, {})
    context.recorder.transition(RunState.VALIDATED)
    context.recorder.transition(RunState.SUBMITTED)
    before = (context.run_dir / "run.json").read_bytes()
    calls = {"source": 0, "input": 0}

    def source_factory():
        calls["source"] += 1
        raise AssertionError("submitted resume must not probe source")

    def input_factory():
        calls["input"] += 1
        raise AssertionError("submitted resume must not probe inputs")

    with pytest.raises(ValueError, match="submitted.*scheduler"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="submitted-factory", resume=True), {}, {},
            source_factory=source_factory, input_factories=[input_factory],
        )
    assert calls == {"source": 0, "input": 0}
    assert (context.run_dir / "run.json").read_bytes() == before


def test_legacy_resume_uses_implicit_default_metadata_without_publishing_it(tmp_path: Path) -> None:
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="legacy-meta"), {"x": 1}, {"x": 1})
    manifest_path = context.run_dir / "run.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["config"].pop("metadata")
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    resumed = RunContext.create(
        tmp_path, "feature", RunOptions(run_id="legacy-meta", resume=True), {"x": 1}, {"x": 1}
    )
    assert "metadata" not in resumed.recorder.manifest.config
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="configuration metadata"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="legacy-meta", resume=True), {"x": 1}, {"x": 1},
            config_metadata={"schema_version": "1.0", "migration_warnings": ["not-default"]},
        )
    assert manifest_path.read_bytes() == before


def test_legacy_source_without_fingerprint_requires_both_sides_clean(tmp_path: Path) -> None:
    source = {"package_version": "0.8.1", "git_commit": "a" * 40, "dirty": False}
    context = RunContext.create(
        tmp_path, "feature", RunOptions(run_id="legacy-source"), {}, {}, source=source
    )
    context.recorder.transition(RunState.VALIDATED)
    resumed = RunContext.create(
        tmp_path, "feature", RunOptions(run_id="legacy-source", resume=True), {}, {}, source=source
    )
    assert resumed.attempt_id == 2
    before = (context.run_dir / "run.json").read_bytes()
    with pytest.raises(ValueError, match="legacy dirty provenance"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="legacy-source", resume=True), {}, {},
            source={**source, "dirty": True},
        )
    assert (context.run_dir / "run.json").read_bytes() == before

    dirty_manifest = RunContext.create(
        tmp_path, "feature", RunOptions(run_id="legacy-dirty"), {}, {},
        source={**source, "dirty": True},
    )
    dirty_manifest.recorder.transition(RunState.VALIDATED)
    with pytest.raises(ValueError, match="legacy dirty provenance"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="legacy-dirty", resume=True), {}, {}, source=source
        )


def test_failed_late_input_collection_preserves_source_and_prior_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    first.write_bytes(b"first")

    def fail_late():
        raise FileNotFoundError("model input does not exist: external/missing.pt")

    with pytest.raises(FileNotFoundError):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="partial-evidence"), {}, {},
            source_factory=lambda: {"package_version": "0.8.1"},
            input_factories=[
                lambda: input_identity(first, "model", tmp_path),
                fail_late,
            ],
        )
    payload = json.loads(
        (tmp_path / ".dpeva/runs/partial-evidence/run.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["source"] == {"package_version": "0.8.1"}
    assert payload["inputs"][0]["ref"] == "first.pt"
    assert str(tmp_path) not in payload["failure"]["message"]


def test_source_factory_does_not_drop_explicit_inputs(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    explicit_inputs = [input_identity(model, "model", tmp_path)]
    context = RunContext.create(
        tmp_path, "feature", RunOptions(run_id="mixed-evidence"), {}, {},
        inputs=explicit_inputs,
        source_factory=lambda: {"package_version": "0.8.1"},
    )
    assert context.recorder.manifest.inputs == explicit_inputs
    assert context.recorder.manifest.source == {"package_version": "0.8.1"}


def test_explicit_inputs_are_persisted_before_first_factory_and_conflicts_fail(tmp_path: Path) -> None:
    explicit = [{"kind": "dataset", "ref": "data", "identity": "sha256:a"}]
    with pytest.raises(RuntimeError, match="later input failure"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="explicit-before-factory"), {}, {},
            inputs=explicit,
            input_factories=[lambda: (_ for _ in ()).throw(RuntimeError("later input failure"))],
        )
    payload = json.loads(
        (tmp_path / ".dpeva/runs/explicit-before-factory/run.json").read_text(encoding="utf-8")
    )
    assert payload["inputs"] == explicit

    duplicate = RunContext.create(
        tmp_path, "feature", RunOptions(run_id="input-merge"), {}, {}, inputs=explicit,
        input_factories=[lambda: dict(explicit[0])],
    )
    assert duplicate.recorder.manifest.inputs == explicit
    with pytest.raises(ValueError, match="conflicting input identity"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="input-conflict"), {}, {},
            inputs=explicit,
            input_factories=[lambda: {**explicit[0], "identity": "sha256:b"}],
        )
    failed = json.loads(
        (tmp_path / ".dpeva/runs/input-conflict/run.json").read_text(encoding="utf-8")
    )
    assert failed["inputs"] == explicit
    contradictory = [explicit[0], {**explicit[0], "identity": "sha256:c"}]
    with pytest.raises(ValueError, match="conflicting input identity"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="explicit-conflict"), {}, {},
            inputs=contradictory,
        )
    assert not (tmp_path / ".dpeva/runs/explicit-conflict").exists()


def test_resume_rejects_changed_config_and_identity_without_mutating_manifest(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-v1")
    inputs = [input_identity(model, "model", tmp_path)]
    source = {"package_version": "0.8.1", "git_commit": "a" * 40, "dirty": False}
    context = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="resume-compare"),
        {"x": 1},
        {"x": 1},
        source=source,
        inputs=inputs,
    )
    context.recorder.transition(RunState.VALIDATED)
    context.recorder.transition(RunState.RUNNING)
    before = (context.run_dir / "run.json").read_bytes()

    with pytest.raises(ValueError, match="configuration"):
        RunContext.create(
            tmp_path,
            "feature",
            RunOptions(run_id="resume-compare", resume=True),
            {"x": 2},
            {"x": 1},
            source=source,
            inputs=inputs,
        )
    assert (context.run_dir / "run.json").read_bytes() == before

    resumed = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="resume-compare", resume=True),
        {"x": 1},
        {"x": 1},
        source=source,
        inputs=[input_identity(model, "model", tmp_path)],
    )
    assert resumed.attempt_id == 2
    before_resume = (context.run_dir / "run.json").read_bytes()

    model.write_bytes(b"model-v2")
    with pytest.raises(ValueError, match="input identity"):
        RunContext.create(
            tmp_path,
            "feature",
            RunOptions(run_id="resume-compare", resume=True),
            {"x": 1},
            {"x": 1},
            source=source,
            inputs=[input_identity(model, "model", tmp_path)],
        )
    assert (context.run_dir / "run.json").read_bytes() == before_resume


def test_new_run_input_collection_failure_leaves_failed_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RunContext.create(
            tmp_path,
            "feature",
            RunOptions(run_id="missing-input"),
            {},
            {},
            input_factories=[lambda: input_identity(tmp_path / "missing.pt", "model", tmp_path, require_exists=True)],
        )
    payload = json.loads(
        (tmp_path / ".dpeva/runs/missing-input/run.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"


def test_resume_rejects_changed_source_and_dataset_structure(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "type.raw").write_text("0\n")
    source = {"package_version": "0.8.1", "git_commit": "b" * 40, "dirty": False}
    original = {"dataset": "dataset"}
    inputs = [input_identity(dataset, "dataset", tmp_path)]
    context = RunContext.create(
        tmp_path, "feature", RunOptions(run_id="identity-compare"), original, original,
        source=source, inputs=inputs, config_metadata={"schema_version": "1.0", "migration_warnings": []},
    )
    context.recorder.transition(RunState.VALIDATED)
    before = (context.run_dir / "run.json").read_bytes()
    with pytest.raises(ValueError, match="source identity"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="identity-compare", resume=True), original, original,
            source={**source, "git_commit": "c" * 40}, inputs=inputs,
            config_metadata={"schema_version": "1.0", "migration_warnings": []},
        )
    assert (context.run_dir / "run.json").read_bytes() == before
    with pytest.raises(ValueError, match="configuration metadata"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="identity-compare", resume=True), original, original,
            source=source, inputs=inputs,
            config_metadata={"schema_version": "1.0", "migration_warnings": ["changed"]},
        )
    assert (context.run_dir / "run.json").read_bytes() == before
    (dataset / "new.raw").write_text("1\n")
    with pytest.raises(ValueError, match="input identity"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="identity-compare", resume=True), original, original,
            source=source, inputs=[input_identity(dataset, "dataset", tmp_path)],
            config_metadata={"schema_version": "1.0", "migration_warnings": []},
        )
    assert (context.run_dir / "run.json").read_bytes() == before


def test_source_identity_requires_tracked_package_file_and_counts_untracked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "src/dpeva/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source")
    (repo / "src/new.py").write_text("new-v1")
    (repo / ".git").mkdir()
    statuses = {"clean": "", "dirty": "?? src/new.py\n"}
    for label, status in statuses.items():
        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            if command[1:3] == ["rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(command, 0, str(repo), "")
            if command[1:3] == ["ls-files", "--error-unmatch"]:
                return subprocess.CompletedProcess(command, 0, "src/dpeva/__init__.py\n", "")
            if command[1:3] == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, "d" * 40 + "\n", "")
            return subprocess.CompletedProcess(command, 0, status.replace("\n", "\0") + "\0", "")

        identity = source_identity(source, run=run)
        assert identity["git_commit"] == "d" * 40
        assert identity["dirty"] is (label == "dirty")
        assert all(str(repo) not in value for value in identity.values() if isinstance(value, str))
        assert ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"] in calls


def test_source_identity_does_not_claim_enclosing_consumer_repo(tmp_path: Path) -> None:
    consumer = tmp_path / "consumer"
    source = consumer / "vendor/dpeva/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# wheel")
    (consumer / ".git").mkdir()

    def run(command, **kwargs):
        if command[1:3] == ["rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(command, 0, str(consumer), "")
        return subprocess.CompletedProcess(command, 1, "", "not tracked")

    identity = source_identity(source, run=run)
    assert "git_commit" not in identity
    assert "dirty" not in identity


def test_source_identity_ignores_run_evidence_and_fingerprints_other_dirty_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "src/dpeva/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source")
    (repo / ".git").mkdir()
    statuses = {
        "clean": "",
        "evidence-only": "?? .dpeva/runs/current/run.json\n",
        "untracked": "?? src/new.py\n",
        "modified": " M src/dpeva/__init__.py\n",
    }
    identities = {}
    for label, status in statuses.items():
        def run(command, **kwargs):
            if command[1:3] == ["rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(command, 0, str(repo), "")
            if command[1:3] == ["ls-files", "--error-unmatch"]:
                return subprocess.CompletedProcess(command, 0, "src/dpeva/__init__.py\n", "")
            if command[1:3] == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, "e" * 40 + "\n", "")
            return subprocess.CompletedProcess(command, 0, status.replace("\n", "\0") + "\0", "")

        identities[label] = source_identity(source, run=run)
    assert identities["evidence-only"]["dirty"] is False
    assert identities["evidence-only"]["dirty_fingerprint"] == identities["clean"]["dirty_fingerprint"]
    assert identities["untracked"]["dirty"] is True
    assert identities["modified"]["dirty"] is True
    assert identities["untracked"]["dirty_fingerprint"] != identities["modified"]["dirty_fingerprint"]


def test_source_identity_changes_when_same_status_path_content_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "src/dpeva/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# tracked-v1")
    (repo / ".git").mkdir()
    untracked = repo / "src/new.py"
    untracked.write_text("untracked-v1")

    def run(command, **kwargs):
        if command[1:3] == ["rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(command, 0, str(repo), "")
        if command[1:3] == ["ls-files", "--error-unmatch"]:
            return subprocess.CompletedProcess(command, 0, "src/dpeva/__init__.py\n", "")
        if command[1:3] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "1" * 40 + "\n", "")
        return subprocess.CompletedProcess(command, 0, " M src/dpeva/__init__.py\0?? src/new.py\0", "")

    first = source_identity(source, run=run)
    source.write_text("# tracked-v2")
    second = source_identity(source, run=run)
    assert first["dirty_fingerprint"] != second["dirty_fingerprint"]
    untracked.write_text("untracked-v2")
    third = source_identity(source, run=run)
    assert second["dirty_fingerprint"] != third["dirty_fingerprint"]


def test_source_identity_real_git_raw_paths_are_content_sensitive(tmp_path: Path) -> None:
    repo = tmp_path / "real repo"
    source = repo / "src/dpeva/package name-来源.py"
    source.parent.mkdir(parents=True)
    source.write_text("tracked-v1", encoding="utf-8")

    def git(*args: str) -> None:
        result = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    git("config", "core.quotePath", "true")
    git("add", "src/dpeva/package name-来源.py")
    git("commit", "-qm", "initial")
    clean = source_identity(source)
    source.write_text("tracked-v2", encoding="utf-8")
    modified = source_identity(source)
    assert clean["dirty"] is False
    assert modified["dirty"] is True
    assert clean["dirty_fingerprint"] != modified["dirty_fingerprint"]

    untracked = repo / "未追踪 file name.py"
    untracked.write_text("untracked-v1", encoding="utf-8")
    untracked_v1 = source_identity(source)
    untracked.write_text("untracked-v2", encoding="utf-8")
    untracked_v2 = source_identity(source)
    assert untracked_v1["dirty_fingerprint"] != untracked_v2["dirty_fingerprint"]


def test_source_identity_real_git_rename_record_is_consumed(tmp_path: Path) -> None:
    repo = tmp_path / "rename repo"
    old = repo / "src/dpeva/old name.py"
    old.parent.mkdir(parents=True)
    old.write_text("same", encoding="utf-8")

    def git(*args: str) -> None:
        result = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    git("add", "src/dpeva/old name.py")
    git("commit", "-qm", "initial")
    new = repo / "src/dpeva/new name-来源.py"
    git("mv", "src/dpeva/old name.py", "src/dpeva/new name-来源.py")
    identity = source_identity(new)
    assert identity["dirty"] is True
    assert identity["dirty_fingerprint"]


def test_resume_rejects_changed_dirty_source_fingerprint(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "src/dpeva/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source")
    (repo / ".git").mkdir()
    current_status = {"value": ""}

    def run(command, **kwargs):
        if command[1:3] == ["rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(command, 0, str(repo), "")
        if command[1:3] == ["ls-files", "--error-unmatch"]:
            return subprocess.CompletedProcess(command, 0, "src/dpeva/__init__.py\n", "")
        if command[1:3] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "f" * 40 + "\n", "")
        return subprocess.CompletedProcess(command, 0, current_status["value"].replace("\n", "\0") + "\0", "")

    def factory():
        return source_identity(source, run=run)
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="dirty-resume"), {}, {}, source_factory=factory)
    context.recorder.transition(RunState.VALIDATED)
    before = (context.run_dir / "run.json").read_bytes()
    current_status["value"] = "?? src/new.py\n"
    with pytest.raises(ValueError, match="source identity"):
        RunContext.create(
            tmp_path, "feature", RunOptions(run_id="dirty-resume", resume=True), {}, {}, source_factory=factory
        )
    assert (context.run_dir / "run.json").read_bytes() == before


def test_doctor_keeps_optional_hardware_failure_out_of_required_status() -> None:
    report = build_doctor_report(
        checks=[
            DoctorCheck(name="deepmd", status="ok", version="3.2.0", detail="ok"),
            DoctorCheck(name="deepmd.cli.test", status="ok", detail="ok"),
            DoctorCheck(name="cuda", status="missing", detail="not installed", required=False),
        ]
    )

    assert report.status == "ok"


def test_input_identity_is_relative_and_dataset_identity_is_structural(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "type.raw").write_text("0\n")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    context = RunContext.create(
        tmp_path,
        "feature",
        RunOptions(run_id="inputs"),
        {},
        {},
        inputs=[
            {"kind": "dataset", "ref": "data", "identity": "structural-sha256:abc", "identity_scope": "bounded-structural"},
            {"kind": "model", "ref": "model.pt", "identity": "sha256:def", "identity_scope": "full-content"},
        ],
    )

    payload = json.loads((context.run_dir / "run.json").read_text())
    assert all(not value.startswith("/") for item in payload["inputs"] for value in item.values())
    assert payload["inputs"][0]["identity_scope"] == "bounded-structural"


def test_doctor_default_probes_required_operation_surfaces(monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("dpeva.run.doctor._probe_python_package", lambda *a, **k: DoctorCheck(name=a[0], status="ok", detail="ok", required=k.get("required", True)))
    report = build_doctor_report(run=run, include_optional=False)

    names = {check.name for check in report.checks}
    assert {"deepmd", "deepmd.cli.test", "deepmd.cli.eval-desc", "deepmd.cli.embed"} <= names
    assert ["dp", "test", "-h"] in calls
    assert ["dp", "eval-desc", "-h"] in calls
    assert ["dp", "embed", "-h"] in calls


def test_doctor_torch_cuda_probe_is_injectable(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            raise AssertionError("default CUDA probe must not run")

    class FakeTorch:
        cuda = FakeCuda()
        version = type("Version", (), {"cuda": "12.4"})()

    monkeypatch.setattr(
        "dpeva.run.doctor._probe_python_package",
        lambda name, *, required: DoctorCheck(name=name, status="ok", detail="ok", required=required),
    )
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "v3.2.0", "")
    report = build_doctor_report(
        run=run, include_optional=False, torch_module=FakeTorch(), cuda_probe=lambda module: True
    )
    assert report.schema_version == "1.0"
    assert report.status == "ok"
    assert next(check for check in report.checks if check.name == "torch.cuda").status == "ok"


def test_feature_registers_concrete_eval_desc_logs_with_checksums(tmp_path: Path) -> None:
    pool = tmp_path / "pool-000" / "eval_desc"
    pool.mkdir(parents=True)
    (pool / "eval_desc.log").write_text("descriptor output\n")
    (pool / "eval_desc.err").write_text("warning\n")
    context = RunContext.create(tmp_path, "feature", RunOptions(run_id="logs"), {}, {})
    workflow = object.__new__(FeatureWorkflow)
    workflow.output_dir = str(tmp_path)
    workflow._register_existing_logs(context)
    records = context.recorder.manifest.artifacts
    assert {record.path for record in records} == {
        "pool-000/eval_desc/eval_desc.log",
        "pool-000/eval_desc/eval_desc.err",
    }
    assert all(record.checksum and record.status == "verified" for record in records)
