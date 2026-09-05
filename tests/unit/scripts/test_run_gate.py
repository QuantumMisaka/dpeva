from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.run_gate import Gate, Manifest, load_manifest, resolve_profile, run_names


MANIFEST = Path("scripts/gates.toml")


def test_pr_profile_has_unique_ordered_gates() -> None:
    manifest = load_manifest(MANIFEST)

    assert resolve_profile(manifest, "pr") == [
        "lint",
        "unit",
        "audit",
        "explore_import",
        "explore_cli",
        "atst_cli",
    ]


def test_runner_uses_argv_and_stops_on_failure(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 3 if argv[-1] == "unit" else 0)

    manifest = Manifest(
        gates={
            name: Gate(name=name, argv=("tool", name), layer="test", owner="owner", basis="basis")
            for name in ("lint", "unit", "audit")
        },
        profiles={},
        repo_root=tmp_path,
    )

    result = run_names(["lint", "unit", "audit"], manifest=manifest, runner=fake_run)

    assert result == 3
    assert [argv for argv, _ in calls] == [["tool", "lint"], ["tool", "unit"]]
    assert all(kwargs["check"] is False for _, kwargs in calls)
    assert all(kwargs["shell"] is False for _, kwargs in calls)
    assert all(kwargs["cwd"] == tmp_path for _, kwargs in calls)


def test_manifest_rejects_unknown_gate_and_top_level_fields(tmp_path: Path) -> None:
    unknown_gate = tmp_path / "unknown-gate.toml"
    unknown_gate.write_text(
        'schema_version = "1.0"\n\n[gates.example]\nargv = ["true"]\nlayer = "test"\nowner = "owner"\nbasis = "basis"\nextra = "nope"\n\n[profiles]\nlocal = ["example"]\n',
        encoding="utf-8",
    )
    unknown_top = tmp_path / "unknown-top.toml"
    unknown_top.write_text(
        'schema_version = "1.0"\nunknown = true\n\n[gates.example]\nargv = ["true"]\nlayer = "test"\nowner = "owner"\nbasis = "basis"\n\n[profiles]\nlocal = ["example"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown"):
        load_manifest(unknown_gate)
    with pytest.raises(ValueError, match="unknown"):
        load_manifest(unknown_top)


@pytest.mark.parametrize(
    "argv",
    [[], [""], ["tool", ""], ["tool", 3], [True]],
)
def test_manifest_rejects_invalid_argv(tmp_path: Path, argv: list[object]) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        'schema_version = "1.0"\n\n[gates.example]\n'
        f"argv = {argv!r}\n"
        'layer = "test"\nowner = "owner"\nbasis = "basis"\n\n'
        '[profiles]\nlocal = ["example"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_manifest(path)


def test_manifest_rejects_duplicate_profile_refs(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-profile-ref.toml"
    path.write_text(
        'schema_version = "1.0"\n\n[gates.example]\nargv = ["true"]\n'
        'layer = "test"\nowner = "owner"\nbasis = "basis"\n\n'
        '[profiles]\nlocal = ["example", "example"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)


def test_manifest_rejects_gate_profile_name_collision(tmp_path: Path) -> None:
    path = tmp_path / "gate-profile-collision.toml"
    path.write_text(
        'schema_version = "1.0"\n\n[gates.same]\nargv = ["true"]\n'
        'layer = "test"\nowner = "owner"\nbasis = "basis"\n\n'
        '[profiles]\nsame = ["same"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collision"):
        load_manifest(path)


def test_profile_and_gate_can_be_resolved_and_missing_names_fail() -> None:
    manifest = Manifest(
        gates={"one": Gate("one", ("true",), "test", "owner", "basis")},
        profiles={"local": ("one",)},
        repo_root=Path.cwd(),
    )

    assert resolve_profile(manifest, "one") == ["one"]
    assert resolve_profile(manifest, "local") == ["one"]
    with pytest.raises(KeyError):
        resolve_profile(manifest, "missing")


def test_missing_executable_returns_nonzero_without_shell(tmp_path: Path) -> None:
    manifest = Manifest(
        gates={"missing": Gate("missing", ("definitely-not-an-executable",), "test", "owner", "basis")},
        profiles={},
        repo_root=tmp_path,
    )

    assert run_names(["missing"], manifest=manifest) != 0


def test_python310_tomli_fallback_is_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_name = "scripts._run_gate_tomli_fallback_test"
    module_path = Path(__file__).parents[3] / "scripts" / "run_gate.py"
    parsed = {
        "schema_version": "1.0",
        "gates": {
            "example": {
                "argv": ["true"],
                "layer": "test",
                "owner": "owner",
                "basis": "basis",
            }
        },
        "profiles": {"local": ["example"]},
    }
    fake_tomli = types.ModuleType("tomli")
    fake_tomli.TOMLDecodeError = ValueError
    fake_tomli.load = lambda _handle: parsed
    monkeypatch.setitem(sys.modules, "tomli", fake_tomli)

    original_import = builtins.__import__

    def import_without_tomllib(name: str, *args: object, **kwargs: object) -> object:
        if name == "tomllib":
            raise ModuleNotFoundError("simulated Python 3.10 environment")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tomllib)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    fallback_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, fallback_module)
    spec.loader.exec_module(fallback_module)

    assert fallback_module.tomllib is fake_tomli
    source = tmp_path / "manifest.toml"
    source.write_text("ignored by fake parser", encoding="utf-8")
    assert fallback_module.resolve_profile(fallback_module.load_manifest(source), "local") == ["example"]


def test_local_gate_delegates_to_manifest() -> None:
    text = Path("scripts/gate.sh").read_text(encoding="utf-8")

    assert 'python scripts/run_gate.py local "$@"' in text
    assert "ruff check" not in text
    assert "pytest tests/unit" not in text


def test_python_quality_jobs_use_gate_names() -> None:
    text = Path(".github/workflows/python-quality.yml").read_text(encoding="utf-8")

    for name in ("lint", "unit", "audit", "explore_import", "explore_cli", "atst_cli"):
        assert f"python scripts/run_gate.py {name}" in text


def test_ci_profiles_keep_routine_integration_and_split_deepmd_contracts() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.gates["deepmd_contract"].argv == (
        "pytest",
        "-m",
        "deepmd_contract",
        "tests/contract/deepmd/test_cli_contract.py",
        "-q",
    )
    assert manifest.gates["deepmd_dpa4c_contract"].argv == (
        "pytest",
        "-m",
        "deepmd_contract",
        "tests/contract/deepmd/test_dpa4c_eval_desc.py",
        "-q",
    )
    assert resolve_profile(manifest, "integration") == ["integration_tests"]
    release = resolve_profile(manifest, "release")
    assert "unit" in release
    assert "integration_tests" in release
    assert "qualification_collect" not in release


def test_traceability_is_a_docs_and_release_gate_only() -> None:
    manifest = load_manifest(MANIFEST)

    assert "traceability" in manifest.profiles["docs"]
    assert "traceability" in manifest.profiles["docs_pr"]
    assert "traceability" in manifest.profiles["release"]
    for profile in ("local", "pr", "integration", "deepmd_release"):
        assert "traceability" not in manifest.profiles[profile]


def test_docs_jobs_use_gate_names() -> None:
    build = Path(".github/workflows/docs-check.yml").read_text(encoding="utf-8")
    lint = Path(".github/workflows/doc-lint.yml").read_text(encoding="utf-8")
    verify = Path("scripts/verify_docs.sh").read_text(encoding="utf-8")
    deploy = Path(".github/workflows/docs-deploy.yml").read_text(encoding="utf-8")

    assert "pip install -e .[docs] tomli" in build
    for name in ("docs_build", "docs_artifacts", "docs_linkcheck"):
        assert f"python scripts/run_gate.py {name}" in build
    lint_runs = [
        line.split("python scripts/run_gate.py ", 1)[1]
        .split(";", 1)[0]
        .strip()
        for line in lint.splitlines()
        if "python scripts/run_gate.py " in line
    ]
    assert lint_runs == ["docs_audit", "traceability", "docs_freshness"]
    verify_runs = [
        line.split("python scripts/run_gate.py ", 1)[1]
        .split(";", 1)[0]
        .strip()
        for line in verify.splitlines()
        if "python scripts/run_gate.py " in line
    ]
    assert verify_runs[:2] == ["docs_audit", "docs_freshness"]
    assert verify_runs.count("docs_build") == 1
    assert "make clean" in verify
    assert verify.count("make clean") == 1
    assert "build/html/guides/quickstart.html" in verify
    assert "pip install -e .[docs] tomli" in deploy
    assert "python scripts/run_gate.py docs_build" in deploy


def test_docs_workflows_watch_gate_manifest() -> None:
    for workflow in (".github/workflows/docs-check.yml", ".github/workflows/doc-lint.yml"):
        text = Path(workflow).read_text(encoding="utf-8")
        assert '"scripts/run_gate.py"' in text
        assert '"scripts/gates.toml"' in text


def test_traceability_has_one_hosted_invocation_and_truthful_triggers() -> None:
    lint = Path(".github/workflows/doc-lint.yml").read_text(encoding="utf-8")
    docs_check = Path(".github/workflows/docs-check.yml").read_text(encoding="utf-8")
    rules = json.loads(Path("docs/governance/rules.json").read_text(encoding="utf-8"))

    assert lint.count("python scripts/run_gate.py traceability") == 1
    assert lint.count('"scripts/check_traceability.py"') == 2
    assert "python scripts/run_gate.py traceability" not in docs_check
    traceability = next(rule for rule in rules if rule["rule_id"] == "CAPABILITY-TRACEABILITY")
    assert traceability["trigger_paths"] == [
        "scripts/gates.toml",
        ".github/workflows/doc-lint.yml",
    ]


def test_docs_entry_points_do_not_duplicate_manifest_commands() -> None:
    verify = Path("scripts/verify_docs.sh").read_text(encoding="utf-8")
    deploy = Path(".github/workflows/docs-deploy.yml").read_text(encoding="utf-8")

    for text in (verify, deploy):
        assert "make html" not in text
        assert "doc_check.py" not in text
        assert "check_docs_freshness.py" not in text
