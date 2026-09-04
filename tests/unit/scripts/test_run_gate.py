from __future__ import annotations

import subprocess
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
