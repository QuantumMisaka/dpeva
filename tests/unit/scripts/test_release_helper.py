from __future__ import annotations

from pathlib import Path

import pytest

from scripts import release_helper
from scripts.run_gate import load_manifest, resolve_profile


def _version_surfaces(root: Path, *, package: str, readme: str, guide: str) -> None:
    init_file = root / "src/dpeva/__init__.py"
    init_file.parent.mkdir(parents=True)
    init_file.write_text(f'__version__ = "{package}"\n', encoding="utf-8")
    (root / "README.md").write_text(
        f"![Version](https://img.shields.io/badge/version-{readme}-blue)\n",
        encoding="utf-8",
    )
    guide_file = root / "docs/guides/developer-guide.md"
    guide_file.parent.mkdir(parents=True)
    guide_file.write_text(
        f"* **版本**: {guide}\n\n### 6.2 版本历史\n\n* **v0.8.1** historical\n",
        encoding="utf-8",
    )


def _use_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(release_helper, "PROJECT_ROOT", root)
    monkeypatch.setattr(release_helper, "INIT_FILE", root / "src/dpeva/__init__.py")
    monkeypatch.setattr(release_helper, "README_FILE", root / "README.md")
    monkeypatch.setattr(
        release_helper,
        "DEV_GUIDE",
        root / "docs/guides/developer-guide.md",
    )


def test_check_reports_each_mismatched_maintained_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _version_surfaces(tmp_path, package="0.8.2", readme="0.8.1", guide="0.8.0")
    _use_root(monkeypatch, tmp_path)

    errors = release_helper.check_versions()

    assert errors == [
        "README.md declares 0.8.1, expected 0.8.2",
        "docs/guides/developer-guide.md declares 0.8.0, expected 0.8.2",
    ]
    assert release_helper.main(["--check"]) == 1


def test_explicit_version_update_synchronizes_current_surfaces_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _version_surfaces(tmp_path, package="0.8.1", readme="0.8.1", guide="0.8.1")
    _use_root(monkeypatch, tmp_path)

    assert release_helper.main(["0.8.2"]) == 0

    assert release_helper.get_current_version() == "0.8.2"
    assert release_helper.check_versions() == []
    guide = release_helper.DEV_GUIDE.read_text(encoding="utf-8")
    assert "* **版本**: 0.8.2" in guide
    assert "* **v0.8.1** historical" in guide


@pytest.mark.parametrize(
    "invalid",
    ["1.2", "v1.2.3", "1.2.3rc1", "1.2.3+local", "01.2.3", "1.2.-1"],
)
def test_invalid_explicit_version_cannot_mutate_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    _version_surfaces(tmp_path, package="0.8.1", readme="0.8.1", guide="0.8.1")
    _use_root(monkeypatch, tmp_path)
    before = {
        path: path.read_bytes()
        for path in (release_helper.INIT_FILE, release_helper.README_FILE, release_helper.DEV_GUIDE)
    }

    with pytest.raises(ValueError, match="Invalid version or bump type"):
        release_helper.bump_version("0.8.1", invalid)

    assert {path: path.read_bytes() for path in before} == before


def test_release_profile_checks_version_surfaces() -> None:
    manifest = load_manifest(Path("scripts/gates.toml"))

    assert "release_version" in resolve_profile(manifest, "release")
    assert manifest.gates["release_version"].argv == (
        "python",
        "scripts/release_helper.py",
        "--check",
    )


@pytest.mark.parametrize("surface", ["INIT_FILE", "README_FILE", "DEV_GUIDE"])
@pytest.mark.parametrize("operation", ["check", "update"])
def test_duplicate_version_declarations_rejected_before_any_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, operation: str
) -> None:
    _version_surfaces(tmp_path, package="0.8.2", readme="0.8.2", guide="0.8.2")
    _use_root(monkeypatch, tmp_path)
    duplicate = getattr(release_helper, surface)
    content = duplicate.read_text(encoding="utf-8")
    duplicate.write_text(content + content.replace("0.8.2", "0.8.1"), encoding="utf-8")
    before = {
        path: path.read_bytes()
        for path in (release_helper.INIT_FILE, release_helper.README_FILE, release_helper.DEV_GUIDE)
    }

    with pytest.raises(ValueError, match="exactly one maintained version"):
        if operation == "check":
            release_helper.main(["--check"])
        else:
            release_helper.update_files("0.8.3")

    assert {path: path.read_bytes() for path in before} == before
