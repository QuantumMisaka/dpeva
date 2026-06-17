from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 CI fallback.
    import tomli as tomllib


def test_core_ase_dependency_matches_atst_tools_baseline() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "ase>=3.28.0" in data["project"]["dependencies"]


def test_atst_tools_remains_optional_explore_dependency() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "atst-tools>=2.1.0" in data["project"]["optional-dependencies"]["explore"]
    assert all(
        not dependency.startswith("atst-tools")
        for dependency in data["project"]["dependencies"]
    )
