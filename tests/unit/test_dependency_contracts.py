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


def test_deepmd_is_bounded_optional_dependency() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    core = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]

    assert all(not item.startswith("deepmd-kit") for item in core)
    assert extras["deepmd"] == ["deepmd-kit>=3.2,<3.3"]


def test_dev_extra_is_independent_of_deepmd_and_exploration() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dev = data["project"]["optional-dependencies"]["dev"]

    assert all(not item.startswith("deepmd-kit") for item in dev)
    assert all(not item.startswith("atst-tools") for item in dev)
