"""Fixtures and fail-closed helpers for real DeepMD CPU contracts."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest


_FIXTURE_ENV = {
    "pt_model": "DPEVA_DEEPMD_PT_MODEL",
    "dpa4c_model": "DPEVA_DEEPMD_DPA4C_MODEL",
    "periodic_data": "DPEVA_DEEPMD_PERIODIC_DATA",
}
CONTRACT_REQUIRED_ENV = "DPEVA_DEEPMD_CONTRACT_REQUIRED"
_CONTRACT_LOG_DIR: Path | None = None
_COMMAND_INDEX = 0


class FixtureConfigurationError(RuntimeError):
    """A required contract fixture is absent or invalid."""


def _required_mode() -> bool:
    return os.environ.get(CONTRACT_REQUIRED_ENV) == "1"


def pytest_configure(config: pytest.Config) -> None:
    """Create a stable, always-uploaded CI evidence directory."""

    global _CONTRACT_LOG_DIR
    _CONTRACT_LOG_DIR = Path(
        os.environ.get("DPEVA_DEEPMD_CONTRACT_ARTIFACT_DIR", "build/deepmd-cpu-contract")
    ).expanduser()
    _CONTRACT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.addinivalue_line(
        "markers", "deepmd_contract: real DeepMD 3.2 CPU CLI/API contract"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Persist the test outcome even when an upstream command fails."""

    if _CONTRACT_LOG_DIR is None:
        return
    terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = len(terminal_reporter.stats.get("skipped", [])) if terminal_reporter else 0
    (_CONTRACT_LOG_DIR / "pytest-session.json").write_text(
        json.dumps(
            {
                "exitstatus": exitstatus,
                "testsfailed": session.testsfailed,
                "testscollected": session.testscollected,
                "skipped": skipped,
                "deepmd_contract": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve_required_path(env_name: str, *, required: bool | None = None) -> Path:
    required = _required_mode() if required is None else required
    value = os.environ.get(env_name)
    if not value:
        message = f"{env_name} is not set; owner: Compatibility Owner"
        if required:
            raise FixtureConfigurationError(
                f"{message}; {CONTRACT_REQUIRED_ENV}=1 requires this fixture"
            )
        pytest.skip(message)
    path = Path(value).expanduser().resolve()
    if not path.exists():
        message = f"{env_name}={path} does not exist; owner: Compatibility Owner"
        if required:
            raise FixtureConfigurationError(message)
        pytest.fail(message)
    return path


@pytest.fixture(scope="session", autouse=True)
def validate_required_contract_fixtures() -> None:
    """Validate protected bundle paths before any contract test command."""

    if _required_mode():
        for env_name in _FIXTURE_ENV.values():
            _resolve_required_path(env_name, required=True)


@pytest.fixture(scope="session")
def dp_executable() -> str:
    """Return ``dp`` or fail; missing named scientific fixtures are skippable."""

    executable = shutil.which("dp")
    if executable is None:
        pytest.fail("DeepMD executable 'dp' is unavailable in the contract environment")
    return executable


@pytest.fixture(scope="session")
def pt_model() -> Path:
    return _resolve_required_path(_FIXTURE_ENV["pt_model"])


@pytest.fixture(scope="session")
def dpa4c_model() -> Path:
    return _resolve_required_path(_FIXTURE_ENV["dpa4c_model"])


@pytest.fixture(scope="session")
def periodic_data() -> Path:
    return _resolve_required_path(_FIXTURE_ENV["periodic_data"])


def _iter_data_files(path: Path) -> list[Path]:
    if path.is_file():
        result: list[Path] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            candidate = Path(line.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            if candidate.exists():
                result.extend(_iter_data_files(candidate))
        return result
    if not path.is_dir():
        return []
    direct = path / "coord.npy"
    sets = sorted(path.glob("set.*/coord.npy"))
    if direct.exists():
        return [direct]
    if sets:
        return sets
    return sorted(path.rglob("coord.npy"))


def frame_count(path: Path) -> int:
    """Count frames without asking DeepMD to execute a command."""

    files = _iter_data_files(path)
    if not files:
        pytest.fail(f"periodic fixture contains no coord.npy: {path}")
    total = 0
    for coord in files:
        array = np.load(coord, mmap_mode="r")
        if array.ndim < 1:
            pytest.fail(f"invalid coordinate shape in {coord}: {array.shape}")
        total += int(array.shape[0])
    return total


def _artifact_safe_name(argv: list[str]) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", " ".join(argv)).strip("_")
    return text[:100] or "command"


def run_contract(argv: list[str], required_paths: list[Path]) -> subprocess.CompletedProcess[str]:
    """Run one real CLI argv and require both success and declared artifacts."""

    global _COMMAND_INDEX
    result = subprocess.run(argv, check=False, capture_output=True, text=True)
    if _CONTRACT_LOG_DIR is not None:
        _COMMAND_INDEX += 1
        stem = f"{_COMMAND_INDEX:02d}-{_artifact_safe_name(argv)}"
        (_CONTRACT_LOG_DIR / f"{stem}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (_CONTRACT_LOG_DIR / f"{stem}.stderr.log").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    missing = [str(path) for path in required_paths if not path.exists()]
    assert missing == [], f"missing contract artifacts: {missing}"
    return result


def classify_contract_result(
    result: Any, required_paths: list[Path]
) -> str:
    """Map command evidence to the run-contract failure categories."""

    if result.returncode != 0:
        return "EXECUTION"
    if any(not path.exists() for path in required_paths):
        return "ARTIFACT"
    return "FINISHED"
