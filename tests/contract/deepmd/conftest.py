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

from dpeva.compatibility import CapabilityAttestation, CapabilityMatrix


_FIXTURE_ENV = {
    "pt_model": "DPEVA_DEEPMD_PT_MODEL",
    "dpa4c_model": "DPEVA_DEEPMD_DPA4C_MODEL",
    "periodic_data": "DPEVA_DEEPMD_PERIODIC_DATA",
}
CONTRACT_REQUIRED_ENV = "DPEVA_DEEPMD_CONTRACT_REQUIRED"
_CONTRACT_LOG_DIR: Path | None = None
_COMMAND_INDEX = 0
_CONTRACT_SCOPES = {
    "dpa4": ("pt_model", "periodic_data"),
    "dpa4c": ("dpa4c_model", "periodic_data"),
    "all": tuple(_FIXTURE_ENV),
}


class FixtureConfigurationError(RuntimeError):
    """A required contract fixture is absent or invalid."""


def _required_mode() -> bool:
    return os.environ.get(CONTRACT_REQUIRED_ENV) == "1"


def _contract_scope() -> str:
    scope = os.environ.get("DPEVA_DEEPMD_CONTRACT_SCOPE", "all")
    if scope not in _CONTRACT_SCOPES:
        raise FixtureConfigurationError(
            f"DPEVA_DEEPMD_CONTRACT_SCOPE must be one of {tuple(_CONTRACT_SCOPES)}"
        )
    return scope


def _fixture_required(name: str) -> bool:
    return _required_mode() and name in _CONTRACT_SCOPES[_contract_scope()]


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
    if _required_mode() and skipped:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    effective_exitstatus = int(session.exitstatus)
    (_CONTRACT_LOG_DIR / "pytest-session.json").write_text(
        json.dumps(
            {
                "exitstatus": effective_exitstatus,
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


def _resolve_optional_head(env_name: str) -> str | None:
    """Resolve a non-sensitive multitask head without ever choosing a default."""

    value = os.environ.get(env_name, "").strip()
    return value or None


def head_args(head: str | None) -> list[str]:
    return ["--head", head] if head else []


@pytest.fixture(scope="session", autouse=True)
def validate_required_contract_fixtures() -> None:
    """Validate protected bundle paths before any contract test command."""

    if _required_mode():
        for name in _CONTRACT_SCOPES[_contract_scope()]:
            _resolve_required_path(_FIXTURE_ENV[name], required=True)


@pytest.fixture(scope="session")
def dp_executable() -> str:
    """Return ``dp`` or fail; missing named scientific fixtures are skippable."""

    executable = shutil.which("dp")
    if executable is None:
        pytest.fail("DeepMD executable 'dp' is unavailable in the contract environment")
    return executable


@pytest.fixture(scope="session")
def pt_model() -> Path:
    return _resolve_required_path(
        _FIXTURE_ENV["pt_model"], required=_fixture_required("pt_model")
    )


@pytest.fixture(scope="session")
def dpa4c_model() -> Path:
    return _resolve_required_path(
        _FIXTURE_ENV["dpa4c_model"], required=_fixture_required("dpa4c_model")
    )


@pytest.fixture(scope="session")
def pt_head() -> str | None:
    return _resolve_optional_head("DPEVA_DEEPMD_PT_HEAD")


@pytest.fixture(scope="session")
def dpa4c_head() -> str | None:
    return _resolve_optional_head("DPEVA_DEEPMD_DPA4C_HEAD")


@pytest.fixture(scope="session")
def periodic_data() -> Path:
    return _resolve_required_path(
        _FIXTURE_ENV["periodic_data"], required=_fixture_required("periodic_data")
    )


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


def write_cpu_attestation(
    operation: str,
    dp_executable: str,
    result: subprocess.CompletedProcess[str],
    *,
    case: str,
) -> Path:
    """Persist an attestation only after a caller's full contract assertions."""

    if result.returncode != 0:
        raise AssertionError("failed contract cannot issue a finished attestation")
    backend = "pt-expt" if case.startswith("dpa4c") else "pt"
    records = [
        record
        for record in CapabilityMatrix.load_default().records
        if record.key.operation == operation
        and record.key.backend == backend
        and record.verification_status == "implemented"
    ]
    if len(records) != 1 or records[0].verification_command is None:
        raise AssertionError(f"no unique implemented manifest record for {operation}/{case}")
    version = subprocess.run([dp_executable, "--version"], check=False, capture_output=True, text=True)
    attestation = CapabilityAttestation(
        status="finished",
        returncode=result.returncode,
        capability_key=records[0].key,
        verification_command=records[0].verification_command,
        deepmd_version=version.stdout.strip(),
        source="cpu-contract",
        case=case,
    )
    target_root = _CONTRACT_LOG_DIR or Path("build/deepmd-cpu-contract")
    target = target_root / "attestations" / f"{case}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(attestation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def classify_contract_result(
    result: Any, required_paths: list[Path]
) -> str:
    """Map command evidence to the run-contract failure categories."""

    if result.returncode != 0:
        return "EXECUTION"
    if any(not path.exists() for path in required_paths):
        return "ARTIFACT"
    return "FINISHED"
