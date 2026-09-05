"""Static tests for the required-fixture fail-closed boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tests.contract.deepmd import conftest as contract_conftest
from tests.contract.deepmd.conftest import (
    CONTRACT_REQUIRED_ENV,
    FixtureConfigurationError,
    _resolve_required_path,
    _resolve_optional_head,
    validate_required_contract_fixtures,
)


@pytest.mark.deepmd_contract
def test_required_missing_fixture_errors_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CONTRACT_REQUIRED_ENV, "1")
    monkeypatch.delenv("DPEVA_DEEPMD_PT_MODEL", raising=False)
    calls: list[object] = []

    def forbidden_run(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("DeepMD subprocess must not start")

    monkeypatch.setattr(subprocess, "run", forbidden_run)
    with pytest.raises(FixtureConfigurationError, match="DPEVA_DEEPMD_PT_MODEL"):
        _resolve_required_path("DPEVA_DEEPMD_PT_MODEL")
    assert calls == []


@pytest.mark.deepmd_contract
def test_required_invalid_fixture_errors_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(CONTRACT_REQUIRED_ENV, "1")
    missing = tmp_path / "missing-model.pt"
    monkeypatch.setenv("DPEVA_DEEPMD_PT_MODEL", str(missing))
    with pytest.raises(FixtureConfigurationError, match="does not exist"):
        _resolve_required_path("DPEVA_DEEPMD_PT_MODEL")


@pytest.mark.deepmd_contract
def test_optional_model_head_does_not_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DPEVA_DEEPMD_PT_HEAD", raising=False)
    assert _resolve_optional_head("DPEVA_DEEPMD_PT_HEAD") is None
    monkeypatch.setenv("DPEVA_DEEPMD_PT_HEAD", "   ")
    assert _resolve_optional_head("DPEVA_DEEPMD_PT_HEAD") is None
    monkeypatch.setenv("DPEVA_DEEPMD_PT_HEAD", " downstream ")
    assert _resolve_optional_head("DPEVA_DEEPMD_PT_HEAD") == "downstream"


@pytest.mark.deepmd_contract
def test_required_fixture_names_follow_explicit_contract_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    pt_model = tmp_path / "model.pt"
    dpa4c_model = tmp_path / "model.pt2"
    periodic_data = tmp_path / "periodic-data"
    pt_model.write_bytes(b"pt")
    dpa4c_model.write_bytes(b"dpa4c")
    periodic_data.mkdir()
    monkeypatch.setenv(CONTRACT_REQUIRED_ENV, "1")
    monkeypatch.setenv("DPEVA_DEEPMD_PT_MODEL", str(pt_model))
    monkeypatch.setenv("DPEVA_DEEPMD_PERIODIC_DATA", str(periodic_data))

    monkeypatch.setenv("DPEVA_DEEPMD_CONTRACT_SCOPE", "dpa4")
    monkeypatch.delenv("DPEVA_DEEPMD_DPA4C_MODEL", raising=False)
    validate_required_contract_fixtures.__wrapped__()

    monkeypatch.setenv("DPEVA_DEEPMD_CONTRACT_SCOPE", "dpa4c")
    monkeypatch.delenv("DPEVA_DEEPMD_PT_MODEL")
    monkeypatch.setenv("DPEVA_DEEPMD_DPA4C_MODEL", str(dpa4c_model))
    validate_required_contract_fixtures.__wrapped__()

    monkeypatch.delenv("DPEVA_DEEPMD_CONTRACT_SCOPE")
    monkeypatch.setenv("DPEVA_DEEPMD_PT_MODEL", str(pt_model))
    validate_required_contract_fixtures.__wrapped__()


@pytest.mark.deepmd_contract
def test_required_contract_scope_rejects_any_skipped_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reporter = SimpleNamespace(stats={"skipped": [object()]})
    pluginmanager = SimpleNamespace(get_plugin=lambda _name: reporter)
    session = SimpleNamespace(
        config=SimpleNamespace(pluginmanager=pluginmanager),
        exitstatus=pytest.ExitCode.OK,
        testsfailed=0,
        testscollected=1,
    )
    monkeypatch.setenv(CONTRACT_REQUIRED_ENV, "1")
    monkeypatch.setattr(contract_conftest, "_CONTRACT_LOG_DIR", tmp_path)

    contract_conftest.pytest_sessionfinish(session, int(pytest.ExitCode.OK))

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


@pytest.mark.deepmd_contract
def test_ci_requires_protected_bundle_and_rejects_skips() -> None:
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/deepmd-contract.yml").read_text(
        encoding="utf-8"
    )
    for name in (
        "DPEVA_DEEPMD_CONTRACT_FIXTURE_URL",
        "DPEVA_DEEPMD_CONTRACT_FIXTURE_SHA256",
        "DPEVA_DEEPMD_PT_MODEL",
        "DPEVA_DEEPMD_DPA4C_MODEL",
        "DPEVA_DEEPMD_PERIODIC_DATA",
        "DPEVA_DEEPMD_CONTRACT_REQUIRED=1",
    ):
        assert name in workflow
    assert "secrets.DPEVA_DEEPMD_CONTRACT_FIXTURE_URL" in workflow
    assert "vars.DPEVA_DEEPMD_CONTRACT_FIXTURE_URL" in workflow
    assert "secrets.DPEVA_DEEPMD_CONTRACT_FIXTURE_SHA256" in workflow
    assert "vars.DPEVA_DEEPMD_CONTRACT_FIXTURE_SHA256" in workflow
    assert "DPEVA_DEEPMD_PT_HEAD" in workflow
    assert "DPEVA_DEEPMD_DPA4C_HEAD" in workflow
    assert "curl --fail --silent --location" in workflow
    assert "sha256sum --check --status" in workflow
    assert "python scripts/run_gate.py deepmd_contract" in workflow
    assert "python scripts/run_gate.py deepmd_dpa4c_contract" in workflow
    assert "pytest -m deepmd_contract" not in workflow
    assert "path: build/deepmd-cpu-contract" in workflow
    assert "path: $RUNNER_TEMP" not in workflow
