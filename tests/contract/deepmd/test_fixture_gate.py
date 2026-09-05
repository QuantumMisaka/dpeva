"""Static tests for the required-fixture fail-closed boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import (
    CONTRACT_REQUIRED_ENV,
    FixtureConfigurationError,
    _resolve_required_path,
    _resolve_optional_head,
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
    assert "pytest -m deepmd_contract tests/contract/deepmd -q" in workflow
    assert "Reject skipped DeepMD contract cases" in workflow
    assert "path: build/deepmd-cpu-contract" in workflow
    assert "path: $RUNNER_TEMP" not in workflow
