"""Structural contracts for hosted CI orchestration boundaries."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def _workflow(name: str) -> dict[str, object]:
    value = yaml.load(
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(value, dict)
    return value


def _run_steps(job: dict[str, object]) -> list[str]:
    steps = job["steps"]
    assert isinstance(steps, list)
    return [str(step["run"]) for step in steps if isinstance(step, dict) and "run" in step]


def test_python_quality_runs_routine_integration_through_manifest() -> None:
    workflow = _workflow("python-quality.yml")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    integration = jobs["integration-tests"]
    assert isinstance(integration, dict)

    runs = _run_steps(integration)
    assert "python scripts/run_gate.py integration" in runs
    assert all("pytest tests/integration" not in run for run in runs)


def test_deepmd_workflow_separates_supported_and_manual_experimental_lanes() -> None:
    workflow = _workflow("deepmd-contract.yml")
    triggers = workflow["on"]
    jobs = workflow["jobs"]
    assert isinstance(triggers, dict)
    assert isinstance(jobs, dict)

    dispatch = triggers["workflow_dispatch"]
    assert isinstance(dispatch, dict)
    inputs = dispatch["inputs"]
    assert isinstance(inputs, dict)
    assert inputs["run_dpa4c"]["type"] == "boolean"

    supported = jobs["dpa4-cpu-contract"]
    experimental = jobs["dpa4c-experimental-contract"]
    assert isinstance(supported, dict)
    assert isinstance(experimental, dict)
    assert "if" not in supported
    assert "workflow_dispatch" in str(experimental["if"])
    assert "run_dpa4c" in str(experimental["if"])
    assert supported["env"]["DPEVA_DEEPMD_CONTRACT_SCOPE"] == "dpa4"
    assert experimental["env"]["DPEVA_DEEPMD_CONTRACT_SCOPE"] == "dpa4c"
    assert "python scripts/run_gate.py deepmd_contract" in _run_steps(supported)
    assert "python scripts/run_gate.py deepmd_dpa4c_contract" in _run_steps(experimental)
    assert "DPEVA_DEEPMD_DPA4C_MODEL" not in "\n".join(_run_steps(supported))
    assert "DPEVA_DEEPMD_DPA4C_MODEL" in "\n".join(_run_steps(experimental))
    for job in (supported, experimental):
        runs = _run_steps(job)
        assert any("deepmd-kit==3.2.0" in run for run in runs)
        assert all("pytest -m deepmd_contract" not in run for run in runs)

    watched = set(triggers["push"]["paths"])
    assert {
        "src/dpeva/**",
        "scripts/gates.toml",
        "scripts/run_gate.py",
        "scripts/validation/**",
    } <= watched


def test_docs_deploy_validates_shared_release_profile_before_write_job() -> None:
    workflow = _workflow("docs-deploy.yml")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert workflow["permissions"]["contents"] == "read"

    prerequisite = jobs["release-validation"]
    deploy = jobs["deploy"]
    assert isinstance(prerequisite, dict)
    assert isinstance(deploy, dict)
    assert "python scripts/run_gate.py release" in _run_steps(prerequisite)
    assert deploy["needs"] == "release-validation"
    assert deploy["permissions"]["contents"] == "write"
    writers = [
        name
        for name, job in jobs.items()
        if isinstance(job, dict)
        and isinstance(job.get("permissions"), dict)
        and job["permissions"].get("contents") == "write"
    ]
    assert writers == ["deploy"]
