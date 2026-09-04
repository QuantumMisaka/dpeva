from __future__ import annotations

import pytest
from pydantic import ValidationError

from dpeva.config import InferenceConfig, SubmissionConfig
from dpeva.config_migration import migrate_legacy_config


def test_flat_submission_keys_migrate_without_mutating_source() -> None:
    raw = {
        "backend": "slurm",
        "env_setup": "module load x",
        "data_path": "data",
    }

    result = migrate_legacy_config(raw)

    assert result.normalized["submission"] == {
        "backend": "slurm",
        "env_setup": "module load x",
    }
    assert "backend" not in result.normalized
    assert "env_setup" not in result.normalized
    assert {item.field for item in result.warnings} == {"backend", "env_setup"}
    assert raw == {
        "backend": "slurm",
        "env_setup": "module load x",
        "data_path": "data",
    }


def test_flat_and_nested_submission_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting legacy field 'backend'"):
        migrate_legacy_config(
            {"backend": "local", "submission": {"backend": "slurm"}}
        )


def test_unknown_public_config_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="results_prefx"):
        InferenceConfig.model_validate(
            {"data_path": "data", "results_prefx": "wrong"}
        )


def test_nested_submission_model_is_strict() -> None:
    with pytest.raises(ValidationError, match="unexpected"):
        SubmissionConfig.model_validate({"unexpected": True})


def test_workflow_backend_is_not_reinterpreted_as_submission_backend() -> None:
    raw = {"backend": "atst-tools", "workflow_type": "md"}

    result = migrate_legacy_config(raw)

    assert result.normalized == raw
    assert result.warnings == ()
