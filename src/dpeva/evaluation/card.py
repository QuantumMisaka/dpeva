"""Schema and deterministic assembly for candidate evaluation cards.

An evaluation card indexes evidence produced by other workflows. It does not
run scientific evaluations or infer a ranking. Absent metric evidence is
represented as ``not-run`` and invalid configured metric evidence as
``failed`` with its source path retained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError, model_validator

from dpeva.config import EvaluationCardConfig
from dpeva.run.dataset import DatasetManifest, validate_lineage_counts
from dpeva.run.model import load_model_ref


MetricStatus = Literal["passed", "failed", "not-run", "not-applicable"]

_DIMENSIONS = (
    "in_domain_cumulative",
    "iter11_last_wave",
    "historical_domain",
    "matpes_retention",
    "training_cost",
    "surface_slice",
)


class EvaluationMetric(BaseModel):
    """One dimension's status and optional measured value/evidence."""

    model_config = ConfigDict(extra="forbid")

    status: MetricStatus
    value: dict[str, Any] | None = None
    evidence_ref: StrictStr | None = None
    detail: StrictStr | None = None


class EvaluationCard(BaseModel):
    """Closed, machine-readable candidate handoff evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    candidate_id: StrictStr
    model_ref: StrictStr
    dataset_refs: list[StrictStr]
    metrics: dict[str, EvaluationMetric]
    downstream_feedback_ref: StrictStr | None = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> "EvaluationCard":
        observed = set(self.metrics)
        required = set(_DIMENSIONS)
        if observed != required:
            missing = sorted(required - observed)
            extra = sorted(observed - required)
            detail: list[str] = []
            if missing:
                detail.append(f"missing={missing}")
            if extra:
                detail.append(f"extra={extra}")
            raise ValueError("evaluation card dimensions must be complete: " + ", ".join(detail))
        return self


def _resolved_file(path: Path, *, description: str) -> Path:
    """Return a normalized regular file path or raise an evidence error."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{description} does not exist or is not a file: {resolved}")
    return resolved


def _load_model_reference(path: Path) -> Path:
    reference_path = _resolved_file(path, description="model reference")
    try:
        load_model_ref(reference_path)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"invalid model reference {reference_path}: {exc}") from exc
    return reference_path


def _load_dataset_manifest(path: Path) -> Path:
    manifest_path = _resolved_file(path, description="dataset manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = DatasetManifest.model_validate(payload)
        validate_lineage_counts(manifest)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"invalid dataset manifest {manifest_path}: {exc}") from exc
    return manifest_path


def _metric_from_file(path: Path) -> EvaluationMetric:
    """Load one strict metric object and attach its immutable source path."""

    evidence_path = Path(path).expanduser().resolve()
    try:
        if not evidence_path.is_file():
            raise ValueError(f"metric evidence does not exist or is not a file: {evidence_path}")
        payload = json.loads(
            evidence_path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_constant,
        )
        if not isinstance(payload, dict):
            raise ValueError("metric evidence must be a JSON object")
        metric = EvaluationMetric.model_validate(payload)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        return EvaluationMetric(status="failed", evidence_ref=str(evidence_path), detail=str(exc))
    return metric.model_copy(update={"evidence_ref": str(evidence_path)})


def _reject_non_finite_constant(value: str) -> Any:
    """Reject JSON extensions that would otherwise admit non-finite values."""

    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def build_evaluation_card(config: EvaluationCardConfig) -> EvaluationCard:
    """Assemble a complete card from strict references and optional metrics."""

    if not isinstance(config, EvaluationCardConfig):
        raise TypeError("config must be an EvaluationCardConfig")

    model_ref_path = _load_model_reference(config.model_ref_path)
    dataset_refs = [str(_load_dataset_manifest(path)) for path in config.dataset_manifest_paths]
    configured_paths: dict[str, Path | None] = {
        "in_domain_cumulative": config.in_domain_cumulative_path,
        "iter11_last_wave": config.iter11_last_wave_path,
        "historical_domain": config.historical_domain_path,
        "matpes_retention": config.matpes_retention_path,
        "training_cost": config.training_cost_path,
        "surface_slice": config.surface_slice_path,
    }
    metrics = {
        name: EvaluationMetric(status="not-run") if path is None else _metric_from_file(path)
        for name, path in configured_paths.items()
    }
    return EvaluationCard(
        candidate_id=config.candidate_id,
        model_ref=str(model_ref_path),
        dataset_refs=dataset_refs,
        metrics=metrics,
        downstream_feedback_ref=config.downstream_feedback_ref,
    )
