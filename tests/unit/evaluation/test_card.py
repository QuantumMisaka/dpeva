from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpeva.config import EvaluationCardConfig
from dpeva.evaluation.card import build_evaluation_card


def _write_model_ref(root: Path) -> Path:
    path = root / "model-ref.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "checkpoint",
                "family": "DPA4",
                "backend": "pt",
                "path": "model.pt",
                "supported_operations": ["test"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, **overrides: object) -> EvaluationCardConfig:
    values: dict[str, object] = {
        "candidate_id": "dpa4-air-iter11",
        "output_path": tmp_path / "evaluation-card.json",
        "model_ref_path": _write_model_ref(tmp_path),
    }
    values.update(overrides)
    return EvaluationCardConfig.model_validate(values)


def test_missing_metrics_remain_not_run(tmp_path: Path) -> None:
    card = build_evaluation_card(_config(tmp_path))

    assert set(card.metrics) == {
        "in_domain_cumulative",
        "iter11_last_wave",
        "historical_domain",
        "matpes_retention",
        "training_cost",
        "surface_slice",
    }
    assert all(metric.status == "not-run" for metric in card.metrics.values())
    assert all(metric.value is None for metric in card.metrics.values())


def test_metric_payload_and_evidence_are_preserved(tmp_path: Path) -> None:
    metric_path = tmp_path / "in-domain.json"
    metric_path.write_text(
        json.dumps({"status": "passed", "value": {"mae": 0.12}, "detail": "held-out"}),
        encoding="utf-8",
    )

    card = build_evaluation_card(
        _config(tmp_path, in_domain_cumulative_path=metric_path)
    )

    metric = card.metrics["in_domain_cumulative"]
    assert metric.status == "passed"
    assert metric.value == {"mae": 0.12}
    assert metric.detail == "held-out"
    assert metric.evidence_ref == "in-domain.json"


@pytest.mark.parametrize("payload", [{"status": "passed", "unknown": 1}, {"value": {}}])
def test_malformed_metric_is_failed_without_aborting_assembly(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    metric_path = tmp_path / "broken.json"
    metric_path.write_text(json.dumps(payload), encoding="utf-8")

    card = build_evaluation_card(
        _config(tmp_path, training_cost_path=metric_path)
    )

    metric = card.metrics["training_cost"]
    assert metric.status == "failed"
    assert metric.value is None
    assert metric.evidence_ref == "broken.json"
    assert card.metrics["surface_slice"].status == "not-run"


def test_missing_configured_metric_is_failed_with_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing-metric.json"

    card = build_evaluation_card(_config(tmp_path, surface_slice_path=missing))

    metric = card.metrics["surface_slice"]
    assert metric.status == "failed"
    assert metric.value is None
    assert metric.evidence_ref == "missing-metric.json"
    assert "does not exist" in (metric.detail or "")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_metric_constant_is_failed_without_serializing_null(
    tmp_path: Path, constant: str
) -> None:
    metric_path = tmp_path / f"non-finite-{constant.replace('-', 'negative-')}.json"
    metric_path.write_text(
        f'{{"status":"passed","value":{{"mae":{constant}}}}}',
        encoding="utf-8",
    )

    card = build_evaluation_card(_config(tmp_path, training_cost_path=metric_path))

    metric = card.metrics["training_cost"]
    assert metric.status == "failed"
    assert metric.value is None
    assert metric.evidence_ref == metric_path.name
    assert "non-finite JSON constant" in (metric.detail or "")
    serialized_payload = json.loads(card.model_dump_json())
    assert serialized_payload["metrics"]["training_cost"]["value"] is None


def test_card_rejects_invalid_model_reference(tmp_path: Path) -> None:
    model_ref = tmp_path / "invalid-model-ref.json"
    model_ref.write_text(
        json.dumps(
            {
                "kind": "checkpoint",
                "family": "DPA4",
                "backend": "pt",
                "path": "model.pt",
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model reference"):
        build_evaluation_card(_config(tmp_path, model_ref_path=model_ref))


def test_card_rejects_missing_model_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model reference.*does not exist"):
        build_evaluation_card(
            _config(tmp_path, model_ref_path=tmp_path / "missing-model-ref.json")
        )


def test_card_rejects_invalid_dataset_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "broken",
                "parents": [{"dataset_id": "old", "frame_count": 2}],
                "transformation": "merge",
                "frame_count": 1,
                "removed_frame_count": 0,
                "system_count": 1,
                "type_map": ["Fe"],
                "format": "deepmd/npy",
                "source_entries": ["old"],
                "validation_result": {
                    "status": "passed", "counts_reconciled": True,
                    "sources_declared": True, "intersections_explained": True,
                    "type_map_compatible": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset manifest"):
        build_evaluation_card(_config(tmp_path, dataset_manifest_paths=[manifest]))


def test_card_rejects_missing_dataset_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset manifest.*does not exist"):
        build_evaluation_card(
            _config(tmp_path, dataset_manifest_paths=[tmp_path / "missing-manifest.json"])
        )


def test_card_preserves_dataset_manifest_references(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "imported",
                "parents": [],
                "transformation": "import",
                "frame_count": 2,
                "removed_frame_count": 0,
                "system_count": 1,
                "type_map": ["Fe"],
                "format": "deepmd/npy",
                "source_entries": ["imported"],
                "validation_result": {
                    "status": "passed", "counts_reconciled": True,
                    "sources_declared": True, "intersections_explained": True,
                    "type_map_compatible": True,
                },
            }
        ),
        encoding="utf-8",
    )

    card = build_evaluation_card(_config(tmp_path, dataset_manifest_paths=[manifest]))

    assert card.dataset_refs == ["dataset-manifest.json"]


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"passed","value":{"mae":1e999}}',
        '{"status":"passed","value":{"nested":[{"rmse":1e999}]}}',
    ],
)
def test_non_finite_metric_values_are_failed_recursively(
    tmp_path: Path, payload: str
) -> None:
    metric_path = tmp_path / "nested-non-finite.json"
    metric_path.write_text(payload, encoding="utf-8")

    card = build_evaluation_card(_config(tmp_path, training_cost_path=metric_path))

    metric = card.metrics["training_cost"]
    assert metric.status == "failed"
    assert metric.evidence_ref == "nested-non-finite.json"
    assert "non-finite metric value" in (metric.detail or "")


def test_card_references_are_portable_after_package_relocation(tmp_path: Path) -> None:
    package = tmp_path / "candidate-package"
    evidence = package / "evidence"
    card_dir = package / "card"
    evidence.mkdir(parents=True)
    card_dir.mkdir()
    model = _write_model_ref(evidence)
    metric = evidence / "metric.json"
    metric.write_text(json.dumps({"status": "passed", "value": {"mae": 0.1}}), encoding="utf-8")
    manifest = evidence / "dataset-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0", "dataset_id": "imported", "parents": [],
                "transformation": "import", "frame_count": 2,
                "removed_frame_count": 0, "system_count": 1, "type_map": ["Fe"],
                "format": "deepmd/npy", "source_entries": ["imported"],
                "validation_result": {
                    "status": "passed", "counts_reconciled": True,
                    "sources_declared": True, "intersections_explained": True,
                    "type_map_compatible": True,
                },
            }
        ), encoding="utf-8",
    )
    card = build_evaluation_card(
        EvaluationCardConfig(
            candidate_id="portable", model_ref_path=model,
            output_path=card_dir / "evaluation-card.json",
            dataset_manifest_paths=[manifest], surface_slice_path=metric,
            downstream_feedback_ref=str(evidence / "feedback.json"),
        )
    )
    assert card.model_ref == "../evidence/model-ref.json"
    assert card.dataset_refs == ["../evidence/dataset-manifest.json"]
    assert card.metrics["surface_slice"].evidence_ref == "../evidence/metric.json"
    assert card.downstream_feedback_ref == "../evidence/feedback.json"
