from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import dpeva.cli as cli


def test_eval_card_cli_writes_all_dimensions_from_config_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_ref = tmp_path / "evidence" / "model-ref.json"
    model_ref.parent.mkdir()
    model_ref.write_text(
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
    metric = tmp_path / "evidence" / "surface.json"
    metric.write_text(json.dumps({"status": "passed", "value": {"mae": 0.1}}), encoding="utf-8")
    config = tmp_path / "eval-card.json"
    config.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-1",
                "model_ref_path": "evidence/model-ref.json",
                "output_path": "artifacts/evaluation-card.json",
                "surface_slice_path": "evidence/surface.json",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "eval-card", str(config)])
    cli.main()

    output = tmp_path / "artifacts" / "evaluation-card.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["metrics"]) == 6
    assert payload["metrics"]["surface_slice"]["status"] == "passed"
    assert payload["metrics"]["surface_slice"]["evidence_ref"] == str(metric.resolve())
    assert payload["metrics"]["in_domain_cumulative"]["status"] == "not-run"


def test_eval_card_cli_does_not_replace_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_ref = tmp_path / "model-ref.json"
    model_ref.write_text(
        json.dumps(
            {
                "kind": "checkpoint",
                "family": "DPA4",
                "backend": "pt",
                "path": "model.pt",
                "supported_operations": ["test"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation-card.json"
    output.write_text("original\n", encoding="utf-8")
    config = tmp_path / "eval-card.json"
    config.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-1",
                "model_ref_path": "model-ref.json",
                "output_path": "evaluation-card.json",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "eval-card", str(config)])
    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert output.read_text(encoding="utf-8") == "original\n"

