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
                "downstream_feedback_ref": "evidence/downstream-review.json",
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
    assert payload["metrics"]["surface_slice"]["evidence_ref"] == "../evidence/surface.json"
    assert payload["metrics"]["in_domain_cumulative"]["status"] == "not-run"
    assert payload["downstream_feedback_ref"] == "../evidence/downstream-review.json"


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


class _CardStub:
    def model_dump_json(self, *, indent: int) -> str:
        assert indent == 2
        return '{"schema_version":"1.0"}'


def test_eval_card_publication_fails_closed_before_temp_on_directory_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifacts" / "evaluation-card.json"

    def fail_fsync(_fd: int) -> None:
        raise OSError("directory fsync unavailable")

    monkeypatch.setattr(cli.os, "fsync", fail_fsync)
    with pytest.raises(cli.EvaluationCardPublicationError, match="not published") as exc:
        cli._publish_evaluation_card(output, _CardStub())

    assert exc.value.published is False
    assert not output.exists()
    assert list(output.parent.glob(".*.tmp")) == []


def test_eval_card_publication_preserves_target_when_post_link_durability_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evaluation-card.json"
    real_fsync = cli.os.fsync
    calls = 0

    def fail_post_link(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("post-publication fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(cli.os, "fsync", fail_post_link)
    with pytest.raises(cli.EvaluationCardPublicationError, match="published but") as exc:
        cli._publish_evaluation_card(output, _CardStub())

    assert exc.value.published is True
    assert output.read_text(encoding="utf-8") == '{"schema_version":"1.0"}\n'
    assert list(tmp_path.glob(".*.tmp")) == []


def test_eval_card_publication_concurrent_winner_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evaluation-card.json"
    real_link = cli.os.link

    def winner_then_link(source: str, destination: str) -> None:
        Path(destination).write_text("winner\n", encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(cli.os, "link", winner_then_link)
    with pytest.raises(FileExistsError):
        cli._publish_evaluation_card(output, _CardStub())

    assert output.read_text(encoding="utf-8") == "winner\n"
    assert list(tmp_path.glob(".*.tmp")) == []
