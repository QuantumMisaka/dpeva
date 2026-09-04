from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from dpeva.run.model import (
    ModelArtifactKind,
    ModelArtifactRef,
    ModelRole,
    load_model_ref,
    require_operation,
    resolve_model_refs,
)


def test_discovery_handles_gaps_and_regular_ema(tmp_path: Path) -> None:
    for relative in ("0/model.ckpt.pt", "0/model_ema.ckpt.pt", "2/model.ckpt.pt"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")

    refs = resolve_model_refs(tmp_path, family="DPA4C", backend="pt-expt")

    assert [(ref.path, ref.role) for ref in refs] == [
        (str(tmp_path / "0/model.ckpt.pt"), ModelRole.REGULAR),
        (str(tmp_path / "0/model_ema.ckpt.pt"), ModelRole.EMA),
        (str(tmp_path / "2/model.ckpt.pt"), ModelRole.REGULAR),
    ]
    assert all(ref.kind is ModelArtifactKind.CHECKPOINT for ref in refs)
    assert all(ref.checksum for ref in refs)
    assert all("test" in ref.supported_operations for ref in refs)


def test_pretrained_alias_must_be_resolved_before_execution() -> None:
    ref = ModelArtifactRef(
        kind="pretrained-alias",
        family="DPA4",
        backend="pt",
        alias="DPA4-Air-OMat24-v20260805",
        supported_operations=["test"],
    )

    with pytest.raises(ValueError, match="resolve pretrained alias"):
        require_operation(ref, "test")


def test_pretrained_alias_rejects_local_path() -> None:
    with pytest.raises(ValueError, match="must not declare path"):
        ModelArtifactRef(
            kind="pretrained-alias",
            family="DPA4",
            backend="pt",
            alias="DPA4-Air-OMat24-v20260805",
            path="model.pt",
        )


def test_local_artifact_rejects_alias() -> None:
    with pytest.raises(ValueError, match="must not declare alias"):
        ModelArtifactRef(
            kind="checkpoint",
            family="DPA4",
            backend="pt",
            alias="not-a-local-file",
        )


def test_unsupported_operation_fails_before_execution() -> None:
    ref = ModelArtifactRef(
        kind="frozen",
        family="DPA4",
        backend="pt",
        path="model.pth",
        supported_operations=["test"],
    )

    with pytest.raises(ValueError, match="does not declare operation: embed"):
        require_operation(ref, "embed")


def test_model_reference_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra_field"):
        ModelArtifactRef(
            kind="checkpoint",
            family="DPA4",
            backend="pt",
            path="model.pt",
            extra_field=True,
        )


def test_load_model_ref_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "model-ref.json"
    path.write_text(
        json.dumps(
            {
                "kind": "exportable",
                "family": "DPA4C",
                "backend": "pt-expt",
                "path": "model.pt2",
                "supported_operations": ["test"],
            }
        ),
        encoding="utf-8",
    )

    ref = load_model_ref(path)

    assert ref.kind is ModelArtifactKind.EXPORTABLE
    assert ref.path == str((tmp_path / "model.pt2").resolve())


def test_load_model_ref_resolves_artifact_path_relative_to_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "model.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"model")
    ref_path = tmp_path / "refs" / "model-ref.json"
    ref_path.parent.mkdir()
    ref_path.write_text(
        json.dumps(
            {
                "kind": "checkpoint",
                "family": "DPA4",
                "backend": "pt",
                "path": "../models/model.pt",
                "checksum": hashlib.sha256(b"model").hexdigest(),
                "supported_operations": ["test"],
            }
        ),
        encoding="utf-8",
    )

    ref = load_model_ref(ref_path)

    assert ref.path == str(artifact.resolve())
    require_operation(ref, "test")


def test_checksum_mismatch_fails_before_execution(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"changed")
    ref = ModelArtifactRef(
        kind="checkpoint",
        family="DPA4",
        backend="pt",
        path=str(artifact),
        checksum="0" * 64,
        supported_operations=["test"],
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        require_operation(ref, "test")


def test_explicit_artifact_path_must_exist_before_execution(tmp_path: Path) -> None:
    ref = ModelArtifactRef(
        kind="frozen",
        family="DPA4",
        backend="pt",
        path=str(tmp_path / "missing.pt"),
        supported_operations=["test"],
    )

    with pytest.raises(ValueError, match="path does not exist"):
        require_operation(ref, "test")
