"""Shared post-execution artifact validation for run contracts."""

from __future__ import annotations

from pathlib import Path


class ArtifactValidationError(RuntimeError):
    """Raised when a workflow produced no usable declared artifact."""


def _require_nonempty(paths: list[Path], description: str) -> list[Path]:
    if not paths or any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ArtifactValidationError(f"missing or empty {description}")
    return paths


def validate_feature_outputs(
    output_dir: Path,
    exporter: str,
    expected_pools: list[str] | None = None,
) -> list[Path]:
    """Return feature outputs, requiring at least one artifact per pool."""

    pattern = "embedding.hdf5" if exporter == "embed" else "*.npy"
    if expected_pools:
        outputs: list[Path] = []
        for pool in expected_pools:
            pool_root = output_dir / pool
            pool_outputs = (
                [pool_root / "embedding.hdf5"]
                if exporter == "embed"
                else sorted(pool_root.rglob(pattern))
            )
            try:
                outputs.extend(_require_nonempty(pool_outputs, f"feature artifacts for pool {pool!r}"))
            except ArtifactValidationError as exc:
                raise ArtifactValidationError(
                    f"missing or empty feature artifacts for pool {pool!r} under {output_dir}"
                ) from exc
        return outputs
    outputs = (
        [output_dir / "embedding.hdf5"]
        if exporter == "embed"
        else sorted(output_dir.rglob(pattern))
    )
    return _require_nonempty(outputs, f"feature artifacts under {output_dir}")


def validate_inference_outputs(output_dir: Path, prefix: str) -> list[Path]:
    """Return non-empty ``dp test`` output files for one model directory."""

    outputs = sorted(output_dir.glob(f"{prefix}.*.out"))
    return _require_nonempty(outputs, f"inference artifacts under {output_dir}")
