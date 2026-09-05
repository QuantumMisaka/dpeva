"""Shared post-execution artifact validation for run contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class ArtifactValidationError(RuntimeError):
    """Raised when a workflow produced no usable declared artifact."""


@dataclass(frozen=True)
class FileIdentity:
    """Cheap identity used to distinguish ordinary output rewrites."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "FileIdentity":
        stat = path.stat()
        return cls(
            device=stat.st_dev,
            inode=stat.st_ino,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,
        )


@dataclass(frozen=True)
class AttemptOutputBaseline:
    """Metadata snapshot of declared outputs that predate one attempt."""

    identities: dict[Path, FileIdentity]

    @classmethod
    def capture(cls, paths: Iterable[Path]) -> "AttemptOutputBaseline":
        identities: dict[Path, FileIdentity] = {}
        for supplied in paths:
            path = Path(supplied).expanduser().resolve(strict=False)
            if path.is_file():
                identities[path] = FileIdentity.from_path(path)
        return cls(identities)

    def fresh(self, paths: Iterable[Path]) -> list[Path]:
        """Return files created or observably rewritten after this snapshot."""

        fresh: list[Path] = []
        for supplied in paths:
            path = Path(supplied).expanduser().resolve(strict=False)
            if not path.is_file():
                continue
            previous = self.identities.get(path)
            if previous is None or FileIdentity.from_path(path) != previous:
                fresh.append(path)
        return fresh


def _feature_output_candidates(
    output_dir: Path,
    exporter: str,
    expected_pools: list[str] | None = None,
) -> list[Path]:
    pattern = "embedding.hdf5" if exporter == "embed" else "*.npy"
    roots = [output_dir / pool for pool in expected_pools] if expected_pools else [output_dir]
    outputs: list[Path] = []
    for root in roots:
        if exporter == "embed":
            outputs.append(root / "embedding.hdf5")
        else:
            outputs.extend(sorted(root.rglob(pattern)))
    return outputs


def snapshot_feature_outputs(
    output_dir: Path,
    exporter: str,
    expected_pools: list[str] | None = None,
) -> AttemptOutputBaseline:
    """Capture declared feature outputs without reading their array contents."""

    return AttemptOutputBaseline.capture(
        _feature_output_candidates(output_dir, exporter, expected_pools)
    )


def snapshot_inference_outputs(output_dir: Path, prefix: str) -> AttemptOutputBaseline:
    """Capture declared inference outputs before a model command starts."""

    return AttemptOutputBaseline.capture(sorted(output_dir.glob(f"{prefix}.*.out")))


def _require_nonempty(paths: list[Path], description: str) -> list[Path]:
    if not paths or any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ArtifactValidationError(f"missing or empty {description}")
    return paths


def _require_current_attempt(
    paths: list[Path],
    baseline: AttemptOutputBaseline | None,
    description: str,
) -> list[Path]:
    if baseline is None:
        return paths
    fresh = baseline.fresh(paths)
    if not fresh:
        raise ArtifactValidationError(
            f"no {description} was created or rewritten by the current attempt"
        )
    return fresh


def validate_feature_outputs(
    output_dir: Path,
    exporter: str,
    expected_pools: list[str] | None = None,
    baseline: AttemptOutputBaseline | None = None,
) -> list[Path]:
    """Return feature outputs, requiring at least one artifact per pool."""

    if expected_pools:
        outputs: list[Path] = []
        for pool in expected_pools:
            pool_root = output_dir / pool
            pool_outputs = (
                [pool_root / "embedding.hdf5"]
                if exporter == "embed"
                else sorted(pool_root.rglob("*.npy"))
            )
            try:
                nonempty = _require_nonempty(
                    pool_outputs, f"feature artifacts for pool {pool!r}"
                )
                outputs.extend(
                    _require_current_attempt(
                        nonempty,
                        baseline,
                        f"feature artifact for pool {pool!r}",
                    )
                )
            except ArtifactValidationError as exc:
                raise ArtifactValidationError(
                    f"invalid feature artifacts for pool {pool!r} under {output_dir}: {exc}"
                ) from exc
        return outputs
    outputs = (
        [output_dir / "embedding.hdf5"]
        if exporter == "embed"
        else sorted(output_dir.rglob("*.npy"))
    )
    nonempty = _require_nonempty(outputs, f"feature artifacts under {output_dir}")
    return _require_current_attempt(nonempty, baseline, "feature artifact")


def validate_inference_outputs(
    output_dir: Path,
    prefix: str,
    baseline: AttemptOutputBaseline | None = None,
) -> list[Path]:
    """Return non-empty ``dp test`` output files for one model directory."""

    outputs = sorted(output_dir.glob(f"{prefix}.*.out"))
    nonempty = _require_nonempty(outputs, f"inference artifacts under {output_dir}")
    return _require_current_attempt(nonempty, baseline, "inference artifact")
