"""ATST-Tools exploration backend."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ase.io import write

from .base import ExplorationArtifact, ExplorationBackend, ExplorationRequest, ExplorationResult
from .io import read_structure_paths


MISSING_ATST_MESSAGE = (
    "Exploration backend 'atst-tools' is enabled but package/CLI 'atst' is not "
    "available. Install dpeva[explore] or pip install atst-tools."
)


class ATSTToolsBackend(ExplorationBackend):
    name = "atst-tools"
    supported_workflow_types = {"md", "relax"}

    def run(self, request: ExplorationRequest) -> ExplorationResult:
        if request.workflow_type not in self.supported_workflow_types:
            raise ValueError(
                f"Unsupported atst-tools workflow_type: {request.workflow_type}. "
                "Supported: md, relax"
            )
        if shutil.which("atst") is None:
            raise RuntimeError(MISSING_ATST_MESSAGE)

        work_dir = Path(request.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        input_artifacts = self._write_input_structures(work_dir, request)
        completed = subprocess.run(
            ["atst", "run", str(request.config_path)],
            cwd=work_dir,
            check=False,
            capture_output=True,
            text=True,
        )

        result_paths = request.metadata.get("result_structure_paths", [])
        artifacts = [
            ExplorationArtifact(kind="config", path=Path(request.config_path)),
            *input_artifacts,
        ]
        status = "success" if completed.returncode == 0 else "failed"
        error_message = completed.stderr if status == "failed" else None
        structures = []
        result_artifacts = [
            ExplorationArtifact(kind="structure", path=Path(path))
            for path in result_paths
        ]
        if status == "success":
            missing_paths = [Path(path) for path in result_paths if not Path(path).exists()]
            if missing_paths:
                status = "failed"
                error_message = (
                    "Configured result structure path does not exist: "
                    + ", ".join(str(path) for path in missing_paths)
                )
            else:
                try:
                    structures = read_structure_paths(result_paths)
                    artifacts.extend(result_artifacts)
                except Exception as exc:
                    status = "failed"
                    error_message = f"Failed to read result structures: {exc}"

        manifest_path = work_dir / "dpeva_exploration_result.json"
        self._write_manifest(
            manifest_path=manifest_path,
            request=request,
            status=status,
            completed=completed,
            input_artifacts=input_artifacts,
            result_artifacts=result_artifacts,
            error_message=error_message,
        )
        artifacts = [
            *artifacts,
            ExplorationArtifact(kind="manifest", path=manifest_path),
        ]

        return ExplorationResult(
            backend=self.name,
            workflow_type=request.workflow_type,
            work_dir=work_dir,
            status=status,
            structures=structures,
            artifacts=artifacts,
            metrics={
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            error_message=error_message,
        )

    def _write_input_structures(
        self, work_dir: Path, request: ExplorationRequest
    ) -> list[ExplorationArtifact]:
        if not request.input_structures:
            return []
        input_dir = work_dir / "dpeva_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for index, atoms in enumerate(request.input_structures):
            path = input_dir / f"input_{index:06d}.extxyz"
            write(path, atoms)
            artifacts.append(ExplorationArtifact(kind="input_structure", path=path))
        return artifacts

    def _write_manifest(
        self,
        *,
        manifest_path: Path,
        request: ExplorationRequest,
        status: str,
        completed: subprocess.CompletedProcess,
        input_artifacts: list[ExplorationArtifact],
        result_artifacts: list[ExplorationArtifact],
        error_message: str | None,
    ) -> None:
        manifest = {
            "backend": self.name,
            "workflow_type": request.workflow_type,
            "status": status,
            "work_dir": str(request.work_dir),
            "config_path": str(request.config_path),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "error_message": error_message,
            "input_artifacts": [str(artifact.path) for artifact in input_artifacts],
            "result_structure_artifacts": [
                str(artifact.path) for artifact in result_artifacts
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
