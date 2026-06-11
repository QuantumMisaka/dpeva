from pathlib import Path

from ase import Atoms

from dpeva.exploration.base import ExplorationBackend, ExplorationRequest, ExplorationResult
from dpeva.exploration.manager import get_backend, register_backend, run_exploration


class DummyBackend(ExplorationBackend):
    name = "dummy"
    supported_workflow_types = {"md"}

    def run(self, request: ExplorationRequest) -> ExplorationResult:
        return ExplorationResult(
            backend=self.name,
            workflow_type=request.workflow_type,
            work_dir=request.work_dir,
            status="success",
            structures=[Atoms("H")],
        )


def test_run_exploration_uses_registered_backend(tmp_path: Path) -> None:
    register_backend("dummy", lambda: DummyBackend())
    request = ExplorationRequest(
        backend="dummy",
        workflow_type="md",
        work_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
    )

    result = run_exploration(request)

    assert result.status == "success"
    assert result.backend == "dummy"
    assert len(result.structures) == 1


def test_get_backend_reports_unknown_backend() -> None:
    try:
        get_backend("missing")
    except ValueError as exc:
        assert "Unknown exploration backend: missing" in str(exc)
    else:
        raise AssertionError("missing backend should fail")
