from pathlib import Path

from dpeva.labeling.postprocess import AbacusPostProcessor


def test_check_convergence_true(tmp_path: Path):
    task_dir = tmp_path / "task"
    out_dir = task_dir / "OUT.ABACUS"
    out_dir.mkdir(parents=True)
    (out_dir / "running_scf.log").write_text("charge density convergence is achieved\n")

    pp = AbacusPostProcessor({})
    assert pp.check_convergence(task_dir) is True


def test_check_convergence_false(tmp_path: Path):
    task_dir = tmp_path / "task"
    out_dir = task_dir / "OUT.ABACUS"
    out_dir.mkdir(parents=True)
    (out_dir / "running_scf.log").write_text("convergence has not been achieved\n")

    pp = AbacusPostProcessor({})
    assert pp.check_convergence(task_dir) is False
