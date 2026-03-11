from pathlib import Path

from dpeva.labeling.strategy import ResubmissionStrategy


def test_get_params_with_empty_attempt_params():
    strategy = ResubmissionStrategy()
    assert strategy.get_params(0) is None
    assert strategy.get_params(-1) is None


def test_apply_returns_false_when_input_missing(tmp_path: Path):
    strategy = ResubmissionStrategy([{"mixing_beta": 0.2}])
    ok = strategy.apply(str(tmp_path / "task"), 0)
    assert ok is False


def test_apply_modifies_and_adds_keys(tmp_path: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True)
    input_file = task_dir / "INPUT"
    input_file.write_text(
        "INPUT_PARAMETERS\n"
        "mixing_beta          0.4\n"
        "# comment\n"
        "kspacing             0.12\n"
    )
    strategy = ResubmissionStrategy([{"mixing_beta": 0.1, "mixing_ndim": 8}])
    ok = strategy.apply(str(task_dir), 0)
    assert ok is True
    content = input_file.read_text()
    assert "0.1" in content
    assert "mixing_ndim" in content


def test_apply_returns_false_without_defined_attempt(tmp_path: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "INPUT").write_text("INPUT_PARAMETERS\n")
    strategy = ResubmissionStrategy([{"mixing_beta": 0.1}])
    assert strategy.apply(str(task_dir), 2) is False
