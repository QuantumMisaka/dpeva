from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from dpeva.config_migration import MigrationResult


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "fp11_1344_recover_after_false_finish.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fp11_1344_recover_after_false_finish", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recover_passes_normalized_config_to_labeling_model(monkeypatch, tmp_path):
    module = load_module()
    captured = {}
    normalized = {"work_dir": str(tmp_path), "attempt_params": []}

    monkeypatch.setattr(
        module,
        "load_and_resolve_config",
        lambda _path: MigrationResult(normalized=normalized, warnings=()),
    )

    def make_config(**values):
        captured["config"] = values
        return SimpleNamespace(attempt_params=[])

    class FakeWorkflow:
        def __init__(self, config):
            self.config = config
            self.manager = SimpleNamespace(
                process_results=lambda _dirs: ([], []),
            )

        def _resolve_packed_job_dirs(self):
            return [tmp_path / "packed"]

        def run_extract(self, packed_job_dirs):
            captured["extract_dirs"] = packed_job_dirs

        def run_postprocess(self):
            captured["postprocess"] = True

    monkeypatch.setattr(module, "LabelingConfig", make_config)
    monkeypatch.setattr(module, "LabelingWorkflow", FakeWorkflow)

    assert module.recover(tmp_path / "config.json", [], next_attempt=2, interval=0) == 0
    assert captured["config"] == normalized
    assert captured["extract_dirs"] == [tmp_path / "packed"]
    assert captured["postprocess"] is True
