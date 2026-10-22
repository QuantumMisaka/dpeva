"""Label output-format contract tests."""

from pathlib import Path

from dpeva.config import LabelingConfig


def _config(tmp_path: Path, **updates):
    values = {
        "work_dir": str(tmp_path),
        "input_data_path": str(tmp_path / "data"),
        "submission": {"backend": "local"},
        "dft_params": {},
        "attempt_params": [],
        "pp_dir": "/tmp/pp",
        "orb_dir": "/tmp/orb",
    }
    values.update(updates)
    return LabelingConfig(**values)


def test_labeling_accepts_lmdb_for_cleaned_and_integration_outputs(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        output_format="deepmd/lmdb",
        integration_output_format="deepmd/lmdb",
    )

    assert config.output_format == "deepmd/lmdb"
    assert config.integration_output_format == "deepmd/lmdb"
