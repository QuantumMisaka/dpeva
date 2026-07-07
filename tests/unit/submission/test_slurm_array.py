import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from dpeva.submission.array import (
    ArrayTaskSpec,
    build_array_command,
    load_array_manifest,
    main,
    run_array_task,
    write_array_manifest,
)
from dpeva.submission.manager import JobManager
from dpeva.submission.names import normalize_slurm_job_name
from dpeva.submission.templates import JobConfig


def test_write_load_array_manifest_round_trip(tmp_path):
    work_dir_0 = tmp_path / "task0"
    work_dir_1 = tmp_path / "task1"
    work_dir_0.mkdir()
    work_dir_1.mkdir()
    tasks = [
        ArrayTaskSpec(index=0, name="first", working_dir=work_dir_0, argv=["echo", "0"]),
        ArrayTaskSpec(index=1, name="second", working_dir=work_dir_1, argv=["echo", "1"]),
    ]
    manifest_path = tmp_path / "manifests" / "array.json"

    write_array_manifest(tasks, manifest_path)

    assert manifest_path.read_text().endswith("\n")
    assert load_array_manifest(manifest_path) == [
        ArrayTaskSpec(
            index=0,
            name="first",
            working_dir=work_dir_0.resolve(),
            argv=["echo", "0"],
        ),
        ArrayTaskSpec(
            index=1,
            name="second",
            working_dir=work_dir_1.resolve(),
            argv=["echo", "1"],
        ),
    ]


def test_write_array_manifest_rejects_non_contiguous_indices(tmp_path):
    work_dir = tmp_path / "task"
    work_dir.mkdir()
    tasks = [
        ArrayTaskSpec(index=0, name="first", working_dir=work_dir, argv=["echo", "0"]),
        ArrayTaskSpec(index=2, name="third", working_dir=work_dir, argv=["echo", "2"]),
    ]

    with pytest.raises(ValueError, match="0..len"):
        write_array_manifest(tasks, tmp_path / "array.json")


def test_load_array_manifest_rejects_non_dict_items(tmp_path):
    manifest_path = tmp_path / "array.json"
    manifest_path.write_text(json.dumps(["not a task"]) + "\n")

    with pytest.raises(ValueError, match="JSON object"):
        load_array_manifest(manifest_path)


def test_load_array_manifest_rejects_argv_string(tmp_path):
    manifest_path = tmp_path / "array.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "name": "task",
                    "working_dir": str(tmp_path),
                    "argv": "echo 0",
                }
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="argv"):
        load_array_manifest(manifest_path)


def test_load_array_manifest_rejects_missing_required_fields(tmp_path):
    manifest_path = tmp_path / "array.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "name": "task",
                    "working_dir": str(tmp_path),
                }
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_array_manifest(manifest_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", "0"),
        ("name", ["task"]),
        ("working_dir", 3),
    ],
)
def test_load_array_manifest_rejects_bad_required_field_types(
    tmp_path, field, value
):
    manifest_path = tmp_path / "array.json"
    item = {
        "index": 0,
        "name": "task",
        "working_dir": str(tmp_path),
        "argv": ["echo", "0"],
    }
    item[field] = value
    manifest_path.write_text(json.dumps([item]) + "\n")

    with pytest.raises(ValueError, match=field):
        load_array_manifest(manifest_path)


def test_load_array_manifest_rejects_empty_argv(tmp_path):
    manifest_path = tmp_path / "array.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "name": "task",
                    "working_dir": str(tmp_path),
                    "argv": [],
                }
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="argv"):
        load_array_manifest(manifest_path)


def test_run_array_task_uses_slurm_array_task_id_and_task_working_dir(
    tmp_path, monkeypatch
):
    work_dir_0 = tmp_path / "task0"
    work_dir_1 = tmp_path / "task1"
    work_dir_0.mkdir()
    work_dir_1.mkdir()
    manifest_path = tmp_path / "array.json"
    write_array_manifest(
        [
            ArrayTaskSpec(index=0, name="first", working_dir=work_dir_0, argv=["echo", "0"]),
            ArrayTaskSpec(index=1, name="second", working_dir=work_dir_1, argv=["echo", "1"]),
        ],
        manifest_path,
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")

    with patch("dpeva.submission.array.subprocess.run") as mock_run:
        run_array_task(manifest_path)

    mock_run.assert_called_once_with(
        ["echo", "1"],
        cwd=str(work_dir_1.resolve()),
        check=True,
    )


def test_build_array_command_contains_module_and_manifest_path(tmp_path):
    manifest_path = tmp_path / "array manifest.json"

    command = build_array_command(manifest_path)

    assert "-u" in command
    assert "array.py" in command
    assert "-m dpeva.submission.array" not in command
    assert str(manifest_path.resolve()) in command


def test_array_worker_script_execution_does_not_emit_import_warnings():
    worker_path = Path(sys.modules[main.__module__].__file__)
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            str(worker_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Warning" not in result.stderr
    assert "Usage: python array.py" in result.stderr


def test_submission_package_lazily_exports_array_task_spec():
    import dpeva.submission as submission

    assert submission.ArrayTaskSpec is ArrayTaskSpec
    assert submission.JobConfig is JobConfig
    assert submission.JobManager is JobManager
    assert submission.normalize_slurm_job_name is normalize_slurm_job_name


def test_submit_array_writes_manifest_and_single_script(tmp_path):
    work_dir_0 = tmp_path / "task0"
    work_dir_1 = tmp_path / "task1"
    work_dir_0.mkdir()
    work_dir_1.mkdir()
    tasks = [
        ArrayTaskSpec(index=0, name="first", working_dir=work_dir_0, argv=["echo", "0"]),
        ArrayTaskSpec(index=1, name="second", working_dir=work_dir_1, argv=["echo", "1"]),
    ]
    manager = JobManager(mode="slurm")
    job_config = JobConfig(job_name="array_job", command="placeholder")
    manifest_path = tmp_path / "array.json"
    script_path = tmp_path / "array.slurm"

    with patch.object(manager, "submit", return_value="Submitted batch job 24680") as submit:
        job_id = manager.submit_array(
            tasks,
            job_config,
            str(manifest_path),
            str(script_path),
            working_dir=str(tmp_path),
            array_task_limit=1,
        )

    assert job_id == "24680"
    assert manifest_path.exists()
    assert script_path.exists()
    content = script_path.read_text()
    assert "#SBATCH --array=0-1%1" in content
    assert "#SBATCH -o slurm-%A_%a.out" in content
    assert "#SBATCH -e slurm-%A_%a.err" in content
    assert "array.py" in content
    assert "-m dpeva.submission.array" not in content
    assert str(manifest_path.resolve()) in content
    submit.assert_called_once_with(str(script_path), working_dir=str(tmp_path))


def test_submit_array_creates_missing_working_dir_before_submit(tmp_path):
    task_dir = tmp_path / "task0"
    task_dir.mkdir()
    tasks = [
        ArrayTaskSpec(index=0, name="first", working_dir=task_dir, argv=["echo", "0"]),
    ]
    manager = JobManager(mode="slurm")
    job_config = JobConfig(job_name="array_job", command="placeholder")
    manifest_path = tmp_path / "array.json"
    script_path = tmp_path / "array.slurm"
    working_dir = tmp_path / "missing-working-dir"

    def submit_requires_existing_dir(script_path, working_dir="."):
        assert Path(working_dir).is_dir()
        return "Submitted batch job 24680"

    with patch.object(manager, "submit", side_effect=submit_requires_existing_dir) as submit:
        job_id = manager.submit_array(
            tasks,
            job_config,
            str(manifest_path),
            str(script_path),
            working_dir=str(working_dir),
        )

    assert job_id == "24680"
    submit.assert_called_once_with(str(script_path), working_dir=str(working_dir))


def test_submit_array_rejects_local_mode(tmp_path):
    work_dir = tmp_path / "task"
    work_dir.mkdir()
    tasks = [
        ArrayTaskSpec(index=0, name="first", working_dir=work_dir, argv=["echo", "0"]),
    ]
    manager = JobManager(mode="local")
    job_config = JobConfig(job_name="array_job", command="placeholder")

    with patch.object(manager, "submit", return_value="Submitted batch job 24680") as submit:
        with pytest.raises(ValueError, match="slurm"):
            manager.submit_array(
                tasks,
                job_config,
                str(tmp_path / "array.json"),
                str(tmp_path / "array.sh"),
                working_dir=str(tmp_path),
            )

    submit.assert_not_called()


def test_run_array_task_executes_in_working_directory(tmp_path, monkeypatch):
    work_dir = tmp_path / "task"
    work_dir.mkdir()
    (work_dir / "mark.py").write_text(
        "from pathlib import Path\n"
        "Path('ran.txt').write_text('ok')\n"
    )
    manifest_path = tmp_path / "array.json"
    write_array_manifest(
        [
            ArrayTaskSpec(
                index=0,
                name="mark",
                working_dir=work_dir,
                argv=[sys.executable, "mark.py"],
            ),
        ],
        manifest_path,
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    run_array_task(manifest_path)

    assert (work_dir / "ran.txt").read_text() == "ok"
    assert not (tmp_path / "ran.txt").exists()
