import json

from dpeva.labeling.packer import TaskPacker


def make_task(task_dir, dataset, stru_type, task_name):
    task_dir.mkdir(parents=True)
    (task_dir / "INPUT").touch()
    (task_dir / "task_meta.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset,
                "stru_type": stru_type,
                "task_name": task_name,
            }
        ),
        encoding="utf-8",
    )


def test_pack_disambiguates_cross_dataset_task_name_collision(tmp_path):
    root = tmp_path / "inputs"
    make_task(root / "DS1" / "cluster" / "same_0", "DS1", "cluster", "same_0")
    make_task(root / "DS2" / "cluster" / "same_0", "DS2", "cluster", "same_0")

    packed_dirs = TaskPacker(tasks_per_job=50).pack(root)

    assert [path.name for path in packed_dirs] == ["N_50_0"]
    assert (root / "N_50_0" / "same_0").exists()
    assert (root / "N_50_0" / "DS2__cluster__same_0").exists()


def test_pack_keeps_non_colliding_task_names_unchanged(tmp_path):
    root = tmp_path / "inputs"
    make_task(root / "DS1" / "cluster" / "a_0", "DS1", "cluster", "a_0")
    make_task(root / "DS2" / "cluster" / "b_0", "DS2", "cluster", "b_0")

    TaskPacker(tasks_per_job=50).pack(root)

    assert (root / "N_50_0" / "a_0").exists()
    assert (root / "N_50_0" / "b_0").exists()
