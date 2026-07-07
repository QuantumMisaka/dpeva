import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


_TASK_REQUIRED_FIELDS = {"index", "name", "working_dir", "argv"}


@dataclass
class ArrayTaskSpec:
    index: int
    name: str
    working_dir: Path
    argv: Sequence[str]

    def to_json_dict(self) -> dict:
        if self.index < 0:
            raise ValueError("array task index must be >= 0")
        if not self.argv:
            raise ValueError("array task argv must be non-empty")

        return {
            "index": self.index,
            "name": str(self.name),
            "working_dir": str(Path(self.working_dir).resolve()),
            "argv": [str(arg) for arg in self.argv],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "ArrayTaskSpec":
        if not isinstance(data, dict):
            raise ValueError("array manifest item must be a JSON object")

        missing = _TASK_REQUIRED_FIELDS - data.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"array manifest item missing required fields: {fields}")

        if not isinstance(data["index"], int) or isinstance(data["index"], bool):
            raise ValueError("array task index must be an integer")
        if not isinstance(data["name"], str):
            raise ValueError("array task name must be a string")
        if not isinstance(data["working_dir"], str):
            raise ValueError("array task working_dir must be a string")
        if not isinstance(data["argv"], list) or not data["argv"]:
            raise ValueError("array task argv must be a non-empty JSON list")

        task = cls(
            index=data["index"],
            name=data["name"],
            working_dir=Path(data["working_dir"]),
            argv=[str(arg) for arg in data["argv"]],
        )
        task.to_json_dict()
        return task


def _validate_task_indices(tasks: Sequence[ArrayTaskSpec]) -> None:
    expected = list(range(len(tasks)))
    actual = [task.index for task in tasks]
    if actual != expected:
        raise ValueError("array task indices must be exactly 0..len(tasks)-1")


def write_array_manifest(
    tasks: Iterable[ArrayTaskSpec],
    manifest_path: str | Path,
) -> None:
    task_list = list(tasks)
    _validate_task_indices(task_list)
    payload = [task.to_json_dict() for task in task_list]

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_array_manifest(manifest_path: str | Path) -> List[ArrayTaskSpec]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("array manifest must contain a JSON list")

    tasks = [ArrayTaskSpec.from_json_dict(item) for item in payload]
    _validate_task_indices(tasks)
    return tasks


def get_array_task(
    tasks: Sequence[ArrayTaskSpec],
    task_id: Optional[int] = None,
) -> ArrayTaskSpec:
    task_list = list(tasks)
    _validate_task_indices(task_list)

    if task_id is None:
        raw_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_task_id is None:
            raise ValueError("SLURM_ARRAY_TASK_ID is not set")
        task_id = int(raw_task_id)

    if task_id < 0 or task_id >= len(task_list):
        raise IndexError(f"array task index out of range: {task_id}")
    return task_list[task_id]


def run_array_task(
    manifest_path: str | Path,
    task_id: Optional[int] = None,
) -> None:
    tasks = load_array_manifest(manifest_path)
    task = get_array_task(tasks, task_id=task_id)
    work_dir = Path(task.working_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"array task working_dir does not exist: {work_dir}")

    print(f"Running array task {task.index}: {task.name}")
    print(f"Working directory: {work_dir}")
    print(f"Command: {' '.join(shlex.quote(str(arg)) for arg in task.argv)}")
    subprocess.run(list(task.argv), cwd=str(work_dir), check=True)


def build_array_command(manifest_path: str | Path) -> str:
    manifest = Path(manifest_path).resolve()
    worker = Path(__file__).resolve()
    return (
        f"{shlex.quote(sys.executable)} "
        f"-u {shlex.quote(str(worker))} "
        f"{shlex.quote(str(manifest))}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python array.py <manifest_path>", file=sys.stderr)
        return 2

    run_array_task(args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
