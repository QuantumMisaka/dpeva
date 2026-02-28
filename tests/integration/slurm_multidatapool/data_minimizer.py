import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class MinimalDatasetSpec:
    candidate_pool: str
    candidate_system: str
    candidate_frames: int
    train_system: str


def prepare_minimal_dataset(src_root: Path, dst_root: Path, spec: MinimalDatasetSpec) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    _copy_base_model(src_root, dst_root)
    _copy_candidate_system(src_root, dst_root, spec)
    _copy_training_system(src_root, dst_root, spec)


def write_minimal_training_input(base_input_path: Path, out_path: Path, systems_path: str) -> None:
    cfg = json.loads(base_input_path.read_text())
    cfg["model"]["type_map"] = ["H", "C", "O", "Fe"]
    training = cfg.get("training", {})
    training_data = training.get("training_data", {})
    training_data["systems"] = systems_path
    if "batch_size" not in training_data:
        training_data["batch_size"] = "auto:32"
    training["training_data"] = training_data
    training["numb_steps"] = 10
    training["warmup_steps"] = 0 # Ensure warmup < numb_steps
    training["save_freq"] = 10
    training["disp_freq"] = 1
    cfg["training"] = training
    out_path.write_text(json.dumps(cfg, indent=2))


def _copy_base_model(src_root: Path, dst_root: Path) -> None:
    src = src_root / "DPA-3.1-3M.pt"
    if not src.exists():
        # Fallback to old structure if needed, or error out
        raise FileNotFoundError(f"Model file not found at {src}")
    shutil.copy2(src, dst_root / "DPA-3.1-3M.pt")


def _copy_candidate_system(src_root: Path, dst_root: Path, spec: MinimalDatasetSpec) -> None:
    src_sys = src_root / "other_dpdata_all" / spec.candidate_pool / spec.candidate_system
    dst_sys = dst_root / "other_dpdata_all" / spec.candidate_pool / spec.candidate_system
    _copy_tree_files(src_sys, dst_sys, include_globs=["type.raw", "type_map.raw"])
    _slice_set_dir(
        src_sys / "set.000",
        dst_sys / "set.000",
        frames=spec.candidate_frames,
        files=["coord.npy", "box.npy"],
    )


def _copy_training_system(src_root: Path, dst_root: Path, spec: MinimalDatasetSpec) -> None:
    src_sys = src_root / "sampled_dpdata" / spec.train_system
    dst_sys = dst_root / "sampled_dpdata" / spec.train_system
    _copy_tree_files(src_sys, dst_sys, include_globs=["type.raw", "type_map.raw"])
    _slice_set_dir(
        src_sys / "set.000",
        dst_sys / "set.000",
        frames=1,
        files=[
            "coord.npy",
            "box.npy",
            "energy.npy",
            "force.npy",
            "virial.npy",
            "spin.npy",
            "real_atom_types.npy",
        ],
    )


def _copy_tree_files(src_dir: Path, dst_dir: Path, include_globs: Iterable[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pat in include_globs:
        for p in src_dir.glob(pat):
            if p.is_file():
                shutil.copy2(p, dst_dir / p.name)


def _slice_set_dir(src_set: Path, dst_set: Path, frames: int, files: Iterable[str]) -> None:
    dst_set.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = src_set / name
        dst = dst_set / name
        arr = np.load(src)
        if arr.ndim >= 1 and arr.shape[0] >= frames:
            arr = arr[:frames]
        np.save(dst, arr)


def copy_reference_outputs(src_root: Path, dst_root: Path, overwrite: bool = False) -> None:
    src = src_root / "dpeva_uq_result"
    dst = dst_root / "reference_outputs" / "dpeva_uq_result"
    if dst.exists():
        if not overwrite:
            return
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)

