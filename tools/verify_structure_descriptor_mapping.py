#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from deepmd_verify_common import (
    frame_count,
    iter_system_dirs,
    load_real_atom_types,
    load_system_map,
    resolve_descriptor_path,
    shape_is_descriptor_like,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate one-to-one mapping between DeepMD structures and descriptor files.",
        epilog=(
            "Examples:\n"
            "  python tools/verify_structure_descriptor_mapping.py "
            "--data_dir test/sampled_dpdata --desc_dir test/desc_train --fmt deepmd/npy/mixed\n"
            "  python tools/verify_structure_descriptor_mapping.py "
            "--data_dir test/sampled_dpdata_npy --desc_dir test/desc_train_npy --fmt deepmd/npy"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--data_dir", required=True, help="Structure dataset directory (contains system subdirectories).")
    p.add_argument("--desc_dir", required=True, help="Descriptor directory (one .npy file per system).")
    p.add_argument(
        "--fmt",
        default="auto",
        choices=["auto", "deepmd/npy", "deepmd/npy/mixed"],
        help="Structure format (default: %(default)s).",
    )
    return p.parse_args()


def verify_mapping(data_dir: str, desc_dir: str, fmt: str) -> int:
    systems = load_system_map(data_dir, fmt)
    if not systems:
        print(f"Error: no systems loaded from {data_dir}")
        return 2

    system_dirs = iter_system_dirs(data_dir, systems.keys())
    checked_systems = 0
    missing_desc = 0
    frame_mismatch = 0
    shape_mismatch = 0
    real_type_mismatch = 0
    ok_systems = 0
    total_frames = 0

    for name in sorted(systems.keys()):
        checked_systems += 1
        system = systems[name]
        n_frames = frame_count(system)
        total_frames += n_frames

        desc_path = resolve_descriptor_path(desc_dir, name)
        if desc_path is None:
            missing_desc += 1
            print(f"[MISSING_DESC] {name}")
            continue

        try:
            desc = np.load(desc_path)
        except Exception as exc:
            shape_mismatch += 1
            print(f"[DESC_LOAD_FAIL] {name}: {exc}")
            continue

        if not shape_is_descriptor_like(desc, n_frames):
            frame_mismatch += 1
            print(f"[FRAME_MISMATCH] {name}: frames={n_frames}, desc_shape={tuple(desc.shape)}")
            continue

        if desc.ndim not in (2, 3):
            shape_mismatch += 1
            print(f"[DESC_DIM_UNSUPPORTED] {name}: desc_shape={tuple(desc.shape)}")
            continue

        if desc.ndim == 3:
            atom_count = len(system["atom_types"])
            if desc.shape[1] != atom_count:
                shape_mismatch += 1
                print(f"[ATOM_MISMATCH] {name}: atoms={atom_count}, desc_shape={tuple(desc.shape)}")
                continue

        sys_dir = system_dirs.get(name)
        if sys_dir:
            rt = load_real_atom_types(sys_dir)
            if rt is not None and rt.shape[0] != n_frames:
                real_type_mismatch += 1
                print(f"[REAL_TYPES_MISMATCH] {name}: real_types={rt.shape[0]}, frames={n_frames}")
                continue

        ok_systems += 1

    print("=" * 48)
    print("Structure-Descriptor Mapping Summary")
    print("=" * 48)
    print(f"Checked Systems     : {checked_systems}")
    print(f"Total Frames        : {total_frames}")
    print(f"OK Systems          : {ok_systems}")
    print(f"Missing Descriptors : {missing_desc}")
    print(f"Frame Mismatch      : {frame_mismatch}")
    print(f"Shape Mismatch      : {shape_mismatch}")
    print(f"RealType Mismatch   : {real_type_mismatch}")
    print("=" * 48)

    bad = missing_desc + frame_mismatch + shape_mismatch + real_type_mismatch
    return 0 if bad == 0 else 1


def main() -> None:
    args = parse_args()
    for p in [args.data_dir, args.desc_dir]:
        if not Path(p).exists():
            print(f"Error: path not found: {p}")
            sys.exit(2)
    code = verify_mapping(args.data_dir, args.desc_dir, args.fmt)
    sys.exit(code)


if __name__ == "__main__":
    main()
