#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from deepmd_verify_common import (
    iter_system_dirs,
    load_real_atom_types,
    load_system_map,
    make_fingerprint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate structural consistency between deepmd/npy and deepmd/npy/mixed datasets.",
        epilog=(
            "Examples:\n"
            "  python tools/compare_npy_mixed_structures.py "
            "--npy_dir test/sampled_dpdata_npy --mixed_dir test/sampled_dpdata\n"
            "  python tools/compare_npy_mixed_structures.py "
            "--npy_dir test/sampled_dpdata_npy --mixed_dir test/sampled_dpdata --coord_decimals 5"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--npy_dir", required=True, help="deepmd/npy dataset directory.")
    p.add_argument("--mixed_dir", required=True, help="deepmd/npy/mixed dataset directory.")
    p.add_argument("--coord_decimals", type=int, default=4, help="Coordinate rounding decimals for fingerprints.")
    return p.parse_args()


def build_mixed_fingerprint_counter(mixed_dir: str, coord_decimals: int) -> Counter:
    systems = load_system_map(mixed_dir, "deepmd/npy/mixed")
    sys_dirs = iter_system_dirs(mixed_dir, systems.keys())
    counter: Counter = Counter()
    for name, sys_obj in systems.items():
        rt = None
        sys_dir = sys_dirs.get(name)
        if sys_dir:
            rt = load_real_atom_types(sys_dir)
            if rt is not None and rt.shape[0] != len(sys_obj):
                rt = None
        for i in range(len(sys_obj)):
            fp = make_fingerprint(sys_obj, i, rt, coord_decimals)
            counter[(fp.symbols, fp.coords_bytes)] += 1
    return counter


def compare(npy_dir: str, mixed_dir: str, coord_decimals: int) -> int:
    mixed_counter = build_mixed_fingerprint_counter(mixed_dir, coord_decimals)
    if not mixed_counter:
        print(f"Error: no mixed structures loaded from {mixed_dir}")
        return 2

    npy_systems = load_system_map(npy_dir, "deepmd/npy")
    if not npy_systems:
        print(f"Error: no npy structures loaded from {npy_dir}")
        return 2

    total = 0
    matched = 0
    unmatched = 0

    for name in sorted(npy_systems.keys()):
        sys_obj = npy_systems[name]
        for i in range(len(sys_obj)):
            total += 1
            fp = make_fingerprint(sys_obj, i, None, coord_decimals)
            key = (fp.symbols, fp.coords_bytes)
            if mixed_counter[key] > 0:
                matched += 1
                mixed_counter[key] -= 1
            else:
                unmatched += 1
                if unmatched <= 10:
                    print(f"[UNMATCHED] {name}-{i}")

    print("=" * 44)
    print("NPY vs Mixed Structure Comparison")
    print("=" * 44)
    print(f"Total NPY Frames : {total}")
    print(f"Matched Frames   : {matched}")
    print(f"Unmatched Frames : {unmatched}")
    print("=" * 44)
    return 0 if unmatched == 0 else 1


def main() -> None:
    args = parse_args()
    for p in [args.npy_dir, args.mixed_dir]:
        if not Path(p).exists():
            print(f"Error: path not found: {p}")
            sys.exit(2)
    code = compare(args.npy_dir, args.mixed_dir, args.coord_decimals)
    sys.exit(code)


if __name__ == "__main__":
    main()
