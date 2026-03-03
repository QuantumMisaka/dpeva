#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deepmd_verify_common import frame_count, load_system_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quickly report system count and total frame count for a DeepMD dataset.",
        epilog=(
            "Examples:\n"
            "  python tools/count_deepmd_system_frames.py --data_dir test/sampled_dpdata --fmt deepmd/npy/mixed\n"
            "  python tools/count_deepmd_system_frames.py --data_dir test/sampled_dpdata_npy --fmt deepmd/npy --detail"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--data_dir", required=True, help="Structure dataset directory (contains system subdirectories).")
    p.add_argument(
        "--fmt",
        default="auto",
        choices=["auto", "deepmd/npy", "deepmd/npy/mixed"],
        help="Structure format (default: %(default)s).",
    )
    p.add_argument("--detail", action="store_true", help="Print per-system frame details.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.data_dir).exists():
        print(f"Error: path not found: {args.data_dir}")
        sys.exit(2)

    systems = load_system_map(args.data_dir, args.fmt)
    if not systems:
        print(f"Error: no systems loaded from {args.data_dir}")
        sys.exit(2)

    total_frames = 0
    for n in sorted(systems.keys()):
        total_frames += frame_count(systems[n])

    print("=" * 36)
    print("DeepMD Dataset Summary")
    print("=" * 36)
    print(f"Systems     : {len(systems)}")
    print(f"Total Frames: {total_frames}")
    print("=" * 36)

    if args.detail:
        for n in sorted(systems.keys()):
            print(f"{n}\t{frame_count(systems[n])}")


if __name__ == "__main__":
    main()
