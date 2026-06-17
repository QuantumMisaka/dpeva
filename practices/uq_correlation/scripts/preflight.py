#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dpdata


ROOT = Path(__file__).resolve().parents[1]


def system_dirs(data_dir: Path) -> list[Path]:
    if (data_dir / "type.raw").exists() or list(data_dir.glob("set.*")):
        return [data_dir]
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and ((path / "type.raw").exists() or list(path.glob("set.*")))
    )


def summarize_dataset(data_dir: Path) -> dict:
    systems = []
    type_maps = set()
    total_frames = 0
    for path in system_dirs(data_dir):
        sys = dpdata.LabeledSystem(str(path), fmt="deepmd/npy")
        type_map_path = path / "type_map.raw"
        type_map = tuple(type_map_path.read_text(encoding="utf-8").split()) if type_map_path.exists() else tuple(sys["atom_names"])
        type_maps.add(type_map)
        nframes = int(sys.get_nframes())
        total_frames += nframes
        systems.append(
            {
                "name": path.name,
                "path": str(path),
                "nframes": nframes,
                "natoms": int(len(sys["atom_types"])),
                "type_map": list(type_map),
            }
        )
    return {
        "data_dir": str(data_dir),
        "n_systems": len(systems),
        "n_frames": total_frames,
        "type_maps": [list(item) for item in sorted(type_maps)],
        "systems": systems,
    }


def run(train_dir: Path, candidate_dir: Path, output_path: Path) -> None:
    train = summarize_dataset(train_dir)
    candidate = summarize_dataset(candidate_dir)
    train_names = {item["name"] for item in train["systems"]}
    candidate_names = {item["name"] for item in candidate["systems"]}
    summary = {
        "train": train,
        "candidate": candidate,
        "overlap_system_names": sorted(train_names & candidate_names),
        "candidate_only_system_names": sorted(candidate_names - train_names),
        "train_only_system_names": sorted(train_names - candidate_names),
        "interpretation": "existing training pool vs labeled candidate pool; no frame-level deduplication is applied",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=ROOT.parent / "sampled_dpdata")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT.parent / "other_dpdata")
    parser.add_argument("--output-path", type=Path, default=ROOT / "outputs" / "preflight_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.train_dir, args.candidate_dir, args.output_path)


if __name__ == "__main__":
    main()
