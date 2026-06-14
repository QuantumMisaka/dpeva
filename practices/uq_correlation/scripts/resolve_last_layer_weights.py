#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpeva.uncertain.dpose import resolve_last_layer_weights  # noqa: E402


def infer_feature_dimension(feature_dir: Path) -> int:
    files = sorted(feature_dir.glob("**/*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy feature files found in {feature_dir}")
    arr = np.asarray(np.load(files[0]))
    if arr.ndim not in {2, 3}:
        raise ValueError(f"Feature file {files[0]} must be 2D or 3D, got {arr.shape}")
    return int(arr.shape[-1])


def run(
    feature_dimension: int | None,
    output_path: Path,
    model_path: Path | None,
    model_head: str | None,
    weights_path: Path | None,
    candidate_feature_dir: Path,
) -> None:
    if feature_dimension is None:
        feature_dimension = infer_feature_dimension(candidate_feature_dir)
    weights = resolve_last_layer_weights(
        feature_dimension=feature_dimension,
        last_layer_weights_path=weights_path,
        model_path=model_path if weights_path is None else None,
        model_head=model_head,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(weights))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dimension", type=int, default=None)
    parser.add_argument("--candidate-feature-dir", type=Path, default=ROOT / "outputs" / "features" / "candidate_last_layer")
    parser.add_argument("--model-path", type=Path, default=ROOT / "work" / "dpa4_mini_ensemble" / "0" / "model.ckpt.pt")
    parser.add_argument("--model-head", default="RANDOM")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=ROOT / "outputs" / "last_layer_weights.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        feature_dimension=args.feature_dimension,
        output_path=args.output_path,
        model_path=args.model_path,
        model_head=args.model_head,
        weights_path=args.weights_path,
        candidate_feature_dir=args.candidate_feature_dir,
    )


if __name__ == "__main__":
    main()
