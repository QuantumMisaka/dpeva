#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpeva.uncertain.dpose import resolve_last_layer_weights  # noqa: E402
from dpeva.uncertain.manager import UQManager  # noqa: E402


def load_feature_sums(feature_dir: Path, normalization: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    files = sorted(feature_dir.glob("**/*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy feature files found in {feature_dir}")
    names: list[str] = []
    features: list[np.ndarray] = []
    atom_counts: list[np.ndarray] = []
    for path in files:
        sys_name = path.relative_to(feature_dir).with_suffix("").as_posix()
        arr = np.asarray(np.load(path))
        if arr.ndim == 3:
            frame_features = arr.mean(axis=1) if normalization == "mean" else arr.sum(axis=1)
            counts = np.full(arr.shape[0], arr.shape[1], dtype=float)
        elif arr.ndim == 2:
            frame_features = arr
            counts = np.ones(arr.shape[0], dtype=float)
        else:
            raise ValueError(f"Feature file {path} must be 2D or 3D, got {arr.shape}")
        names.extend([f"{sys_name}-{idx}" for idx in range(frame_features.shape[0])])
        features.append(frame_features)
        atom_counts.append(counts)
    return names, np.vstack(features), np.concatenate(atom_counts)


def run(
    train_feature_dir: Path,
    candidate_feature_dir: Path,
    candidate_energy_path: Path,
    model_path: Path,
    model_head: str,
    output_csv: Path,
    ensemble_output_path: Path,
    n_members: int = 8,
    normalization: str = "mean",
    regularizer: float = 1e-8,
    random_seed: int = 19090,
    weights_path: Path | None = None,
) -> None:
    _, train_features, _ = load_feature_sums(train_feature_dir, normalization)
    names, candidate_features, atom_counts = load_feature_sums(candidate_feature_dir, normalization)
    mean_energy = np.asarray(np.load(candidate_energy_path), dtype=float).reshape(-1)
    if mean_energy.shape[0] != candidate_features.shape[0]:
        raise ValueError("candidate_energy length must match candidate feature frame count")
    weights = resolve_last_layer_weights(
        feature_dimension=int(candidate_features.shape[1]),
        last_layer_weights_path=weights_path,
        model_path=model_path if weights_path is None else None,
        model_head=model_head,
    )
    manager = UQManager(
        project_dir=str(ROOT / "work" / "dpa4_mini_ensemble"),
        testing_dir="test_val",
        testing_head="results",
        uq_config={
            "llpr_regularizer": regularizer,
            "llpr_feature_normalization": normalization,
            "llpr_num_ensemble_members": n_members,
            "llpr_random_seed": random_seed,
            "llpr_targets": "energy",
            "llpr_ensemble_output_path": str(ensemble_output_path),
            "llpr_weight_source": str(weights_path or model_path),
        },
        num_models=4,
    )
    result = manager.run_llpr_analysis(
        train_features=train_features,
        candidate_features=candidate_features,
        candidate_atom_counts=atom_counts,
        mean_energy=mean_energy,
        last_layer_weights=weights,
    )
    columns = {"dataname": names}
    for key, value in result.items():
        if key == "energy_ensemble":
            continue
        if np.isscalar(value):
            columns[key] = [value] * len(names)
        else:
            arr = np.asarray(value)
            if arr.ndim == 1 and arr.shape[0] == len(names):
                columns[key] = arr
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-feature-dir", type=Path, default=ROOT / "outputs" / "features" / "train_last_layer")
    parser.add_argument("--candidate-feature-dir", type=Path, default=ROOT / "outputs" / "features" / "candidate_last_layer")
    parser.add_argument("--candidate-energy-path", type=Path, default=ROOT / "outputs" / "candidate_energy.npy")
    parser.add_argument("--model-path", type=Path, default=ROOT / "work" / "dpa4_mini_ensemble" / "0" / "model.ckpt.pt")
    parser.add_argument("--model-head", default="RANDOM")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--n-members", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "llpr_energy_uq.csv")
    parser.add_argument("--ensemble-output-path", type=Path, default=ROOT / "outputs" / "energy_ensemble.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
