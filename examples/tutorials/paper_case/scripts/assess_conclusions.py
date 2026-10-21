"""Bounded assessment of claims with tutorial inputs; no training."""

import json
import os
import sys
import platform

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
import scipy
import sklearn
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tutorial_support as t
from dpeva.sampling.manager import SamplingManager


def main():
    b = t.load_bundle()
    manager = SamplingManager(
        {
            "direct_n_clusters": t.PAPER["cold_start_clusters"],
            "direct_k": t.PAPER["direct_k"],
            "direct_thr_init": t.PAPER["direct_thr_init"],
        }
    )
    rows = b["cold_start_order"]
    np.random.seed(42)
    cold = manager.execute_sampling(b["features"][rows])
    pc = cold["pca_features"]

    def coverage(coords, indices):
        return manager._calc_coverage(coords, indices)

    historical_idx = np.flatnonzero(rows < 3000)
    current_cov = coverage(pc, cold["selected_indices"])
    historical_cov = coverage(pc, historical_idx)
    random_cov = np.stack(
        [
            coverage(
                pc, np.random.default_rng(seed).choice(len(rows), 3000, replace=False)
            )
            for seed in range(20)
        ]
    )
    report = {
        "kind": "Tutorial conclusion assessment; no training or inference rerun",
        "threadpools": [
            {
                k: pool[k]
                for k in ("internal_api", "version", "num_threads", "architecture")
                if k in pool
            }
            for pool in threadpool_info()
        ],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "cold_start": {
            "input_frames": len(rows),
            "selected": len(cold["selected_indices"]),
            "retained_pcs": pc.shape[1],
            "metric": "Mean occupied-bin coverage across retained PCs, 50 bins per PC, common PCA basis",
            "current_per_pc": current_cov.tolist(),
            "historical_per_pc": historical_cov.tolist(),
            "current_mean": float(current_cov.mean()),
            "historical_mean": float(historical_cov.mean()),
            "random_seeds": list(range(20)),
            "random_mean_each": random_cov.mean(axis=1).tolist(),
            "random_mean": float(random_cov.mean()),
            "random_min": float(random_cov.mean(axis=1).min()),
            "random_max": float(random_cov.mean(axis=1).max()),
        },
    }
    print("cold", report["cold_start"], flush=True)
    uq = b["uq"]
    candidate, accurate, failed, labels = t.classify(uq)
    error = uq.diff_maxf_0_frame.to_numpy()
    q = uq.uq_qbc_for.to_numpy()
    r = uq.uq_rnd_for_rescaled.to_numpy()
    bounds = t.paper_thresholds()
    lo, hi = bounds["qbc"]
    qbc_accurate = q < lo
    dual_notaccurate = labels.to_numpy() != "accurate"
    rescued = qbc_accurate & dual_notaccurate
    uq_report = {
        "error_metric": "Maximum atomic force-vector error per frame [eV/A], from archived base-member predictions; not force RMSE",
        "threshold_note": "0.3, 0.5 and 1.0 eV/A are post-selection diagnostic thresholds, not paper acceptance criteria or sampling inputs",
        "classification_counts": labels.value_counts().to_dict(),
        "per_region_error": {},
        "qbc_only_accurate_count": int(qbc_accurate.sum()),
        "qbc_low_but_dual_notaccurate": int(rescued.sum()),
        "qbc_low_but_dual_candidate": int(
            (qbc_accurate & (labels.to_numpy() == "candidate")).sum()
        ),
        "qbc_low_but_dual_failed": int(
            (qbc_accurate & (labels.to_numpy() == "failed")).sum()
        ),
        "qbc_error_spearman": float(spearmanr(q, error).statistic),
        "rnd_error_spearman": float(spearmanr(r, error).statistic),
    }
    for label in ["accurate", "candidate", "failed"]:
        e = error[labels.to_numpy() == label]
        uq_report["per_region_error"][label] = {
            "median": float(np.median(e)),
            "p95": float(np.quantile(e, 0.95)),
            "max": float(e.max()),
        }
    for threshold in [0.3, 0.5, 1.0]:
        uq_report[f"error_gt_{threshold}"] = {
            "qbc_high_but_dual_accurate": int(
                (
                    (~qbc_accurate)
                    & (labels.to_numpy() == "accurate")
                    & (error > threshold)
                ).sum()
            ),
            "qbc_low": int((qbc_accurate & (error > threshold)).sum()),
            "rescued_from_qbc_low": int((rescued & (error > threshold)).sum()),
            "still_dual_accurate": int(
                ((labels.to_numpy() == "accurate") & (error > threshold)).sum()
            ),
        }
    uq_report["qbc_high_but_dual_accurate"] = int(
        ((~qbc_accurate) & (labels.to_numpy() == "accurate")).sum()
    )
    report["uq"] = uq_report
    print("uq", uq_report, flush=True)
    results = {}
    for joint, use_uq, name in [
        (False, False, "normal_no_uq"),
        (True, False, "joint_no_uq"),
        (False, True, "normal_uq"),
        (True, True, "joint_uq"),
    ]:
        x = t.sample_live(joint=joint, use_uq=use_uq)
        results[name] = x
    report["sampling_counts"] = {
        name: {
            "candidate_count": x["candidate_count"],
            "cluster_budget": x["cluster_budget"],
            "new": len(x["selected_pool_indices"]),
            "training_representatives": x["selected_training_count"],
            "retained_pcs": x["retained_pcs"],
        }
        for name, x in results.items()
    }
    joint = results["joint_uq"]
    pcs = joint["pca_features"]
    n = joint["n_candidates"]
    nn = cKDTree(pcs[n:]).query(pcs[:n], k=1)[0]
    lookup = {int(pool): i for i, pool in enumerate(candidate.index)}
    normal_equal = t.sample_live(
        joint=False, use_uq=True, n_clusters=len(joint["selected_pool_indices"])
    )
    sets = {
        "joint": joint["selected_indices"][joint["selected_indices"] < n],
        "normal_equal_new_count": np.array(
            [lookup[int(i)] for i in normal_equal["selected_pool_indices"]]
        ),
        "historical": np.array(
            [
                lookup[int(i)]
                for i in uq.index[uq.dataname.isin(b["paper_selected"].dataname)]
            ]
        ),
    }
    novelty = {}
    for name, indices in sets.items():
        dist = nn[indices]
        novelty[name] = {
            "n": len(indices),
            "median_nearest_train": float(np.median(dist)),
            "mean_nearest_train": float(dist.mean()),
            "p10_nearest_train": float(np.quantile(dist, 0.1)),
            "coverage_with_training": float(
                coverage(pcs, np.r_[np.arange(n, len(pcs)), indices]).mean()
            ),
        }
    random_novelty = []
    for seed in range(20):
        indices = np.random.default_rng(seed).choice(
            n, len(sets["joint"]), replace=False
        )
        random_novelty.append(
            {
                "median_nearest_train": float(np.median(nn[indices])),
                "coverage_with_training": float(
                    coverage(pcs, np.r_[np.arange(n, len(pcs)), indices]).mean()
                ),
            }
        )
    report["joint_diagnostic"] = {
        "metric_space": "Common weighted PCA fit on UQ candidates + historical training features; dimensionless distances",
        "selection_fit": "Each selection method retains its own original PCA/clustering; comparisons project selected identities into the common joint PCA",
        "methods": novelty,
        "random_seeds": list(range(20)),
        "random": random_novelty,
        "not_a_paper_metric": True,
    }
    report["training_fraction"] = {
        "current": (3000 + len(sets["joint"])) / 30656,
        "historical": 3843 / 30656,
    }
    (ROOT / "work/conclusions").mkdir(parents=True, exist_ok=True)
    (ROOT / "work/conclusions/assessment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("counts", report["sampling_counts"])
    print("novelty", novelty)
    print("fractions", report["training_fraction"])


if __name__ == "__main__":
    main()
