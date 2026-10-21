"""Record numerical stages to investigate pointwise paper reproducibility.

Historical labels are used only for output comparison. Joint candidates are
recomputed by the tutorial auto-UQ implementation. No training is performed.
"""

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parents[2] / "src"))
from dpeva.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters


def main():
    p = argparse.ArgumentParser(
        description="逐层记录论文采样复算，并比较历史名单；不替换选择结果。"
    )
    p.add_argument("name", help="输出文件名前缀，保存到 work/reproduction/")
    p.add_argument("--stage", choices=["cold", "joint"], default="joint")
    p.add_argument("--order", choices=["C", "F"], default="C")
    p.add_argument(
        "--solver", choices=["auto", "full", "covariance_eigh"], default="auto"
    )
    p.add_argument(
        "--pcs-from",
        type=Path,
        help="固定已保存的 PCA 坐标，做跨环境对照；需同名 JSON 验证输入",
    )
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        p.error("name 只能包含英文字母、数字、下划线和连字符")
    out = root / "work/reproduction"
    out.mkdir(parents=True, exist_ok=True)
    with (root / "data/paper_selected.csv").open() as h:
        paper = {r["dataname"] for r in csv.DictReader(h)}
    with gzip.open(root / "data/manifest.csv.gz", "rt") as h:
        manifest = list(csv.DictReader(h))
    with gzip.open(root / "data/uq.csv.gz", "rt") as h:
        uq = list(csv.DictReader(h))
    if args.stage == "joint":
        with np.load(root / "data/iteration_features.npz") as archive:
            a = {key: archive[key] for key in archive.files}
        sys.path.insert(0, str(root))
        import tutorial_support as t

        idx = t.classify(t.load_bundle()["uq"])[0].index.to_numpy()
        n = len(idx)
        x = np.vstack([a["features"][3000 + idx], a["features"][a["training_order"]]])
        names = [uq[i]["dataname"] for i in idx]
    else:
        with np.load(root / "data/features.npz") as archive:
            a = {key: archive[key] for key in archive.files}
        idx = a["paper_order"]
        x = a["features"][idx]
        n = len(x)
        names = [manifest[i]["source_dataname"] for i in idx]
        paper = {r["source_dataname"] for r in manifest[:3000]}
    x = np.array(x, order=args.order)
    s = DIRECTSampler(
        clustering=BirchClustering(n=3000, threshold_init=0.1),
        select_k_from_clusters=SelectKFromClusters(k=1),
    )
    s.pca.pca.set_params(svd_solver=args.solver)
    z = s.scaler.fit_transform(x)
    pc = s.pca.fit_transform(z)
    if args.pcs_from:
        reference = json.loads(args.pcs_from.with_suffix(".json").read_text())
        if (
            reference["hashes"]["input"] != hashlib.sha256(x.tobytes()).hexdigest()
            or reference["stage"] != args.stage
        ):
            p.error("PCA 参考文件的输入与本次不一致")
        with np.load(args.pcs_from, allow_pickle=False) as saved:
            pc = saved["pcs"]
        if hashlib.sha256(pc.tobytes()).hexdigest() != reference["hashes"]["pcs"]:
            p.error("PCA 参考数组与同名 JSON 的校验值不一致")
        if (
            pc.ndim != 2
            or pc.shape[0] != len(x)
            or pc.shape[1] == 0
            or not np.isfinite(pc).all()
        ):
            p.error("PCA 参考数组维度或数值不合法")
    clusters = s.clustering.fit_transform(pc)
    res = s.select_k_from_clusters.fit_transform(clusters)
    selected = np.array(res["selected_indices"])
    new = selected[selected < n]
    actual = {names[i] for i in new}
    report = {
        "name": args.name,
        "pca_source": str(args.pcs_from) if args.pcs_from else "recomputed",
        "pca_metadata_source": "local recomputation; frozen PCs override coordinates only",
        "stage": args.stage,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "machine": platform.machine(),
        "threadpools": threadpool_info(),
        "shape": list(x.shape),
        "order": args.order,
        "solver": s.pca.pca._fit_svd_solver,
        "threshold": s.clustering.threshold_init,
        "pcs": pc.shape[1],
        "total": len(selected),
        "new": len(actual),
        "historical": len(paper),
        "overlap": len(actual & paper),
        "only_current": sorted(actual - paper),
        "only_historical": sorted(paper - actual),
        "hashes": {
            k: hashlib.sha256(v.tobytes()).hexdigest()
            for k, v in [
                ("input", x),
                ("scaled", z),
                ("pcs", pc),
                ("labels", clusters["labels"]),
            ]
        },
    }
    base = out / args.name
    base.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        base.with_suffix(".npz"),
        scaled=z,
        pcs=pc,
        labels=clusters["labels"],
        selected=selected,
        centers=np.array(
            [clusters["label_centers"][i] for i in sorted(clusters["label_centers"])]
        ),
        local_eigenvalues=s.pca.pca.explained_variance_,
        subcenters=s.clustering.model.subcluster_centers_,
        sublabels=s.clustering.model.subcluster_labels_,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "name",
                    "numpy",
                    "scipy",
                    "sklearn",
                    "pcs",
                    "threshold",
                    "new",
                    "overlap",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
