#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from dpeva.labeling.manager import LabelingManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair labeling branch statistics for historical workdirs."
    )
    parser.add_argument(
        "--workdir",
        action="append",
        default=[],
        help="Single labeling workdir path, can be repeated.",
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Root directory for batch scan.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern used under --root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any directory fails or report is untrusted.",
    )
    return parser


def discover_workdirs(explicit: List[str], root: str, pattern: str) -> List[Path]:
    candidates: List[Path] = []
    for item in explicit:
        p = Path(item).expanduser().resolve()
        if p.is_dir():
            candidates.append(p)
    if root:
        root_path = Path(root).expanduser().resolve()
        if root_path.is_dir():
            for p in sorted(root_path.glob(pattern)):
                if not p.is_dir():
                    continue
                if (p / "CONVERGED").exists() or (p / "inputs").exists():
                    candidates.append(p.resolve())
    deduped: List[Path] = []
    seen = set()
    for p in candidates:
        if str(p) in seen:
            continue
        deduped.append(p)
        seen.add(str(p))
    return deduped


def run_repair(workdir: Path) -> Tuple[bool, Dict]:
    cfg = {
        "work_dir": str(workdir),
        "dft_params": {},
        "attempt_params": [],
    }
    manager = LabelingManager(cfg)
    report = manager.rebuild_statistics_report()
    trusted = bool(report.get("consistency", {}).get("trusted", False))
    return trusted, report


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    workdirs = discover_workdirs(args.workdir, args.root, args.pattern)
    if not workdirs:
        print("No valid workdirs found.")
        return 1

    total = len(workdirs)
    success = 0
    untrusted = 0
    failed = 0
    failed_items: List[Dict[str, str]] = []

    for wd in workdirs:
        print(f"[START] {wd}")
        try:
            trusted, report = run_repair(wd)
            report_path = wd / "outputs" / "labeling_stats_report.repaired.json"
            global_stats = report.get("global", {})
            print(
                f"[DONE] {wd} trusted={trusted} "
                f"total={global_stats.get('total', 0)} conv={global_stats.get('conv', 0)} "
                f"fail={global_stats.get('fail', 0)} clean={global_stats.get('clean', 0)} "
                f"filt={global_stats.get('filt', 0)} report={report_path}"
            )
            if trusted:
                success += 1
            else:
                untrusted += 1
                errors = report.get("consistency", {}).get("errors", [])
                failed_items.append({"workdir": str(wd), "reason": "; ".join(errors) or "untrusted_report"})
        except Exception as exc:
            failed += 1
            failed_items.append({"workdir": str(wd), "reason": str(exc)})
            print(f"[FAIL] {wd} reason={exc}")

    summary = {
        "total": total,
        "success": success,
        "untrusted": untrusted,
        "failed": failed,
        "failed_items": failed_items,
    }
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict and (failed > 0 or untrusted > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
