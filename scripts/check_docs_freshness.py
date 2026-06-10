#!/usr/bin/env python3
import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXCLUDE_DIRS = ["archive", "_templates"]

def get_git_last_commit_date(file_path):
    try:
        result = subprocess.check_output(
            ["git", "log", "-1", "--format=%cd", "--date=iso-strict", str(file_path)],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        if not result:
            return None
        return datetime.fromisoformat(result)
    except subprocess.CalledProcessError:
        return None
    except ValueError:
        return None


def check_freshness(root_dir, days_threshold=30, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS

    root_path = Path(root_dir)
    now = datetime.now().astimezone()
    threshold = timedelta(days=days_threshold)
    stale_files = []

    print(f"Checking documentation freshness (Threshold: {days_threshold} days)...")

    for file_path in root_path.rglob("*.md"):
        rel_path = file_path.relative_to(root_path)
        if any(part in exclude_dirs for part in rel_path.parts):
            continue

        last_modified = get_git_last_commit_date(file_path)
        if last_modified:
            age = now - last_modified
            if age > threshold:
                stale_files.append((str(rel_path), age.days))

    return stale_files


def main():
    parser = argparse.ArgumentParser(description="Check documentation freshness.")
    parser.add_argument("--dir", type=str, default=str(PROJECT_ROOT / "docs"), help="Root directory to check")
    parser.add_argument("--days", type=int, default=90, help="Max age in days (default: 90)")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude. Can be provided multiple times.",
    )
    args = parser.parse_args()

    stale_files = check_freshness(args.dir, args.days, DEFAULT_EXCLUDE_DIRS + args.exclude_dir)

    if stale_files:
        print(f"\nFound {len(stale_files)} stale documents (> {args.days} days):")
        stale_files.sort(key=lambda x: x[1], reverse=True)
        for f, age in stale_files:
            print(f"  - {f}: {age} days old")
        sys.exit(1)
    else:
        print("\nAll core documents are fresh!")
        sys.exit(0)


if __name__ == "__main__":
    main()
