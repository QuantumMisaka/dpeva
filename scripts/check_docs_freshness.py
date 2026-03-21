#!/usr/bin/env python3
"""
Documentation Freshness Checker
===============================

Checks if core documentation files have been updated within a specified threshold.
"""

import sys
import argparse
from datetime import datetime, timedelta
import subprocess
from pathlib import Path

def get_git_last_commit_date(file_path):
    """Get the last commit date of a file using git."""
    try:
        # Returns timestamp in ISO 8601 format
        result = subprocess.check_output(
            ['git', 'log', '-1', '--format=%cd', '--date=iso-strict', str(file_path)],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        if not result:
            return None
        # Parse ISO format (e.g., 2023-10-27T10:00:00+08:00)
        return datetime.fromisoformat(result)
    except subprocess.CalledProcessError:
        return None
    except ValueError:
        # Fallback for older python versions if fromisoformat is limited
        return None

def check_freshness(root_dir, days_threshold=30, exclude_dirs=None):
    """Check freshness of markdown files."""
    if exclude_dirs is None:
        exclude_dirs = ['archive', '_templates', 'plans', 'reports']
    
    root_path = Path(root_dir)
    now = datetime.now().astimezone()
    threshold = timedelta(days=days_threshold)
    
    stale_files = []
    
    print(f"Checking documentation freshness (Threshold: {days_threshold} days)...")
    
    for file_path in root_path.rglob('*.md'):
        # Check exclusions
        rel_path = file_path.relative_to(root_path)
        if any(part in exclude_dirs for part in rel_path.parts):
            continue
            
        last_modified = get_git_last_commit_date(file_path)
        
        if last_modified:
            age = now - last_modified
            if age > threshold:
                stale_files.append((str(rel_path), age.days))
        else:
            # File not in git or error
            # print(f"Warning: Could not get git history for {rel_path}")
            pass

    return stale_files

def main():
    parser = argparse.ArgumentParser(description="Check documentation freshness.")
    parser.add_argument('--dir', type=str, default='docs', help='Root directory to check')
    parser.add_argument('--days', type=int, default=90, help='Max age in days (default: 90)')
    args = parser.parse_args()

    stale_files = check_freshness(args.dir, args.days)
    
    if stale_files:
        print(f"\nFound {len(stale_files)} stale documents (> {args.days} days):")
        # Sort by age descending
        stale_files.sort(key=lambda x: x[1], reverse=True)
        for f, age in stale_files:
            print(f"  - {f}: {age} days old")
        sys.exit(1)
    else:
        print("\nAll core documents are fresh!")
        sys.exit(0)

if __name__ == "__main__":
    main()
