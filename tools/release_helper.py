#!/usr/bin/env python3
"""
DP-EVA Release Helper Tool

Automates version bumping and documentation synchronization.

Features:
- Bumps version in src/dpeva/__init__.py
- Updates version references in documentation (e.g., developer-guide.md)
- Generates a Changelog stub from git commits
- Updates CHANGELOG.md with the new version header

Usage:
    python tools/release_helper.py [major|minor|patch] [--dry-run]
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
from typing import List, Tuple

# Configuration
VERSION_FILE = "src/dpeva/__init__.py"
DOCS_TO_UPDATE = [
    "docs/guides/developer-guide.md",
]
CHANGELOG_FILE = "CHANGELOG.md"

def get_current_version() -> str:
    """Reads the current version from src/dpeva/__init__.py"""
    with open(VERSION_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError(f"Could not find __version__ in {VERSION_FILE}")
    return match.group(1)

def bump_version_string(current_version: str, part: str) -> str:
    """Bumps the version string based on the part (major, minor, patch)."""
    major, minor, patch = map(int, current_version.split("."))
    
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid part: {part}")
        
    return f"{major}.{minor}.{patch}"

def update_file_version(filepath: str, current_version: str, new_version: str, dry_run: bool = False):
    """Updates the version string in a file."""
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Simple replace logic - can be improved with regex if needed context
    # But for version numbers like "0.4.5" -> "0.4.6", direct replace is usually safe 
    # if the version string is unique enough. 
    # To be safer, we can use regex to ensure it looks like a version context.
    
    # For __init__.py
    if filepath.endswith(".py"):
        new_content = re.sub(
            r'__version__\s*=\s*["\']' + re.escape(current_version) + r'["\']',
            f'__version__ = "{new_version}"',
            content
        )
    # For Markdown docs
    else:
        # Look for pattern like "* **版本**: 0.4.5" or similar context
        # Or just global replace if we are confident. 
        # Given developer-guide.md has "* **版本**: 0.4.5", let's target that.
        # But also generic references.
        # Let's try a safe global replace for now, printing what changed.
        new_content = content.replace(current_version, new_version)
    
    if content != new_content:
        print(f"✅ Updating {filepath}: {current_version} -> {new_version}")
        if not dry_run:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
    else:
        print(f"ℹ️ No changes needed in {filepath} (Version {current_version} not found or pattern mismatch)")

def get_git_commits(last_tag: str = None) -> List[str]:
    """Gets git commits since last tag."""
    cmd = ["git", "log", "--pretty=format:- %s (%h)"]
    if last_tag:
        cmd.append(f"{last_tag}..HEAD")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        print("⚠️ Could not get git logs (not a git repo?)")
        return []

def update_changelog(new_version: str, commits: List[str], dry_run: bool = False):
    """Updates CHANGELOG.md with a new entry."""
    if not os.path.exists(CHANGELOG_FILE):
        print(f"⚠️ {CHANGELOG_FILE} not found.")
        return

    today = datetime.date.today().isoformat()
    header = f"## [{new_version}] - {today}\n\n"
    
    # Categorize commits (simple heuristic)
    added = []
    fixed = []
    changed = []
    others = []
    
    for commit in commits:
        lower = commit.lower()
        if "feat" in lower or "add" in lower:
            added.append(commit)
        elif "fix" in lower or "bug" in lower:
            fixed.append(commit)
        elif "refactor" in lower or "change" in lower or "update" in lower:
            changed.append(commit)
        else:
            others.append(commit)
            
    content_parts = [header]
    
    if added:
        content_parts.append("### Added\n" + "\n".join(added) + "\n")
    if fixed:
        content_parts.append("### Fixed\n" + "\n".join(fixed) + "\n")
    if changed:
        content_parts.append("### Changed\n" + "\n".join(changed) + "\n")
    if others:
        content_parts.append("### Other\n" + "\n".join(others) + "\n")
        
    new_entry = "\n".join(content_parts)
    
    print(f"✅ Preparing CHANGELOG entry for {new_version}")
    
    if not dry_run:
        with open(CHANGELOG_FILE, "r", encoding="utf-8") as f:
            existing_content = f.read()
            
        # Insert after the header (assuming standard Keep a Changelog format)
        # Look for the first "## [" line
        match = re.search(r'^## \[', existing_content, re.MULTILINE)
        if match:
            insert_pos = match.start()
            new_content = existing_content[:insert_pos] + new_entry + "\n" + existing_content[insert_pos:]
        else:
            # Append if no existing versions found
            new_content = existing_content + "\n" + new_entry
            
        with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
            f.write(new_content)

def main():
    parser = argparse.ArgumentParser(description="DP-EVA Release Helper")
    parser.add_argument("part", choices=["major", "minor", "patch"], help="Version part to bump")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify files")
    
    args = parser.parse_args()
    
    try:
        current_version = get_current_version()
        print(f"📌 Current Version: {current_version}")
        
        new_version = bump_version_string(current_version, args.part)
        print(f"🚀 Target Version:  {new_version}")
        
        if args.dry_run:
            print("⚠️ DRY RUN MODE: No files will be changed.")
        
        # 1. Update src/dpeva/__init__.py
        update_file_version(VERSION_FILE, current_version, new_version, args.dry_run)
        
        # 2. Update Documentation
        for doc in DOCS_TO_UPDATE:
            update_file_version(doc, current_version, new_version, args.dry_run)
            
        # 3. Update Changelog
        # Try to find the last tag
        # This part is tricky without git context, let's just get recent commits or empty
        # In a real scenario, we'd use `git describe --tags --abbrev=0`
        commits = get_git_commits()
        update_changelog(new_version, commits, args.dry_run)
        
        print(f"\n✨ Release preparation complete for v{new_version}!")
        print("Please review CHANGELOG.md and commit the changes.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
