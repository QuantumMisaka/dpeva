#!/usr/bin/env python3
"""
Release Helper Script for DP-EVA
Usage: python scripts/release_helper.py [patch|minor|major|version_string]
"""
import sys
import re
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INIT_FILE = PROJECT_ROOT / "src" / "dpeva" / "__init__.py"
README_FILE = PROJECT_ROOT / "README.md"
DEV_GUIDE = PROJECT_ROOT / "docs" / "source" / "guides" / "developer-guide.md"

def get_current_version():
    with open(INIT_FILE, "r") as f:
        content = f.read()
        match = re.search(r'__version__ = "(.*?)"', content)
        if match:
            return match.group(1)
    raise ValueError("Could not find __version__ in src/dpeva/__init__.py")

def bump_version(current_ver, bump_type):
    major, minor, patch = map(int, current_ver.split("."))
    if bump_type == "patch":
        patch += 1
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif re.match(r"^\d+\.\d+\.\d+$", bump_type):
        return bump_type
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    return f"{major}.{minor}.{patch}"

def update_files(new_ver):
    # Update __init__.py
    with open(INIT_FILE, "r") as f:
        content = f.read()
    new_content = re.sub(r'__version__ = ".*?"', f'__version__ = "{new_ver}"', content)
    with open(INIT_FILE, "w") as f:
        f.write(new_content)
    print(f"Updated {INIT_FILE}")

    # Update README.md badge
    if README_FILE.exists():
        with open(README_FILE, "r") as f:
            content = f.read()
        # Update badge like: ![Version](https://img.shields.io/badge/version-0.6.0-blue)
        new_content = re.sub(r"badge/version-[\d\.]+-", f"badge/version-{new_ver}-", content)
        with open(README_FILE, "w") as f:
            f.write(new_content)
        print(f"Updated {README_FILE}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python release_helper.py [patch|minor|major|X.Y.Z]")
        sys.exit(1)

    bump_type = sys.argv[1]
    current_ver = get_current_version()
    new_ver = bump_version(current_ver, bump_type)
    
    print(f"Bumping version: {current_ver} -> {new_ver}")
    update_files(new_ver)
    
    print("\nNext Steps:")
    print(f"1. Update {DEV_GUIDE} with release notes.")
    print(f"2. git commit -am 'chore: release v{new_ver}'")
    print(f"3. git tag v{new_ver}")

if __name__ == "__main__":
    main()
