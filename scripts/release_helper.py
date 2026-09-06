#!/usr/bin/env python3
"""Synchronize and check DP-EVA's maintained release-version surfaces."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).parent.parent
INIT_FILE = PROJECT_ROOT / "src" / "dpeva" / "__init__.py"
README_FILE = PROJECT_ROOT / "README.md"
DEV_GUIDE = PROJECT_ROOT / "docs" / "guides" / "developer-guide.md"

_VERSION = r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
_VERSION_RE = re.compile(rf"^{_VERSION}$")
_INIT_RE = re.compile(r'^__version__ = "(?P<version>[^"]+)"$', re.MULTILINE)
_README_RE = re.compile(r"badge/version-(?P<version>[^-]+)-")
_DEV_GUIDE_RE = re.compile(r"^\* \*\*版本\*\*: (?P<version>\S+)$", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _declared_version(path: Path, pattern: re.Pattern[str]) -> str:
    matches = list(pattern.finditer(_read(path)))
    if len(matches) != 1:
        raise ValueError(f"Could not find exactly one maintained version in {path}")
    return matches[0].group("version")


def _validate_version(value: str) -> str:
    if _VERSION_RE.fullmatch(value) is None:
        raise ValueError(f"Invalid version or bump type: {value}")
    return value


def get_current_version() -> str:
    """Return the validated package version."""

    return _validate_version(_declared_version(INIT_FILE, _INIT_RE))


def bump_version(current_ver: str, bump_type: str) -> str:
    """Return a major/minor/patch bump or a validated explicit version."""

    major, minor, patch = map(int, _validate_version(current_ver).split("."))
    if bump_type == "patch":
        patch += 1
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        return _validate_version(bump_type)
    return f"{major}.{minor}.{patch}"


def _replace_version(
    content: str,
    pattern: re.Pattern[str],
    replacement: str,
    path: Path,
) -> str:
    updated, count = pattern.subn(replacement, content)
    if count != 1:
        raise ValueError(f"Could not find exactly one maintained version in {path}")
    return updated


def update_files(new_ver: str) -> None:
    """Update all maintained version surfaces after validating every edit."""

    version = _validate_version(new_ver)
    updates = {
        INIT_FILE: _replace_version(
            _read(INIT_FILE), _INIT_RE, f'__version__ = "{version}"', INIT_FILE
        ),
        README_FILE: _replace_version(
            _read(README_FILE),
            _README_RE,
            f"badge/version-{version}-",
            README_FILE,
        ),
        DEV_GUIDE: _replace_version(
            _read(DEV_GUIDE),
            _DEV_GUIDE_RE,
            f"* **版本**: {version}",
            DEV_GUIDE,
        ),
    }
    for path, content in updates.items():
        path.write_text(content, encoding="utf-8")
        print(f"Updated {path.relative_to(PROJECT_ROOT)}")


def check_versions() -> list[str]:
    """Return mismatches between the package and maintained documentation."""

    expected = get_current_version()

    declarations = (
        (README_FILE, _README_RE),
        (DEV_GUIDE, _DEV_GUIDE_RE),
    )
    errors: list[str] = []
    for path, pattern in declarations:
        actual = _declared_version(path, pattern)
        if actual != expected:
            errors.append(
                f"{path.relative_to(PROJECT_ROOT).as_posix()} declares {actual}, "
                f"expected {expected}"
            )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", help="patch, minor, major, or X.Y.Z")
    parser.add_argument(
        "--check",
        action="store_true",
        help="check maintained version surfaces without writing files",
    )
    args = parser.parse_args(argv)

    if args.check:
        if args.target is not None:
            parser.error("--check does not accept a version target")
        errors = check_versions()
        if errors:
            for error in errors:
                print(error)
            return 1
        print(f"Version surfaces are synchronized at {get_current_version()}")
        return 0

    if args.target is None:
        parser.error("a bump type/version or --check is required")
    current_ver = get_current_version()
    new_ver = bump_version(current_ver, args.target)
    print(f"Bumping version: {current_ver} -> {new_ver}")
    update_files(new_ver)
    print("\nNext Steps:")
    print(f"1. Add release notes for {new_ver} to {DEV_GUIDE.relative_to(PROJECT_ROOT)}")
    print("2. Run the release profile and build/archive smoke checks")
    print("3. Commit; tagging and publishing remain separate authorized actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
