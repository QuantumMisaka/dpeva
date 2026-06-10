#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = PROJECT_ROOT / "docs"
IGNORE_DIRS = {
    "_templates",
    "_static",
    "assets",
    "img",
    "build",
    "source",
    ".github",
    "__pycache__",
    "logo_design",
    "archive",
}
REQUIRED_METADATA = {"status", "audience", "last-updated"}


def parse_front_matter(content):
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def should_skip(path):
    return any(part in IGNORE_DIRS for part in path.parts) or path.name.startswith(".")


def iter_doc_dirs(docs_root):
    for root, dirs, files in os.walk(docs_root):
        path = Path(root)
        if should_skip(path):
            dirs[:] = []
            continue
        yield path, files


def iter_markdown_files(docs_root):
    for path, files in iter_doc_dirs(docs_root):
        for file in files:
            if file.endswith(".md"):
                yield path / file


def read_text(file_path):
    return file_path.read_text(encoding="utf-8")


def resolve_link_target(file_path, link_target):
    target_path_str = link_target.split("#")[0]
    if not target_path_str:
        return None
    if target_path_str.startswith("/"):
        return (PROJECT_ROOT / target_path_str.lstrip("/")).resolve()
    return (file_path.parent / target_path_str).resolve()


def check_structure(docs_root):
    missing_readmes = []
    for path, files in iter_doc_dirs(docs_root):
        if not any(f.lower() == "readme.md" for f in files):
            missing_readmes.append(path)
    return missing_readmes


def check_front_matter(docs_root):
    invalid_files = []
    for file_path in iter_markdown_files(docs_root):
        try:
            meta = parse_front_matter(read_text(file_path))
            if not meta:
                invalid_files.append((file_path, "Missing or invalid YAML front matter"))
                continue
            missing_keys = REQUIRED_METADATA - set(meta.keys())
            if missing_keys:
                invalid_files.append((file_path, f"Missing keys: {missing_keys}"))
        except Exception as exc:
            invalid_files.append((file_path, f"Error reading file: {exc}"))
    return invalid_files


def check_links(docs_root):
    broken_links = []
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    for file_path in iter_markdown_files(docs_root):
        try:
            content = read_text(file_path)
            for match in re.finditer(link_pattern, content):
                link_target = match.group(2).strip()
                if link_target.startswith(("http", "https", "mailto:", "#")):
                    continue
                target_full_path = resolve_link_target(file_path, link_target)
                if target_full_path is not None and not target_full_path.exists():
                    broken_links.append((file_path, link_target, "File not found"))
        except Exception as exc:
            broken_links.append((file_path, "Error", str(exc)))
    return broken_links


def check_forbidden_links(docs_root):
    forbidden_links = []
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    for file_path in iter_markdown_files(docs_root):
        try:
            content = read_text(file_path)
            for match in re.finditer(link_pattern, content):
                link_target = match.group(2).strip()
                if link_target.startswith(("/home/", "\\home\\")):
                    forbidden_links.append((file_path, link_target, "Filesystem absolute path is forbidden"))
                if re.match(r"^[A-Za-z]:[\\/]", link_target):
                    forbidden_links.append((file_path, link_target, "Windows absolute path is forbidden"))
        except Exception as exc:
            forbidden_links.append((file_path, "Error", str(exc)))
    return forbidden_links


def matches_owner_scope(file_path, docs_root, owner_dirs):
    if not owner_dirs:
        return True
    rel_parts = file_path.relative_to(docs_root).parts
    return any(part in owner_dirs for part in rel_parts)


def check_owner_metadata(docs_root, owner_dirs=None):
    missing_owner = []
    for file_path in iter_markdown_files(docs_root):
        try:
            meta = parse_front_matter(read_text(file_path))
            if not meta:
                continue
            if not matches_owner_scope(file_path, docs_root, owner_dirs or []):
                continue
            if str(meta.get("status", "")).strip().lower() == "active":
                if "owner" not in meta and "owners" not in meta:
                    missing_owner.append(file_path)
        except Exception:
            continue
    return missing_owner


def parse_args():
    parser = argparse.ArgumentParser(description="Run documentation governance checks.")
    parser.add_argument("--strict-owner", action="store_true", help="Fail when active docs miss owner metadata.")
    parser.add_argument(
        "--strict-owner-dir",
        action="append",
        default=[],
        help="Directory name to enforce owner coverage on. Can be provided multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("🔍 Starting Documentation Governance Audit...\n")

    print("📂 Checking Directory Structure...")
    missing_readmes = check_structure(DOCS_ROOT)
    if missing_readmes:
        print("❌ Missing README.md in:")
        for path in missing_readmes:
            print(f"  - {path}")
    else:
        print("✅ Structure OK")

    print("\n📝 Checking Front Matter...")
    invalid_meta = check_front_matter(DOCS_ROOT)
    if invalid_meta:
        print("❌ Invalid Metadata in:")
        for file_path, reason in invalid_meta:
            print(f"  - {file_path}: {reason}")
    else:
        print("✅ Metadata OK")

    print("\n🔗 Checking Internal Links...")
    broken_links = check_links(DOCS_ROOT)
    if broken_links:
        print("❌ Broken Links Found:")
        for file_path, target, reason in broken_links:
            print(f"  - {file_path} -> {target} ({reason})")
    else:
        print("✅ Links OK")

    print("\n⛔ Checking Forbidden Absolute Paths...")
    forbidden_links = check_forbidden_links(DOCS_ROOT)
    if forbidden_links:
        print("❌ Forbidden Links Found:")
        for file_path, target, reason in forbidden_links:
            print(f"  - {file_path} -> {target} ({reason})")
    else:
        print("✅ Forbidden Path Check OK")

    print(f"\n👤 Checking Owner Coverage ({'blocking' if args.strict_owner else 'non-blocking'})...")
    missing_owner = check_owner_metadata(DOCS_ROOT, args.strict_owner_dir)
    if missing_owner:
        prefix = "❌" if args.strict_owner else "⚠️"
        print(f"{prefix} Missing owner/owners in active docs:")
        for file_path in missing_owner:
            print(f"  - {file_path}")
    else:
        print("✅ Owner Coverage OK")

    failed = bool(missing_readmes or invalid_meta or broken_links or forbidden_links)
    if args.strict_owner and missing_owner:
        failed = True

    if failed:
        print("\n🚫 Audit FAILED. Please fix the issues above.")
        sys.exit(1)

    print("\n✨ Audit PASSED. Documentation system is healthy.")
    sys.exit(0)


if __name__ == "__main__":
    main()
