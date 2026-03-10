#!/usr/bin/env python3
"""
DP-EVA Documentation Governance Auditor.
Checks for:
1. Directory Structure: Every active directory must have a README.md.
2. Front Matter: Every Markdown file must have valid YAML metadata.
3. Link Integrity: Internal relative links must be valid.
"""
import os
import re
import sys
import yaml
from pathlib import Path

DOCS_ROOT = Path("docs")
IGNORE_DIRS = {
    "_templates", "_static", "assets", "img", "build", "source", 
    ".github", "__pycache__", "logo_design"
}
REQUIRED_METADATA = {"status", "audience", "last-updated"}

def parse_front_matter(content):
    """Extract and parse YAML front matter from markdown content."""
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
    return None

def check_structure():
    """Check if all directories have README.md."""
    missing_readmes = []
    for root, dirs, files in os.walk(DOCS_ROOT):
        path = Path(root)
        # Skip ignored directories
        if any(part in IGNORE_DIRS for part in path.parts) or path.name.startswith("."):
            continue
            
        # Check for README.md (case insensitive)
        has_readme = any(f.lower() == "readme.md" for f in files)
        if not has_readme:
            missing_readmes.append(path)
            
    return missing_readmes

def check_front_matter():
    """Check if markdown files have valid front matter."""
    invalid_files = []
    for root, dirs, files in os.walk(DOCS_ROOT):
        path = Path(root)
        if any(part in IGNORE_DIRS for part in path.parts) or path.name.startswith("."):
            continue
        if "archive" in path.parts:
            continue
        if "archive" in path.parts:
            continue
            
        for file in files:
            if file.endswith(".md"):
                file_path = path / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    meta = parse_front_matter(content)
                    if not meta:
                        invalid_files.append((file_path, "Missing or invalid YAML front matter"))
                        continue
                        
                    missing_keys = REQUIRED_METADATA - set(meta.keys())
                    if missing_keys:
                        invalid_files.append((file_path, f"Missing keys: {missing_keys}"))
                        
                except Exception as e:
                    invalid_files.append((file_path, f"Error reading file: {e}"))
                    
    return invalid_files

def check_links():
    """Check for broken internal relative links."""
    broken_links = []
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    
    for root, dirs, files in os.walk(DOCS_ROOT):
        path = Path(root)
        if any(part in IGNORE_DIRS for part in path.parts) or path.name.startswith("."):
            continue
        if "archive" in path.parts:
            continue
            
        for file in files:
            if file.endswith(".md"):
                file_path = path / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    for match in re.finditer(link_pattern, content):
                        link_text = match.group(1)
                        link_target = match.group(2)
                        
                        # Skip external links, anchors only, and mailto
                        if link_target.startswith(("http", "https", "mailto:", "#")):
                            continue
                            
                        # Remove anchor from target
                        target_path_str = link_target.split("#")[0]
                        if not target_path_str:
                            continue
                            
                        # Resolve path
                        if target_path_str.startswith("/"):
                            target_full_path = (Path.cwd() / target_path_str.lstrip("/")).resolve()
                        else:
                            target_full_path = (file_path.parent / target_path_str).resolve()
                            
                        if not target_full_path.exists():
                            broken_links.append((file_path, link_target, "File not found"))
                            
                except Exception as e:
                    broken_links.append((file_path, "Error", str(e)))
                    
    return broken_links

def check_forbidden_links():
    forbidden_links = []
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    for root, dirs, files in os.walk(DOCS_ROOT):
        path = Path(root)
        if any(part in IGNORE_DIRS for part in path.parts) or path.name.startswith("."):
            continue
        for file in files:
            if file.endswith(".md"):
                file_path = path / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    for match in re.finditer(link_pattern, content):
                        link_target = match.group(2).strip()
                        if link_target.startswith(("/home/", "\\home\\")):
                            forbidden_links.append((file_path, link_target, "Filesystem absolute path is forbidden"))
                        if re.match(r"^[A-Za-z]:[\\/]", link_target):
                            forbidden_links.append((file_path, link_target, "Windows absolute path is forbidden"))
                except Exception as e:
                    forbidden_links.append((file_path, "Error", str(e)))
    return forbidden_links

def check_owner_metadata():
    missing_owner = []
    for root, dirs, files in os.walk(DOCS_ROOT):
        path = Path(root)
        if any(part in IGNORE_DIRS for part in path.parts) or path.name.startswith("."):
            continue
        for file in files:
            if file.endswith(".md"):
                file_path = path / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    meta = parse_front_matter(content)
                    if not meta:
                        continue
                    if str(meta.get("status", "")).strip().lower() == "active":
                        if "owner" not in meta and "owners" not in meta:
                            missing_owner.append(file_path)
                except Exception:
                    continue
    return missing_owner

def main():
    print("🔍 Starting Documentation Governance Audit...\n")
    
    # 1. Structure Check
    print("📂 Checking Directory Structure...")
    missing_readmes = check_structure()
    if missing_readmes:
        print("❌ Missing README.md in:")
        for p in missing_readmes:
            print(f"  - {p}")
    else:
        print("✅ Structure OK")
        
    # 2. Front Matter Check
    print("\n📝 Checking Front Matter...")
    invalid_meta = check_front_matter()
    if invalid_meta:
        print("❌ Invalid Metadata in:")
        for f, reason in invalid_meta:
            print(f"  - {f}: {reason}")
    else:
        print("✅ Metadata OK")

    # 3. Link Check
    print("\n🔗 Checking Internal Links...")
    broken_links = check_links()
    if broken_links:
        print("❌ Broken Links Found:")
        for f, target, reason in broken_links:
            print(f"  - {f} -> {target} ({reason})")
    else:
        print("✅ Links OK")

    print("\n⛔ Checking Forbidden Absolute Paths...")
    forbidden_links = check_forbidden_links()
    if forbidden_links:
        print("❌ Forbidden Links Found:")
        for f, target, reason in forbidden_links:
            print(f"  - {f} -> {target} ({reason})")
    else:
        print("✅ Forbidden Path Check OK")

    print("\n👤 Checking Owner Coverage (non-blocking)...")
    missing_owner = check_owner_metadata()
    if missing_owner:
        print("⚠️ Missing owner/owners in active docs:")
        for f in missing_owner:
            print(f"  - {f}")
    else:
        print("✅ Owner Coverage OK")
        
    # Summary
    if missing_readmes or invalid_meta or broken_links or forbidden_links:
        print("\n🚫 Audit FAILED. Please fix the issues above.")
        sys.exit(1)
    else:
        print("\n✨ Audit PASSED. Documentation system is healthy.")
        sys.exit(0)

if __name__ == "__main__":
    main()
