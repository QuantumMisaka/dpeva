#!/usr/bin/env python3
import re
from pathlib import Path

def get_version():
    init_path = Path("src/dpeva/__init__.py")
    content = init_path.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Version not found in __init__.py")

def update_readme(version):
    readme_path = Path("README.md")
    content = readme_path.read_text()
    
    # Pattern for the version badge
    # Matches: [![Version](.../badge/version-X.Y.Z-green)](...)
    pattern = r'(\[!\[Version\]\(https://img\.shields\.io/badge/version-)([^/-]+)(-green\)\].*)'
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, fr'\g<1>{version}\g<3>', content)
        if new_content != content:
            readme_path.write_text(new_content)
            print(f"Updated README.md version to {version}")
        else:
            print(f"README.md already up to date ({version})")
    else:
        print("Version badge not found in README.md")

if __name__ == "__main__":
    try:
        ver = get_version()
        update_readme(ver)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
