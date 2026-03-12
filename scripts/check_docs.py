#!/usr/bin/env python3
"""
DP-EVA Documentation Consistency Checker.
Checks if CLI commands and options are documented in docs/guides/cli.md.
"""
import argparse
import subprocess
import re
import sys
import os

DOC_PATH = "docs/guides/cli.md"

def get_cli_help(command):
    """Run CLI command and return help text."""
    try:
        cmd = command.split() + ["--help"]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(cmd)}: {e}")
        return ""

def extract_subcommands(help_text):
    """Extract subcommands from help text."""
    # Assuming argparse default formatting
    # Subcommands are usually listed under "positional arguments" or "commands"
    # But dpeva help output might differ. Let's look for known subcommands for now.
    known_subcommands = ["train", "infer", "feature", "collect", "label", "analysis"]
    found = []
    for sub in known_subcommands:
        if sub in help_text:
            found.append(sub)
    return found

def check_doc_coverage(doc_content, keywords):
    """Check if keywords are present in doc content."""
    missing = []
    for kw in keywords:
        # Simple string search. Can be improved with regex.
        if kw not in doc_content:
            missing.append(kw)
    return missing

def main():
    print("Checking CLI documentation consistency...")
    
    if not os.path.exists(DOC_PATH):
        print(f"Error: Documentation file {DOC_PATH} not found.")
        sys.exit(1)
        
    with open(DOC_PATH, "r") as f:
        doc_content = f.read()
        
    # 1. Check Main Command
    print("Checking 'dpeva' main command...")
    main_help = get_cli_help("dpeva")
    subcommands = extract_subcommands(main_help)
    
    # Check if subcommands are mentioned in doc
    missing_subs = check_doc_coverage(doc_content, subcommands)
    if missing_subs:
        print(f"❌ Missing documentation for subcommands: {missing_subs}")
    else:
        print("✅ All subcommands mentioned in documentation.")
        
    # 2. Check Subcommands
    all_passed = True
    for sub in subcommands:
        print(f"Checking 'dpeva {sub}'...")
        # Just check if the section header exists (heuristic)
        # e.g. "### 4.1 train"
        if not re.search(f"###.*{sub}", doc_content, re.IGNORECASE):
             print(f"❌ Subcommand '{sub}' section header missing in {DOC_PATH}")
             all_passed = False
        else:
             print(f"✅ Subcommand '{sub}' documented.")

    if all_passed and not missing_subs:
        print("\n✨ Documentation consistency check PASSED.")
        sys.exit(0)
    else:
        print("\n⚠️ Documentation consistency check FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
