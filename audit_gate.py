#!/usr/bin/env python3
"""
DP-EVA Automated Quality Gate
Enforces Maintainability and Purity rules.
Usage: python scripts/audit_gate.py
Exit Code: 0 (Pass), 1 (Fail)
"""
import ast
import os
import re
import sys
from typing import List, Tuple

SRC_DIR = "src/dpeva"
EXCLUDE_DIRS = ["__pycache__", "tests", "docs"]

# Rules Configuration
PATH_REGEX = re.compile(r'(/|\\\\)[a-zA-Z0-9_]+')
MAGIC_EXCLUSIONS = [0, 1, -1, 0.0, 1.0, -1.0]

class AuditViolation:
    def __init__(self, file: str, line: int, rule: str, detail: str):
        self.file = file
        self.line = line
        self.rule = rule
        self.detail = detail

    def __str__(self):
        return f"[{self.rule}] {self.file}:{self.line} - {self.detail}"

class GateKeeper(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[AuditViolation] = []

    def visit_Constant(self, node):
        # Rule 1.1: Hardcoded Paths
        # Ignore if in constants.py
        if "constants.py" in self.filename:
            return

        if isinstance(node.value, str):
            if PATH_REGEX.search(node.value) and len(node.value) > 2 and " " not in node.value:
                if not any(x in node.value for x in ["http", "urn", "<", ">", "dpeva"]):
                     # Heuristic: Ignore strings that look like keys or formats
                    if not (node.value.startswith("/") or node.value.startswith("./")):
                        return
                    
                    self.violations.append(AuditViolation(self.filename, node.lineno, "NO_HARDCODED_PATH", f"Found path-like string: '{node.value}'"))

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Rule 2.5: Missing Docstrings (Public API)
        if not ast.get_docstring(node) and not node.name.startswith("_"):
             self.violations.append(AuditViolation(self.filename, node.lineno, "MISSING_DOCSTRING", f"Function '{node.name}' has no docstring"))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # Rule 2.5: Missing Docstrings (Public API)
        if not ast.get_docstring(node) and not node.name.startswith("_"):
             self.violations.append(AuditViolation(self.filename, node.lineno, "MISSING_DOCSTRING", f"Class '{node.name}' has no docstring"))
        self.generic_visit(node)

def run_audit(directory: str) -> bool:
    all_violations = []
    print(f"Starting DP-EVA Quality Audit on {directory}...")
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())
                    visitor = GateKeeper(filepath)
                    visitor.visit(tree)
                    all_violations.extend(visitor.violations)
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

    if all_violations:
        print(f"\n❌ Audit FAILED with {len(all_violations)} violations:")
        for v in all_violations:
            print(str(v))
        return False
    else:
        print("\n✅ Audit PASSED. No violations found.")
        return True

if __name__ == "__main__":
    success = run_audit(SRC_DIR)
    sys.exit(0 if success else 1)
