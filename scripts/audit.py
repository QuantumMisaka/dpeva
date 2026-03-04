#!/usr/bin/env python3
"""
DP-EVA Automated Code Auditor (Combined Tool)
Integrates Quality Gate (Blocking) and Code Smell Analysis (Reporting).

Features:
- Detects hardcoded absolute paths (Critical for HPC portability)
- Checks for missing docstrings in Public API
- Scans for Magic Numbers
- Identifies commented-out code blocks

Usage:
    python tools/audit.py [directory] [--strict]
    
    --strict: Exit with error if any violation found (CI/CD mode)
"""
import ast
import os
import re
import sys
import argparse
from typing import List, Tuple, Dict, Any

# Configuration
DEFAULT_SRC_DIR = "src/dpeva"
EXCLUDE_DIRS = ["__pycache__", "tests", "docs", "build", "dist", ".git", ".idea", ".vscode"]
PATH_REGEX = re.compile(r'(/|\\\\)[a-zA-Z0-9_]+')
MAGIC_EXCLUSIONS = [0, 1, -1, 0.0, 1.0, -1.0]

class AuditViolation:
    def __init__(self, file: str, line: int, rule: str, detail: str, severity: str = "WARNING"):
        self.file = file
        self.line = line
        self.rule = rule
        self.detail = detail
        self.severity = severity

    def __str__(self):
        return f"[{self.severity}] [{self.rule}] {self.file}:{self.line} - {self.detail}"

class CombinedAuditVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, lines: List[str]):
        self.filename = filename
        self.lines = lines
        self.violations: List[AuditViolation] = []
        self.in_class = False

    def visit_Constant(self, node):
        # 1. Hardcoded Paths Check
        if isinstance(node.value, str):
            # Ignore if in specific config/constant files
            if "constants.py" in self.filename or "config.py" in self.filename:
                pass
            elif PATH_REGEX.search(node.value) and len(node.value) > 2 and " " not in node.value:
                # Heuristics to reduce false positives
                if not any(x in node.value for x in ["http", "urn", "<", ">", "dpeva", "%", "{"]):
                    if node.value.startswith("/") or node.value.startswith("./"):
                         self.violations.append(AuditViolation(
                             self.filename, node.lineno, "HARDCODED_PATH", 
                             f"Found path-like string: '{node.value}'", "ERROR"
                         ))

        # 2. Magic Numbers Check
        elif isinstance(node.value, (int, float)):
            if node.value not in MAGIC_EXCLUSIONS and not self.filename.endswith("_test.py"):
                 self.violations.append(AuditViolation(
                     self.filename, node.lineno, "MAGIC_NUMBER", 
                     f"Found unnamed numeric literal: {node.value}", "INFO"
                 ))

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # 3. Missing Docstrings (Public API)
        if not ast.get_docstring(node) and not node.name.startswith("_"):
             self.violations.append(AuditViolation(
                 self.filename, node.lineno, "MISSING_DOCSTRING", 
                 f"Function '{node.name}' has no docstring", "WARNING"
             ))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # 3. Missing Docstrings (Public API)
        if not ast.get_docstring(node) and not node.name.startswith("_"):
             self.violations.append(AuditViolation(
                 self.filename, node.lineno, "MISSING_DOCSTRING", 
                 f"Class '{node.name}' has no docstring", "WARNING"
             ))
        self.generic_visit(node)

def check_commented_code(filename: str, lines: List[str]) -> List[AuditViolation]:
    violations = []
    code_keywords = ["def ", "class ", "import ", "from ", "return ", "if ", "for ", "while "]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            content = stripped[1:].strip()
            # Ignore linter directives
            if "noqa" in content or "pylint" in content:
                continue
            
            for kw in code_keywords:
                if content.startswith(kw):
                    violations.append(AuditViolation(
                        filename, i + 1, "COMMENTED_CODE", 
                        f"Potential commented out code: '{stripped}'", "INFO"
                    ))
                    break
    return violations

def run_audit(directory: str, strict: bool = False) -> bool:
    all_violations = []
    stats = {"files": 0, "lines": 0}
    
    print(f"Starting DP-EVA Code Audit on {directory}...")
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                stats["files"] += 1
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        source = f.read()
                        lines = source.splitlines()
                        stats["lines"] += len(lines)
                        
                    tree = ast.parse(source)
                    visitor = CombinedAuditVisitor(filepath, lines)
                    visitor.visit(tree)
                    
                    # Combine AST violations with regex-based violations
                    all_violations.extend(visitor.violations)
                    all_violations.extend(check_commented_code(filepath, lines))
                    
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

    # Sort violations by severity then file
    severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    all_violations.sort(key=lambda x: (severity_order.get(x.severity, 3), x.file, x.line))

    # Print Report
    print(f"\n=== Audit Summary ===")
    print(f"Scanned: {stats['files']} files, {stats['lines']} lines")
    print(f"Total Violations: {len(all_violations)}")
    
    if all_violations:
        print("\n=== Details ===")
        current_file = ""
        for v in all_violations:
            if v.file != current_file:
                print(f"\n📄 {v.file}")
                current_file = v.file
            
            icon = "🔴" if v.severity == "ERROR" else "🟡" if v.severity == "WARNING" else "🔵"
            print(f"  {icon} Line {v.line}: [{v.rule}] {v.detail}")

    # Determine Exit Code
    # In strict mode, fail on ERROR or WARNING. In normal mode, fail only on ERROR.
    errors = [v for v in all_violations if v.severity == "ERROR"]
    warnings = [v for v in all_violations if v.severity == "WARNING"]
    
    if strict:
        if errors or warnings:
            print(f"\n❌ Audit FAILED (Strict Mode): Found {len(errors)} errors and {len(warnings)} warnings.")
            return False
    else:
        if errors:
            print(f"\n❌ Audit FAILED: Found {len(errors)} critical errors.")
            return False
            
    print("\n✅ Audit PASSED.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP-EVA Code Auditor")
    parser.add_argument("directory", nargs="?", default=DEFAULT_SRC_DIR, help="Directory to scan")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings as well as errors")
    
    args = parser.parse_args()
    
    success = run_audit(args.directory, args.strict)
    sys.exit(0 if success else 1)
