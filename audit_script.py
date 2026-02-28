import ast
import os
import re
import sys

# Configuration
SRC_DIR = "src/dpeva"
EXCLUDE_DIRS = ["__pycache__", "tests", "docs"]
PATH_REGEX = re.compile(r'(/|\\)[a-zA-Z0-9_]+')  # Simple path detection
MAGIC_STRING_REGEX = re.compile(r'^[a-zA-Z0-9_]+$') # Simple string detection

# Report Data
report = {
    "hardcoded_paths": [],
    "magic_values": [],
    "missing_docstrings": [],
    "commented_code": [],
    "total_files": 0,
    "total_lines": 0
}

class AuditVisitor(ast.NodeVisitor):
    def __init__(self, filename, source_lines):
        self.filename = filename
        self.source_lines = source_lines

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            # Check for paths
            if PATH_REGEX.search(node.value) and len(node.value) > 2 and " " not in node.value:
                # Exclude common false positives (e.g., mime types, simple slashes)
                if not node.value.startswith("http") and not node.value.startswith("urn"):
                     report["hardcoded_paths"].append((self.filename, node.lineno, node.value))
            
            # Check for magic strings (heuristic: length > 3, not in upper case constant)
            # This is hard to get right without context, so we'll be lenient
            pass

        elif isinstance(node.value, (int, float)):
            # Check for magic numbers (exclude 0, 1, -1)
            if node.value not in [0, 1, -1, 0.0, 1.0, -1.0]:
                report["magic_values"].append((self.filename, node.lineno, node.value))

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not ast.get_docstring(node) and not node.name.startswith("_"):
             report["missing_docstrings"].append((self.filename, node.lineno, f"Function: {node.name}"))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if not ast.get_docstring(node) and not node.name.startswith("_"):
             report["missing_docstrings"].append((self.filename, node.lineno, f"Class: {node.name}"))
        self.generic_visit(node)

    def visit_Module(self, node):
        if not ast.get_docstring(node):
             report["missing_docstrings"].append((self.filename, 1, "Module"))
        self.generic_visit(node)

def check_commented_code(filename, lines):
    # Heuristic for commented code
    code_keywords = ["def ", "class ", "import ", "from ", "return ", "if ", "for ", "while "]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            content = stripped[1:].strip()
            for kw in code_keywords:
                if content.startswith(kw) and not content.endswith("  # noqa"):
                    report["commented_code"].append((filename, i + 1, stripped))
                    break

def audit_directory(directory):
    for root, dirs, files in os.walk(directory):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                report["total_files"] += 1
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        source = f.read()
                        lines = source.splitlines()
                        report["total_lines"] += len(lines)
                        
                    tree = ast.parse(source)
                    visitor = AuditVisitor(filepath, lines)
                    visitor.visit(tree)
                    
                    check_commented_code(filepath, lines)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    audit_directory(SRC_DIR)
    
    # Output Report
    print("=== Audit Report ===")
    print(f"Total Files Scanned: {report['total_files']}")
    print(f"Total Lines Scanned: {report['total_lines']}")
    
    print("\n[1. Maintainability - Hardcoded Paths]")
    for item in report["hardcoded_paths"]:
        print(f"{item[0]}:{item[1]} - {item[2]}")
        
    print("\n[1. Maintainability - Magic Numbers]")
    # Limit output for magic numbers as there might be many false positives
    for i, item in enumerate(report["magic_values"]):
        if i < 20:
             print(f"{item[0]}:{item[1]} - {item[2]}")
    if len(report["magic_values"]) > 20:
        print(f"... and {len(report['magic_values']) - 20} more.")

    print("\n[2. Purity - Missing Docstrings]")
    for item in report["missing_docstrings"]:
        print(f"{item[0]}:{item[1]} - {item[2]}")

    print("\n[2. Purity - Commented Code]")
    for item in report["commented_code"]:
        print(f"{item[0]}:{item[1]} - {item[2]}")
