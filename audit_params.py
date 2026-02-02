
import ast
import os
import json
import glob
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

@dataclass
class ParameterInfo:
    file_path: str
    context: str  # Class.method or Function
    name: str
    param_type: str
    default_value: str
    is_required: bool
    docstring: str = ""
    source_type: str = "explicit" # explicit (arg) or implicit (config.get)
    risk_level: str = "info" # info, warning (implicit default)

@dataclass
class AuditReport:
    total_params: int = 0
    default_params: int = 0
    missing_docs: int = 0
    implicit_params: int = 0
    parameters: List[ParameterInfo] = field(default_factory=list)

class DocstringParser:
    @staticmethod
    def parse(docstring: str) -> Dict[str, str]:
        """Parses Google/NumPy style docstrings to extract param descriptions."""
        if not docstring:
            return {}
        
        param_docs = {}
        lines = docstring.split('\n')
        current_param = None
        in_args_section = False
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('args:') or line.lower().startswith('parameters:'):
                in_args_section = True
                continue
            if line.lower().startswith('returns:') or line.lower().startswith('raises:'):
                in_args_section = False
                continue
                
            if in_args_section and line:
                # Regex might be better, but simple split for now
                # Format: param_name (type): description
                if ':' in line:
                    parts = line.split(':', 1)
                    # Check if the part before colon looks like a param name
                    # It might contain type info like "param (str)"
                    pre_colon = parts[0].strip()
                    desc = parts[1].strip()
                    
                    # Extract param name
                    if '(' in pre_colon:
                        param_name = pre_colon.split('(')[0].strip()
                    else:
                        param_name = pre_colon
                    
                    current_param = param_name
                    param_docs[current_param] = desc
                elif current_param and line:
                    # Continuation of description
                    param_docs[current_param] += " " + line
                    
        return param_docs

class CodeVisitor(ast.NodeVisitor):
    def __init__(self, file_path):
        self.file_path = file_path
        self.parameters = []
        self.current_class = None
        self.current_function = None

    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        # Skip nested functions (implementation details)
        if self.current_function is not None:
            self.generic_visit(node)
            return

        prev_function = self.current_function
        self.current_function = node.name
        
        context = f"{self.current_class}.{node.name}" if self.current_class else node.name
        
        # 1. Parse Docstring
        docstring = ast.get_docstring(node)
        param_docs = DocstringParser.parse(docstring)
        
        # 2. Parse Explicit Arguments
        defaults = node.args.defaults
        # defaults correspond to the last n arguments
        args = node.args.args
        num_args = len(args)
        num_defaults = len(defaults)
        
        for i, arg in enumerate(args):
            if arg.arg == 'self':
                continue
                
            has_default = i >= (num_args - num_defaults)
            default_val = "None"
            if has_default:
                default_idx = i - (num_args - num_defaults)
                try:
                    default_val = ast.unparse(defaults[default_idx])
                except:
                    default_val = "<complex>"
            
            # Type annotation
            type_hint = "Any"
            if arg.annotation:
                try:
                    type_hint = ast.unparse(arg.annotation)
                except:
                    pass

            desc = param_docs.get(arg.arg, "")
            
            self.parameters.append(ParameterInfo(
                file_path=self.file_path,
                context=context,
                name=arg.arg,
                param_type=type_hint,
                default_value=default_val if has_default else "None",
                is_required=not has_default,
                docstring=desc,
                source_type="explicit",
                risk_level="info"
            ))

        # 3. Scan for Implicit Config Usage (config.get)
        # We walk the body of the function to find Call nodes
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_Call(self, node):
        # Look for config.get('key', default)
        # Structure: Attribute(value=Name(id='config'), attr='get')
        is_config_get = False
        
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'get':
                # Check if it's config.get or self.config.get
                if isinstance(node.func.value, ast.Name) and 'config' in node.func.value.id:
                    is_config_get = True
                elif isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'config':
                    is_config_get = True
        
        if is_config_get and node.args:
            # Extract key
            key = "unknown"
            if isinstance(node.args[0], ast.Constant): # Python 3.8+
                key = node.args[0].value
            elif isinstance(node.args[0], ast.Str): # Older Python
                key = node.args[0].s
            
            # Extract default
            default_val = "None"
            if len(node.args) > 1:
                try:
                    default_val = ast.unparse(node.args[1])
                except:
                    default_val = "<complex>"
            
            context = f"{self.current_class}.{self.current_function}" if self.current_class else self.current_function
            
            self.parameters.append(ParameterInfo(
                file_path=self.file_path,
                context=context,
                name=key,
                param_type="Unknown",
                default_value=default_val,
                is_required=False, # .get implies optional
                docstring="", # Implicit usually has no docstring unless mapped manually
                source_type="implicit",
                risk_level="warning"
            ))

        self.generic_visit(node)

def run_audit(root_dir):
    report = AuditReport()
    
    # Files to ignore
    ignore_patterns = ["tests", "setup.py", "conftest.py"]
    
    for root, dirs, files in os.walk(root_dir):
        if any(p in root for p in ignore_patterns):
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # print(f"Scanning {file_path}...")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                        visitor = CodeVisitor(file_path)
                        visitor.visit(tree)
                        report.parameters.extend(visitor.parameters)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")

    # Calculate stats
    report.total_params = len(report.parameters)
    report.default_params = sum(1 for p in report.parameters if not p.is_required)
    report.missing_docs = sum(1 for p in report.parameters if not p.docstring and p.source_type == "explicit")
    report.implicit_params = sum(1 for p in report.parameters if p.source_type == "implicit")
    
    return report

def generate_markdown(report: AuditReport):
    md = "# DP-EVA User Configurable Parameters Audit\n\n"
    
    md += "## Summary\n"
    md += f"- **Total Parameters**: {report.total_params}\n"
    md += f"- **Parameters with Defaults**: {report.default_params}\n"
    md += f"- **Implicit (config.get) Parameters**: {report.implicit_params}\n"
    md += f"- **Missing Docstrings (Explicit)**: {report.missing_docs}\n\n"
    
    md += "## Implicit Dependency Risks (Top 10)\n"
    implicit = [p for p in report.parameters if p.source_type == "implicit"]
    for p in implicit[:10]:
        md += f"- `{p.name}` in `{p.context}` (Default: `{p.default_value}`)\n"
    md += "\n"
    
    md += "## Detailed Parameter List\n\n"
    md += "| Context | Parameter | Type | Default | Required | Description | Source |\n"
    md += "|---|---|---|---|---|---|---|\n"
    
    for p in report.parameters:
        # relative path
        rel_path = p.file_path.split("dpeva/src/dpeva/")[-1] if "dpeva/src/dpeva/" in p.file_path else os.path.basename(p.file_path)
        context_link = f"[{p.context}]({p.file_path})"
        
        row = f"| {context_link} | `{p.name}` | {p.param_type} | `{p.default_value}` | {p.is_required} | {p.docstring[:50]}... | {p.source_type} |\n"
        md += row
        
    return md

if __name__ == "__main__":
    target_dir = "/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva"
    report = run_audit(target_dir)
    
    # Dump JSON
    with open("parameter_audit.json", "w") as f:
        json.dump(asdict(report), f, indent=2)
        
    # Dump Markdown
    md_content = generate_markdown(report)
    with open("parameter_audit.md", "w") as f:
        f.write(md_content)
        
    print("Audit completed. Generated parameter_audit.json and parameter_audit.md")
