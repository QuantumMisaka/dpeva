import ast
import os
import json
import re
from collections import defaultdict

TARGET_DIR = "/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows"

def parse_google_docstring(docstring):
    """
    Parses a Google-style docstring to extract parameter descriptions.
    Returns a dict: {param_name: {'desc': str, 'type': str}}
    """
    if not docstring:
        return {}
    
    params = {}
    lines = docstring.split('\n')
    current_section = None
    current_param = None
    
    # Regex for "param_name (type): description"
    # Handling indentation is key
    param_pattern = re.compile(r'^\s*(\w+)\s*(\((.*?)\))?:\s*(.*)')
    
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "args:":
            current_section = "args"
            continue
        elif stripped.lower() in ["returns:", "raises:", "attributes:", "example:"]:
            current_section = None
            continue
            
        if current_section == "args":
            match = param_pattern.match(line)
            if match:
                name = match.group(1)
                type_str = match.group(3)
                desc = match.group(4)
                params[name] = {
                    'desc': desc.strip(),
                    'type': type_str if type_str else "Unknown"
                }
                current_param = name
            elif current_param and line.strip():
                # Continuation of description
                params[current_param]['desc'] += " " + line.strip()
                
    return params

class WorkflowVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.workflow_data = {} # {ClassName: {FuncName: [Params]}}
        self.current_class = None
        
    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.workflow_data[node.name] = {}
        self.generic_visit(node)
        self.current_class = None
        
    def visit_FunctionDef(self, node):
        if self.current_class:
            func_name = node.name
            # Only interested in __init__ (Configuration) and run/execute methods (Core Logic)
            # Or just all public methods
            if func_name.startswith('_') and func_name != '__init__':
                return

            docstring = ast.get_docstring(node)
            parsed_docs = parse_google_docstring(docstring)
            
            params = []
            defaults = node.args.defaults
            # align defaults with args (defaults are at the end)
            args_with_defaults = len(defaults)
            offset = len(node.args.args) - args_with_defaults
            
            for i, arg in enumerate(node.args.args):
                if arg.arg == 'self':
                    continue
                
                default_val = "Required"
                if i >= offset:
                    default_node = defaults[i - offset]
                    # Try to extract literal value
                    try:
                        if isinstance(default_node, ast.Constant):
                            default_val = default_node.value
                        elif isinstance(default_node, ast.Name): # e.g. None
                            default_val = default_node.id
                        elif isinstance(default_node, ast.Attribute):
                            default_val = f"{default_node.value.id}.{default_node.attr}" # e.g. os.getcwd
                        else:
                            default_val = "ComplexExpression"
                    except:
                        default_val = "Unknown"

                param_info = {
                    'name': arg.arg,
                    'default': default_val,
                    'doc_info': parsed_docs.get(arg.arg, {})
                }
                params.append(param_info)
                
            self.workflow_data[self.current_class][func_name] = {
                'docstring': docstring,
                'params': params
            }
        self.generic_visit(node)

def audit_workflows():
    results = {}
    all_params_flat = [] # For redundancy check
    
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                module_name = file.replace('.py', '')
                
                with open(filepath, "r") as f:
                    tree = ast.parse(f.read())
                    
                visitor = WorkflowVisitor(module_name)
                visitor.visit(tree)
                results[module_name] = visitor.workflow_data
                
                # Flatten for redundancy
                for cls_name, methods in visitor.workflow_data.items():
                    for func_name, info in methods.items():
                        for p in info['params']:
                            all_params_flat.append({
                                'module': module_name,
                                'class': cls_name,
                                'function': func_name,
                                'name': p['name'],
                                'default': p['default'],
                                'doc_desc': p['doc_info'].get('desc', '')
                            })

    return results, all_params_flat

def check_redundancy(all_params):
    # 1. Exact name Match
    by_name = defaultdict(list)
    for p in all_params:
        by_name[p['name']].append(p)
        
    redundancy_report = []
    
    for name, occurrences in by_name.items():
        if len(occurrences) > 1:
            # Check for inconsistent defaults
            defaults = set(str(o['default']) for o in occurrences)
            if len(defaults) > 1:
                redundancy_report.append({
                    'type': 'Inconsistent Default',
                    'name': name,
                    'details': [{'loc': f"{o['module']}.{o['class']}.{o['function']}", 'default': o['default']} for o in occurrences]
                })
            else:
                 redundancy_report.append({
                    'type': 'Duplicate (Consistent)',
                    'name': name,
                    'count': len(occurrences),
                    'locations': [f"{o['module']}.{o['class']}.{o['function']}" for o in occurrences]
                })
                
    # 2. Similar Names (e.g. uq_trust_lo vs trust_lo)
    # Simple check: one name is a substring of another
    keys = sorted(by_name.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            if len(k1) > 4 and len(k2) > 4: # avoid short generic names
                if k1 in k2 or k2 in k1:
                    redundancy_report.append({
                        'type': 'Similar Name (Potential Alias)',
                        'pair': (k1, k2),
                        'locations_k1': [f"{o['module']}.{o['class']}" for o in by_name[k1]],
                        'locations_k2': [f"{o['module']}.{o['class']}" for o in by_name[k2]]
                    })

    return redundancy_report

if __name__ == "__main__":
    structure, flat_params = audit_workflows()
    redundancy = check_redundancy(flat_params)
    
    output = {
        'structure': structure,
        'redundancy': redundancy
    }
    
    print(json.dumps(output, indent=2))
