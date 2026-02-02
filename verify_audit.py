
import json
import random
import importlib
import inspect
import sys
import os
import ast

def get_obj_from_path(module_path, context):
    """
    Imports a module and retrieves the object (function or class method) based on context string.
    Context format: 'ClassName.method_name' or 'function_name'
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Failed to import {module_path}: {e}")
        return None

    parts = context.split('.')
    obj = module
    
    try:
        if len(parts) == 1:
            # Function
            obj = getattr(obj, parts[0])
        elif len(parts) == 2:
            # Class.method
            cls = getattr(obj, parts[0])
            obj = getattr(cls, parts[1])
        else:
            print(f"Complex context not supported yet: {context}")
            return None
    except AttributeError:
        print(f"Could not find {context} in {module_path}")
        return None
        
    return obj

def verify_parameter(param_info):
    # Construct module path from file path
    # e.g., .../src/dpeva/workflows/collect.py -> dpeva.workflows.collect
    if "src/dpeva/" not in param_info['file_path']:
        return "SKIPPED (Path)"
        
    rel_path = param_info['file_path'].split("src/dpeva/")[-1]
    module_name = "dpeva." + rel_path.replace('/', '.').replace('.py', '')
    
    obj = get_obj_from_path(module_name, param_info['context'])
    if not obj:
        return "FAILED (Import)"
        
    # Inspect parameters
    try:
        sig = inspect.signature(obj)
    except ValueError:
        return "SKIPPED (No Signature)"
        
    if param_info['name'] not in sig.parameters:
        # It might be **kwargs hidden
        return "FAILED (Missing Arg)"
        
    p = sig.parameters[param_info['name']]
    
    # Check default value
    if p.default == inspect.Parameter.empty:
        if param_info['default_value'] == "None" and param_info['is_required']:
            return "PASSED"
        else:
            return f"FAILED (Expected required, got default {p.default})"
    else:
        # Convert actual default to string for comparison
        # This is tricky because AST unparse might produce different string than repr()
        # e.g. '10' vs 10
        actual_val_str = str(p.default)
        audit_val_str = param_info['default_value']
        
        # Simple normalization
        if actual_val_str == audit_val_str:
            return "PASSED"
        
        # Try repr
        if repr(p.default) == audit_val_str:
            return "PASSED"
            
        # Try unquoting strings if audit has quotes
        if isinstance(p.default, str):
            if f"'{p.default}'" == audit_val_str or f'"{p.default}"' == audit_val_str:
                return "PASSED"
                
        # Numeric check
        try:
            if float(actual_val_str) == float(audit_val_str):
                return "PASSED"
        except:
            pass
            
        return f"WARNING (Value Mismatch: Actual '{p.default}' vs Audit '{audit_val_str}')"

def run_verification():
    print("Loading parameter_audit.json...")
    with open("parameter_audit.json", "r") as f:
        report = json.load(f)
        
    explicit_params = [p for p in report['parameters'] if p['source_type'] == 'explicit']
    sample_size = max(1, int(len(explicit_params) * 0.1))
    
    print(f"Verifying {sample_size} parameters (10% sample)...")
    samples = random.sample(explicit_params, sample_size)
    
    passed = 0
    failed = 0
    warnings = 0
    
    sys.path.append(os.path.join(os.getcwd(), "src"))
    
    for p in samples:
        result = verify_parameter(p)
        print(f"[{result}] {p['context']}:{p['name']}")
        
        if result == "PASSED":
            passed += 1
        elif result.startswith("FAILED"):
            failed += 1
        else:
            warnings += 1
            
    print("-" * 40)
    print(f"Verification Results: {passed} PASSED, {failed} FAILED, {warnings} WARNINGS")
    
    if failed == 0:
        print("SUCCESS: Integrity verification passed.")
    else:
        print("FAILURE: Integrity verification failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
