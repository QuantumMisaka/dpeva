#!/usr/bin/env python3
import os
import argparse
import sys
import fnmatch

def modify_input_file(filepath, changes):
    """
    Modifies an ABACUS INPUT file with the given changes.
    changes: dict of key -> value
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    new_lines = []
    modified_keys = set()
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines, comments (except possibly headers if we want to preserve them, which we do)
        # We just want to identify key-value pairs.
        if not stripped or stripped.startswith('#') or stripped.startswith('INPUT_PARAMETERS'):
            new_lines.append(line)
            continue
        
        # Parse key-value
        parts = stripped.split()
        if len(parts) >= 2:
            key = parts[0]
            if key in changes:
                
                original_key_end_index = line.find(key) + len(key)
                value_start_index = -1
                
                # Find first non-whitespace after key
                for i in range(original_key_end_index, len(line)):
                    if not line[i].isspace():
                        value_start_index = i
                        break
                
                if value_start_index != -1:
                    # Construct new line: prefix + new_value + newline
                    prefix = line[:value_start_index]
                    new_line = f"{prefix}{changes[key]}\n"
                else:
                    # Fallback
                    new_line = f"{key:<40}{changes[key]}\n"
                
                new_lines.append(new_line)
                modified_keys.add(key)
                print(f"  Modified {key}: {parts[1]} -> {changes[key]}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    # Add missing keys
    for key, value in changes.items():
        if key not in modified_keys:
            new_lines.append(f"{key:<40}{value}\n")
            print(f"  Added {key}: {value}")
            
    try:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        return True
    except Exception as e:
        print(f"Error writing {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch modify ABACUS INPUT files.")
    parser.add_argument("root_dir", help="Root directory to search for INPUT files")
    parser.add_argument("--beta", type=float, help="mixing_beta value")
    parser.add_argument("--ndim", type=int, help="mixing_ndim value")
    parser.add_argument("--scf_nmax", type=int, help="scf_nmax value")
    parser.add_argument("--scf_os_stop", help="scf_os_stop value")
    parser.add_argument("--scf_os_ndim", type=int, help="scf_os_ndim value")
    parser.add_argument("--set", action='append', help="Set arbitrary key=value. E.g. --set key=value")
    parser.add_argument("--pattern", default=None, help="Glob pattern to match file paths relative to root_dir (e.g. '*/*/N_*/INPUT')")
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root_dir)
    changes = {}
    
    if args.beta is not None:
        changes["mixing_beta"] = str(args.beta)
    if args.ndim is not None:
        changes["mixing_ndim"] = str(args.ndim)
    if args.scf_nmax is not None:
        changes["scf_nmax"] = str(args.scf_nmax)
    if args.scf_os_stop is not None:
        changes["scf_os_stop"] = str(args.scf_os_stop)
    if args.scf_os_ndim is not None:
        changes["scf_os_ndim"] = str(args.scf_os_ndim)

    if args.set:
        for s in args.set:
            if '=' in s:
                k, v = s.split('=', 1)
                changes[k.strip()] = v.strip()
            else:
                print(f"Warning: Ignoring invalid set argument '{s}'. Must be KEY=VALUE.")

    if not changes:
        print("No changes specified. Use --beta, --ndim, --scf_nmax, --scf_os_stop, --scf_os_ndim or --set.")
        return
    
    print(f"Searching in: {root_dir}")
    print(f"Applying changes: {changes}")
    if args.pattern:
        print(f"Filtering with pattern: {args.pattern}")
    
    count = 0
    # Walk through directory
    for root, dirs, files in os.walk(root_dir):
        if "INPUT" in files:
            filepath = os.path.join(root, "INPUT")
            
            if args.pattern:
                relpath = os.path.relpath(filepath, root_dir)
                if not fnmatch.fnmatch(relpath, args.pattern):
                    # print(f"Skipping {relpath} (no match)")
                    continue
            
            print(f"Processing: {filepath}")
            if modify_input_file(filepath, changes):
                count += 1
                
    print(f"Done. Modified {count} files.")

if __name__ == "__main__":
    main()
