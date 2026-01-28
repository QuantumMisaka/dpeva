
#!/usr/bin/env python3
"""
Descriptor Consistency Verification Tool
========================================

This script verifies the consistency of descriptors calculated from two different data formats:
1. `deepmd/npy/mixed` (typically `sampled_dpdata`): A format where atom types can vary between frames, 
   often using a `real_atom_types.npy` file to specify actual types per frame.
2. `deepmd/npy` (typically `sampled_dpdata_npy`): The standard DeepMD-kit format.

Problem Statement:
------------------
When converting or sampling data, the order of atoms or the storage format might change. 
Specifically, `mixed` format datasets might appear to have a single atom type (e.g., all 'H') in `type.raw` 
while storing real element information in `set.XXX/real_atom_types.npy`. 
Direct comparison of descriptors requires:
1. Correctly interpreting the atom types for each frame.
2. Matching structures between the two datasets, as they might be stored in different orders or folders.
3. Handling atom permutation invariances by sorting atoms and coordinates to generate a unique structural fingerprint.

Usage:
------
    python verify_desc_consistency.py --mixed_dir <path> --npy_dir <path> --desc_mixed <path> --desc_npy <path>

    # To run with default paths (reproduce verification test):
    python verify_desc_consistency.py

"""

import os
import argparse
import numpy as np
import dpdata
from collections import defaultdict
import sys

# Default paths from the user's environment for easy reproduction
DEFAULT_MIXED_DIR = "/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/test/desc-test/sampled_dpdata"
DEFAULT_NPY_DIR = "/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/test/desc-test/sampled_dpdata_npy"
DEFAULT_DESC_MIXED_DIR = "/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/test/desc-test/desc_train"
DEFAULT_DESC_NPY_DIR = "/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/test/desc-test/desc_train_npy"

def get_sorted_info(ls, frame_idx, desc, real_types=None):
    """
    Generate a unique structural fingerprint and sort the descriptor to match a canonical atom order.

    Parameters
    ----------
    ls : dpdata.LabeledSystem
        The loaded system containing coordinates and atom names.
    frame_idx : int
        The index of the frame to process.
    desc : np.ndarray
        The descriptor array for this frame. Shape: (n_atoms, n_features) or (n_atoms * n_features,).
    real_types : np.ndarray, optional
        The actual atom types for this frame (used for 'mixed' format). 
        If None, uses `ls['atom_types']`.
        Shape: (n_frames, n_atoms) usually, but here we pass the row for the specific frame or handle it outside.
        Actually, in the caller we pass `real_types[frame_idx]` if it's the whole array, 
        or `ls['atom_types']` if static.
        
        WAIT: The previous implementation passed `real_types` as the full array or None, 
        and sliced it inside. Let's stick to that pattern for safety.
        
        Correct logic:
        If `real_types` (the full array for the system) is provided, we use `real_types[frame_idx]`.
        Otherwise, we use `ls['atom_types']`.

    Returns
    -------
    key : tuple
        A unique key identifying the structure: (tuple(sorted_symbols), bytes(sorted_rounded_coords)).
    sorted_desc : np.ndarray
        The descriptor reordered according to the sorted atom indices.
    """
    atom_names = ls['atom_names']
    
    if real_types is not None:
        # Use real types for this frame (Mixed format specific)
        current_types = real_types[frame_idx]
    else:
        # Fallback to system global types (Standard NPY format)
        current_types = ls['atom_types']
        
    symbols = [atom_names[t] for t in current_types]
    coords = ls['coords'][frame_idx]
    
    # Create structured array for sorting: Symbol -> X -> Y -> Z
    # We include 'orig_idx' to track the permutation of atoms
    dtype = [('sym', 'U2'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('orig_idx', 'i4')]
    struct_data = []
    for idx, (s, c) in enumerate(zip(symbols, coords)):
        struct_data.append((s, c[0], c[1], c[2], idx))
    
    arr = np.array(struct_data, dtype=dtype)
    # Sort to canonical order
    sorted_indices = np.argsort(arr, order=['sym', 'x', 'y', 'z'])
    sorted_arr = arr[sorted_indices]
    
    # Generate Key
    sorted_symbols = tuple(sorted_arr['sym'])
    sorted_coords = np.array([[x['x'], x['y'], x['z']] for x in sorted_arr])
    # Round coordinates to 4 decimal places to tolerate minor floating point differences
    rounded_coords = np.round(sorted_coords, 4)
    key = (sorted_symbols, rounded_coords.tobytes())
    
    # Sort Descriptor to match the canonical atom order
    # Descriptor shape: (n_atoms, n_features) or flattened
    if desc.ndim == 2:
        sorted_desc = desc[sorted_indices]
    elif desc.ndim == 1:
        n_atoms = len(symbols)
        if n_atoms > 0:
            n_feat = desc.shape[0] // n_atoms
            # Reshape to (n_atoms, n_feat) -> reorder -> flatten back
            d_reshaped = desc.reshape(n_atoms, n_feat)
            sorted_desc = d_reshaped[sorted_indices].flatten()
        else:
            sorted_desc = desc
    else:
        # Fallback for unexpected shapes
        sorted_desc = desc
        
    return key, sorted_desc

def load_and_index_mixed(data_dir, desc_dir):
    """
    Load 'mixed' format data and build a lookup database.

    Parameters
    ----------
    data_dir : str
        Path to the mixed data directory.
    desc_dir : str
        Path to the directory containing descriptors for the mixed data.

    Returns
    -------
    dict
        A dictionary mapping structure keys to a list of descriptor data.
        Key: (sorted_symbols, sorted_coords_bytes)
        Value: List of dicts {'desc': np.array, 'folder': str, 'frame': int}
    """
    print(f"Loading Mixed Data from: {data_dir}")
    print(f"Loading Mixed Descriptors from: {desc_dir}")
    
    mixed_db = defaultdict(list)
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    count = 0
    for folder in folders:
        sys_path = os.path.join(data_dir, folder)
        desc_path = os.path.join(desc_dir, f"{folder}.npy")
        
        if not os.path.exists(desc_path):
            continue
            
        try:
            # Load system using dpdata
            ls = dpdata.LabeledSystem(sys_path, fmt='deepmd/npy')
            full_desc = np.load(desc_path)
            
            # Crucial Step: Check for real_atom_types.npy in set.000
            # This file defines the actual element types for each frame in mixed format
            real_types = None
            real_types_path = os.path.join(sys_path, "set.000", "real_atom_types.npy")
            if os.path.exists(real_types_path):
                real_types = np.load(real_types_path)
                if real_types.shape[0] != len(ls):
                    print(f"Warning: real_atom_types length {real_types.shape[0]} != system length {len(ls)} in {folder}. Ignoring real_types.")
                    real_types = None
            
            # Sanity check frame counts
            if len(ls) != full_desc.shape[0]:
                print(f"Skipping {folder}: mismatch len(ls)={len(ls)} vs desc={full_desc.shape[0]}")
                continue
                
            for i in range(len(ls)):
                curr_desc = full_desc[i]
                
                # Get canonical key and sorted descriptor
                key, sorted_desc = get_sorted_info(ls, i, curr_desc, real_types)
                
                mixed_db[key].append({
                    'desc': sorted_desc,
                    'folder': folder,
                    'frame': i
                })
                count += 1
                
        except Exception as e:
            print(f"Error loading mixed folder {folder}: {e}")
            
    print(f"Successfully indexed {count} frames from Mixed data.")
    return mixed_db

def verify_npy(data_dir, desc_dir, mixed_db, tolerance=1e-5):
    """
    Verify 'npy' format data against the indexed 'mixed' database.

    Parameters
    ----------
    data_dir : str
        Path to the npy data directory.
    desc_dir : str
        Path to the directory containing descriptors for the npy data.
    mixed_db : dict
        The lookup database created by `load_and_index_mixed`.
    tolerance : float
        The maximum allowed difference between descriptors to consider them a match.
    """
    print(f"\nVerifying NPY Data from: {data_dir}")
    print(f"Verifying NPY Descriptors from: {desc_dir}")
    
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    stats = {
        'total': 0,
        'matched_identical': 0, # diff == 0
        'matched_close': 0,     # diff < tolerance
        'mismatched': 0,        # diff >= tolerance
        'missing': 0,           # structure not found in mixed_db
        'error': 0
    }
    
    for folder in folders:
        sys_path = os.path.join(data_dir, folder)
        desc_path = os.path.join(desc_dir, f"{folder}.npy")
        
        if not os.path.exists(desc_path):
            continue
            
        try:
            ls = dpdata.LabeledSystem(sys_path, fmt='deepmd/npy')
            full_desc = np.load(desc_path)
            
            if len(ls) != full_desc.shape[0]:
                print(f"Skipping {folder}: mismatch len(ls)={len(ls)} vs desc={full_desc.shape[0]}")
                continue
                
            for i in range(len(ls)):
                stats['total'] += 1
                curr_desc = full_desc[i]
                
                # NPY format usually implies static atom types, so we pass None for real_types
                key, sorted_desc = get_sorted_info(ls, i, curr_desc, None)
                
                if key in mixed_db:
                    matches = mixed_db[key]
                    best_diff = float('inf')
                    
                    # Compare against all structures in mixed DB that share the same fingerprint
                    for match in matches:
                        ref_desc = match['desc']
                        
                        # Check shape consistency
                        if ref_desc.shape != sorted_desc.shape:
                            continue
                            
                        diff = np.max(np.abs(ref_desc - sorted_desc))
                        if diff < best_diff:
                            best_diff = diff
                    
                    if best_diff == 0:
                        stats['matched_identical'] += 1
                    elif best_diff < tolerance:
                        stats['matched_close'] += 1
                    else:
                        stats['mismatched'] += 1
                        if stats['mismatched'] <= 5:
                            print(f"Mismatch! Diff: {best_diff:.2e}. Folder: {folder}, Frame: {i}")
                else:
                    stats['missing'] += 1
                    if stats['missing'] <= 5:
                         print(f"Missing match in Mixed DB for {folder} frame {i}")
                    
        except Exception as e:
            print(f"Error processing npy folder {folder}: {e}")
            stats['error'] += 1
            
    print("\n" + "="*40)
    print("Verification Results Summary")
    print("="*40)
    print(f"Total Frames Checked      : {stats['total']}")
    print(f"Identical Matches (Diff=0): {stats['matched_identical']}")
    print(f"Close Matches (<{tolerance:.1e})  : {stats['matched_close']}")
    print(f"Mismatches (>={tolerance:.1e})   : {stats['mismatched']}")
    print(f"Missing (No Struct Match) : {stats['missing']}")
    print(f"Errors (Load/Process)     : {stats['error']}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description="Verify descriptor consistency between Mixed and NPY DeepMD formats.")
    
    parser.add_argument("--mixed_dir", type=str, default=DEFAULT_MIXED_DIR,
                        help=f"Directory containing Mixed format dpdata (default: {DEFAULT_MIXED_DIR})")
    parser.add_argument("--npy_dir", type=str, default=DEFAULT_NPY_DIR,
                        help=f"Directory containing NPY format dpdata (default: {DEFAULT_NPY_DIR})")
    parser.add_argument("--desc_mixed", type=str, default=DEFAULT_DESC_MIXED_DIR,
                        help=f"Directory containing descriptors for Mixed data (default: {DEFAULT_DESC_MIXED_DIR})")
    parser.add_argument("--desc_npy", type=str, default=DEFAULT_DESC_NPY_DIR,
                        help=f"Directory containing descriptors for NPY data (default: {DEFAULT_DESC_NPY_DIR})")
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="Tolerance for floating point comparison of descriptors (default: 1e-5)")

    args = parser.parse_args()

    # Verify directories exist
    for p in [args.mixed_dir, args.npy_dir, args.desc_mixed, args.desc_npy]:
        if not os.path.exists(p):
            print(f"Error: Directory not found: {p}")
            sys.exit(1)

    # 1. Load and Index Mixed Data
    db = load_and_index_mixed(args.mixed_dir, args.desc_mixed)
    
    if not db:
        print("Error: No data loaded from Mixed directory. Exiting.")
        sys.exit(1)

    # 2. Verify NPY Data against Index
    verify_npy(args.npy_dir, args.desc_npy, db, args.tolerance)

if __name__ == "__main__":
    main()
