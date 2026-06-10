import os
import logging
import numpy as np
import dpdata
from typing import List, Optional

logger = logging.getLogger(__name__)

def _resolve_target_system_dir(data_dir: str, sys_name: str):
    data_dir_abs = os.path.abspath(data_dir)
    data_base = os.path.basename(os.path.normpath(data_dir_abs))
    normalized = os.path.normpath(sys_name)
    normalized = normalized.lstrip("/\\")
    candidates_rel = []
    seen = set()

    def add_candidate(path_rel: str):
        path_rel_norm = os.path.normpath(path_rel).lstrip("/\\")
        if not path_rel_norm or path_rel_norm in seen:
            return
        seen.add(path_rel_norm)
        candidates_rel.append(path_rel_norm)

    add_candidate(normalized)
    prefix = f"{data_base}{os.sep}"
    if normalized.startswith(prefix):
        add_candidate(normalized[len(prefix):])

    parts = normalized.split(os.sep)
    if len(parts) > 1 and parts[0] == data_base:
        add_candidate(os.path.join(*parts[1:]))

    tried_paths = []
    for rel_path in candidates_rel:
        abs_path = os.path.join(data_dir_abs, rel_path)
        tried_paths.append(abs_path)
        if os.path.isdir(abs_path):
            return abs_path, tried_paths

    return None, tried_paths

def _load_single_path(path, sys_name, fmt="auto"):
    """
    Helper to load a single path with format retry.
    """
    formats_to_try = ["deepmd/npy/mixed", "deepmd/npy"] if fmt == "auto" else [fmt]
    
    for f in formats_to_try:
        try:
            try:
                sys = dpdata.LabeledSystem(path, fmt=f)
            except Exception:
                sys = dpdata.System(path, fmt=f)
            
            # Attach target name for reference
            sys.target_name = sys_name
            # Fix duplicate atom names
            _fix_duplicate_atom_names(sys, sys_name)
            return sys
        except Exception:
            continue
    raise ValueError(f"Failed to load system at {path} with formats {formats_to_try}")

def load_systems(
    data_dir: str, 
    fmt: str = "auto", 
    target_systems: Optional[List[str]] = None
) -> List[dpdata.System]:
    """
    Load systems from a directory using dpdata with auto-format detection.
    
    Args:
        data_dir (str): Path to the data directory.
        fmt (str): Format of the data. 
                   If "auto" (default), attempts to detect "deepmd/npy/mixed" or "deepmd/npy".
        target_systems (list, optional): List of specific system names to load. 
                                         If None, tries to load all subdirectories or the dir itself.
    
    Returns:
        List[dpdata.System]: List of loaded dpdata Systems.
    """
    loaded_systems = []
    
    # 1. Determine directories to load
    if target_systems:
        dirs_to_load = []
        for sys_name in target_systems:
            resolved_dir, tried_paths = _resolve_target_system_dir(data_dir, sys_name)
            if resolved_dir is None:
                logger.warning(
                    f"Data directory not found for system: {sys_name}. Tried: {tried_paths}"
                )
                continue
            dirs_to_load.append((sys_name, resolved_dir))
    else:
        # Optimization: Check if data_dir is itself a system
        try:
            sys = _load_single_path(data_dir, os.path.basename(data_dir), fmt)
            return [sys]
        except Exception:
            pass

        # Try to load as MultiSystems first if no target specified (Auto-Discovery Mode)
        try:
            formats_to_try = ["deepmd/npy/mixed", "deepmd/npy"] if fmt == "auto" else [fmt]
            for f in formats_to_try:
                try:
                    ms = dpdata.MultiSystems.from_file(data_dir, fmt=f)
                    if len(ms) > 0:
                        logger.info(f"Loaded {len(ms)} systems using {f} format.")
                        fixed_systems = []
                        for s in ms:
                            _fix_duplicate_atom_names(s)
                            fixed_systems.append(s)
                        return fixed_systems
                except Exception:
                    continue
            
            # Fallback to scanning directories manually if MultiSystems fails
            subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            # Filter out internal deepmd directories (set.*) to avoid false positives
            subdirs = [d for d in subdirs if not d.startswith("set.")]
            dirs_to_load = [(d, os.path.join(data_dir, d)) for d in subdirs]
            
        except Exception as e:
            logger.error(f"Failed to auto-discover systems in {data_dir}: {e}")
            return []

    # 2. Load each directory
    for name, path in dirs_to_load:
        try:
            sys = _load_single_path(path, name, fmt)
            loaded_systems.append(sys)
        except Exception as e:
            logger.warning(f"Failed to load system at {path}: {e}")
            
    return loaded_systems

def _fix_duplicate_atom_names(sys: dpdata.System, sys_name: str = "Unknown"):
    """
    Detects and merges duplicate atom names in a dpdata System.
    Modifies the system in-place.

    Args:
        sys (dpdata.System): The dpdata System object to fix.
        sys_name (str, optional): Name of the system for logging. Defaults to "Unknown".
    """
    atom_names = sys['atom_names']
    if len(atom_names) != len(set(atom_names)):
        logger.warning(f"Duplicate atom names detected in {sys_name}: {atom_names}. Merging duplicate types.")
        
        # 1. Determine unique names (preserve order)
        new_atom_names = []
        seen = set()
        for name in atom_names:
            if name not in seen:
                new_atom_names.append(name)
                seen.add(name)
        
        # 2. Map old index to new index
        old_to_new_map = {}
        for old_idx, name in enumerate(atom_names):
            new_idx = new_atom_names.index(name)
            old_to_new_map[old_idx] = new_idx
        
        # 3. Update atom_types
        old_atom_types = sys['atom_types']
        new_atom_types = np.array([old_to_new_map[t] for t in old_atom_types], dtype=int)
        
        # 4. Update atom_numbs
        new_atom_numbs = []
        for i in range(len(new_atom_names)):
            count = np.sum(new_atom_types == i)
            new_atom_numbs.append(int(count))
        
        logger.info(f"Merged atom names to: {new_atom_names}")
        
        # 5. Apply changes
        sys.data['atom_names'] = new_atom_names
        sys.data['atom_numbs'] = new_atom_numbs
        sys.data['atom_types'] = new_atom_types
