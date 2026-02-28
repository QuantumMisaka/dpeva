import os
import logging
import numpy as np
import dpdata
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

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
    
    # Helper to load a single path with format retry
    def _load_single_path(path, sys_name):
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

    # 1. Determine directories to load
    if target_systems:
        dirs_to_load = []
        for sys_name in target_systems:
            d = os.path.join(data_dir, sys_name)
            if not os.path.isdir(d):
                logger.warning(f"Data directory not found for system: {sys_name} at {d}")
                continue
            dirs_to_load.append((sys_name, d))
    else:
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
            dirs_to_load = [(d, os.path.join(data_dir, d)) for d in subdirs]
            
        except Exception as e:
            logger.error(f"Failed to auto-discover systems in {data_dir}: {e}")
            return []

    # 2. Load each directory
    for name, path in dirs_to_load:
        try:
            sys = _load_single_path(path, name)
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
