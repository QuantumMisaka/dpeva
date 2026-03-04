import os
import pathlib
import re
from typing import Optional

def validate_filename(filename: str) -> str:
    """
    Validates that a filename is safe to use in a path.
    Rejects filenames containing directory traversal sequences or restricted characters.
    
    Args:
        filename: The filename to validate.
        
    Returns:
        The validated filename.
        
    Raises:
        ValueError: If the filename is invalid.
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
        
    # Check for path separators
    if os.path.sep in filename or (os.path.altsep and os.path.altsep in filename):
        raise ValueError(f"Filename '{filename}' contains path separators")
        
    # Check for traversal
    if ".." in filename:
        raise ValueError(f"Filename '{filename}' contains traversal sequence '..'")
        
    # Optional: Restrict allowed characters (e.g. alphanumeric, dot, dash, underscore)
    # This might be too strict for existing datasets, so we stick to path safety.
    # But checking for control characters is good.
    # Control chars: 0-31 and 127
    if any(ord(c) < 32 or ord(c) == 127 for c in filename):
        raise ValueError(f"Filename '{filename}' contains control characters")
        
    return filename

def safe_join(base_dir: str, *paths: str) -> str:
    """
    Safely joins paths ensuring the result is within base_dir.
    
    Args:
        base_dir: The trusted base directory.
        *paths: Path components to join.
        
    Returns:
        The absolute joined path.
        
    Raises:
        ValueError: If the resulting path attempts to traverse outside base_dir.
    """
    base_abs = os.path.abspath(base_dir)
    joined = os.path.abspath(os.path.join(base_dir, *paths))
    
    if not joined.startswith(base_abs):
        raise ValueError(f"Path traversal attempt detected: {joined} is not within {base_abs}")
        
    return joined

def normalize_sys_name(sys_name: str) -> str:
    """
    Normalizes a system name to be safe for filesystem usage.
    Replaces path separators with underscores and validates safety.
    
    Args:
        sys_name: The raw system name.
        
    Returns:
        A sanitized system name safe for directory naming.
    """
    # Replace path separators with underscores to flatten structure if needed,
    # OR reject them.
    # For DP-EVA, sys_name is often used as directory name.
    # If sys_name comes from "path/to/sys", we might want "sys" or "path_to_sys".
    # But usually sys.target_name is just the leaf or a unique identifier.
    
    # Strategy: Replace separators with underscores
    clean_name = sys_name.replace(os.path.sep, "_")
    if os.path.altsep:
        clean_name = clean_name.replace(os.path.altsep, "_")
        
    # Remove traversal sequences
    while ".." in clean_name:
        clean_name = clean_name.replace("..", "")
        
    # Validate final result
    try:
        validate_filename(clean_name)
    except ValueError as e:
        # Fallback or strict error? 
        # Given we cleaned it, if it still fails, it's weird.
        # But '..' replacement might leave empty string or still fail control chars.
        if not clean_name:
            raise ValueError(f"System name '{sys_name}' reduced to empty string after sanitization")
        raise e
        
    return clean_name
