import os

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
    if "/" in filename or "\\" in filename or os.path.sep in filename or (os.path.altsep and os.path.altsep in filename):
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
    joined = os.path.abspath(os.path.join(base_abs, *paths))
    try:
        common = os.path.commonpath([base_abs, joined])
    except ValueError as e:
        raise ValueError(f"Invalid path relationship between {base_abs} and {joined}: {e}") from e
    if common != base_abs:
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
    raw_name = str(sys_name).strip()
    if not raw_name:
        raise ValueError("System name cannot be empty")
    unified = raw_name.replace("\\", "/")
    if unified.startswith("/"):
        raise ValueError(f"System name '{sys_name}' cannot be an absolute path")
    parts = []
    for part in unified.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"System name '{sys_name}' contains traversal segment '..'")
        validate_filename(part)
        parts.append(part)
    if not parts:
        raise ValueError(f"System name '{sys_name}' reduced to empty path after normalization")
    return os.path.join(*parts)
