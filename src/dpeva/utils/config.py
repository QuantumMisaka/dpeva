import os
from typing import Dict, List, Any, Union

def resolve_config_paths(config: Dict[str, Any], config_file_path: str, path_keys: List[str] = None) -> Dict[str, Any]:
    """
    Resolves relative paths in a configuration dictionary relative to the configuration file's location.
    
    Args:
        config: The configuration dictionary.
        config_file_path: The path to the configuration file (JSON).
        path_keys: A list of keys in the config dictionary that represent file paths. 
                   If None, it defaults to a common set of path keys used in DPEVA.
    
    Returns:
        The configuration dictionary with resolved absolute paths.
    """
    if not config_file_path:
        return config

    config_dir = os.path.dirname(os.path.abspath(config_file_path))
    
    # Default keys if not provided
    if path_keys is None:
        path_keys = [
            "data_path", "model_path", "savedir", "work_dir", 
            "input_json_path", "base_model_path", "training_data_path",
            "desc_dir", "testdata_dir", "training_desc_dir", "root_savedir",
            "output_basedir", "result_dir", "output_dir", "config_path"
        ]
        
    for key in path_keys:
        if key in config:
            val = config[key]
            if isinstance(val, str) and val:
                # If path is not absolute, make it absolute relative to config file
                if not os.path.isabs(val):
                    config[key] = os.path.abspath(os.path.join(config_dir, val))
            elif isinstance(val, list):
                # Handle list of paths if necessary? Currently DPEVA mostly uses single paths.
                # But let's be safe for future or specific fields like template_path which might be list?
                # For now, strict string check.
                pass
                
    return config
