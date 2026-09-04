import os
import re
from typing import Dict, List, Any


_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _resolve_local_path(value: str, config_dir: str) -> str:
    """Resolve a config-local path while preserving explicit URI references."""
    expanded = os.path.expanduser(os.path.expandvars(value))
    if _URI_SCHEME.match(expanded):
        return expanded
    return expanded if os.path.isabs(expanded) else os.path.abspath(os.path.join(config_dir, expanded))

def resolve_config_paths(config: Dict[str, Any], config_file_path: str, path_keys: List[str] = None) -> Dict[str, Any]:
    """
    Resolves relative paths in a configuration dictionary relative to the configuration file's location.
    Also expands environment variables (e.g. $HOME) in path strings.
    
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
            "output_basedir", "result_dir", "output_dir", "config_path",
            "input_data_path", "pp_dir", "orb_dir", "dataset_dir",
            "template_path", "existing_training_data_path", "merged_training_data_path",
            "llpr_train_feature_dir", "llpr_candidate_feature_dir",
            "llpr_model_path", "llpr_last_layer_weights_path",
            "llpr_candidate_energy_path", "llpr_state_path",
            "llpr_save_state_path", "llpr_ensemble_output_path", "model_ref_paths",
            # Evaluation-card evidence is resolved from the configuration
            # file, so a portable recipe can be copied as a package.
            "output_path", "model_ref_path", "in_domain_cumulative_path", "iter11_last_wave_path",
            "historical_domain_path", "matpes_retention_path", "training_cost_path",
            "surface_slice_path", "dataset_manifest_paths", "downstream_feedback_ref",
        ]
        
    for key in path_keys:
        if key in config:
            val = config[key]
            if isinstance(val, str) and val:
                # Expand user (~) and environment variables ($HOME, etc.)
                config[key] = _resolve_local_path(val, config_dir)
            elif isinstance(val, list) and key in {"model_ref_paths", "dataset_manifest_paths"}:
                resolved_paths = []
                for item in val:
                    if not isinstance(item, str):
                        resolved_paths.append(item)
                        continue
                    resolved_paths.append(_resolve_local_path(item, config_dir))
                config[key] = resolved_paths
                
    return config
