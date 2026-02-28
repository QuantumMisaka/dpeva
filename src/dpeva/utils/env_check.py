import subprocess
import warnings
import re
from packaging import version
from dpeva.constants import MIN_DEEPMD_VERSION

def check_deepmd_version():
    """
    Check if the installed DeepMD-kit version meets the minimum requirement.
    Raises a warning if the version is too old or if 'dp' command is missing.
    """
    try:
        # Check CLI version
        # dp --version output example: "DeePMD-kit v2.2.9" or "DeePMD-kit 3.0.0"
        out = subprocess.check_output(["dp", "--version"], text=True, stderr=subprocess.STDOUT).strip()
        
        # Robust parsing: extract the first version-like string
        # Regex looks for patterns like "1.2.3", "v1.2.3", "2.0.0-beta.1"
        match = re.search(r"v?(\d+\.\d+\.\d+([.\-]\w+)?)", out)
        
        if match:
            v_str = match.group(1)
            current_ver = version.parse(v_str)
            min_ver = version.parse(MIN_DEEPMD_VERSION)
            
            if current_ver < min_ver:
                warnings.warn(
                    f"\n{'='*60}\n"
                    f"WARNING: DeepMD-kit version {v_str} is older than the recommended version {MIN_DEEPMD_VERSION}.\n"
                    f"Some features may not work as expected.\n"
                    f"Please upgrade DeepMD-kit: pip install --upgrade deepmd-kit\n"
                    f"{'='*60}",
                    UserWarning,
                    stacklevel=2
                )
        else:
            # If output format is unexpected, just warn we couldn't parse it but don't fail
            # warnings.warn(f"Could not parse DeepMD-kit version from output: '{out}'")
            pass
            
    except FileNotFoundError:
        warnings.warn(
            f"\n{'='*60}\n"
            f"WARNING: 'dp' command not found in PATH.\n"
            f"DeepMD-kit is required for most workflows.\n"
            f"Please ensure it is installed and added to PATH.\n"
            f"{'='*60}",
            UserWarning,
            stacklevel=2
        )
    except Exception as e:
        # Don't crash app on version check failure
        # In testing environments, subprocess might fail in various ways (e.g. no shell), so we suppress generic errors or log them softly.
        # warnings.warn(f"Failed to check DeepMD-kit version: {e}")
        pass
