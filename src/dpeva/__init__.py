
"""
DPEVA: Deep Potential EVolution Accelerator
"""

from __future__ import annotations
from dpeva.utils.env_check import check_deepmd_version

__version__ = "0.5.1"

# Perform environment checks on import
# Wrap in try-except to avoid breaking CI/Test environments where dp might be missing
try:
    check_deepmd_version()
except ImportError:
    # This might happen if dpeva dependencies are not fully installed
    import warnings
    warnings.warn("Failed to import dependencies for environment check.", ImportWarning)
except Exception as e:
    # Don't crash app on version check failure, but warn the user
    import warnings
    warnings.warn(f"DeepMD-kit environment check failed: {e}", UserWarning)
