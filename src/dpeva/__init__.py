
"""
DPEVA: Deep Potential EVolution Accelerator
"""

from __future__ import annotations
from dpeva.utils.env_check import check_deepmd_version

__version__ = "0.4.1"

# Perform environment checks on import
# Wrap in try-except to avoid breaking CI/Test environments where dp might be missing
try:
    check_deepmd_version()
except Exception:
    pass