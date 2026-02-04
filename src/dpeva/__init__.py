
"""
DPEVA: Deep Potential EVolution Accelerator
"""

from __future__ import annotations
from dpeva.utils.env_check import check_deepmd_version

__version__ = "0.4.0"

# Perform environment checks on import
check_deepmd_version()