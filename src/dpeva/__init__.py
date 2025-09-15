
"""
DPEVA: Deep Potential EVolution Accelerator
"""

from __future__ import annotations
from importlib.metadata import PackageNotFoundError, metadata # version

__version__ = "0.2.0"

try:
    __version__ = package_metadata.get("version")
except PackageNotFoundError:
    __version__ = "0.2.0-alpha"

__author__ = "James Misaka"