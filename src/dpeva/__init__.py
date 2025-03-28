
"""
DPEVA: Deep Potential EVolution Accelerator
"""

from __future__ import annotations
from importlib.metadata import PackageNotFoundError, metadata # version

package_metadata = metadata("dpeva")

try:
    __version__ = package_metadata.get("version")
except PackageNotFoundError:
    __version__ = "0.2.0-alpha"

__author__ = "James Misaka"

# from .sampling import clustering, direct, pca, stratified_sampling
# from .uncertain import rnd, rnd_models

print(f"Initializing DP-EVA version {__version__}")