
"""
DPEVA: Deep Potential EVolution Accelerator
"""

from __future__ import annotations
from importlib.metadata import PackageNotFoundError, metadata # version

# 版本号

package_metadata = metadata("dpeva")

try:
    __version__ = package_metadata.get("version")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "James Misaka"

# 导入包的核心功能
# from .sampling import clustering, direct, pca, stratified_sampling
# from .uncertain import rnd, rnd_models

# 包的初始化代码（可选）
print(f"Initializing DP-EVA version {__version__}")