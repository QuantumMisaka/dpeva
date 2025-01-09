
"""
DPEVA: Deep Potential EVolution Accelerator
"""

# 版本号
__version__ = "0.1.0"
__author__ = "James Misaka"

# 导入包的核心功能
from .sampling import clustering, direct, pca, stratified_sampling
from .uncertain import rnd, rnd_models

# 包的初始化代码（可选）
print(f"Initializing DP-EVA version {__version__}")