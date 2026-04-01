"""I/O package public exports.

This package collects reusable I/O helpers shared by multiple workflows.
Workflow execution paths import concrete modules directly.
"""

from .dataproc import DPTestResultParser

__all__ = ["DPTestResultParser"]
