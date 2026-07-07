"""Slurm-safe naming helpers."""

from __future__ import annotations

import re


def normalize_slurm_job_name(raw: str, fallback: str = "dpeva-job", max_length: int = 128) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw)).strip("-._")
    value = re.sub(r"-{2,}", "-", value)
    if not value:
        value = fallback
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    return value[:max_length]
