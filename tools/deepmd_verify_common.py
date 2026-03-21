#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import dpdata
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class StructureFingerprint:
    symbols: Tuple[str, ...]
    coords_bytes: bytes


def discover_system_names(data_dir: str) -> List[str]:
    p = Path(data_dir)
    if not p.exists():
        return []
    names = [x.name for x in sorted(p.iterdir()) if x.is_dir()]
    if names:
        return names
    return []


def load_system_map(data_dir: str, fmt: str) -> Dict[str, dpdata.System]:
    from dpeva.io.dataset import load_systems

    names = discover_system_names(data_dir)
    logging.getLogger("dpeva.io.dataset").setLevel(logging.ERROR)
    systems = load_systems(data_dir, fmt=fmt, target_systems=names if names else None)
    if not systems and fmt == "deepmd/npy/mixed":
        systems = load_systems(data_dir, fmt="deepmd/npy", target_systems=names if names else None)
    if not systems and fmt == "deepmd/npy":
        systems = load_systems(data_dir, fmt="deepmd/npy/mixed", target_systems=names if names else None)
    if not systems and fmt == "auto":
        systems = load_systems(data_dir, fmt="auto", target_systems=None)
    result: Dict[str, dpdata.System] = {}
    if names:
        for s in systems:
            key = getattr(s, "target_name", None) or getattr(s, "short_name", None)
            if key:
                result[key] = s
    else:
        for i, s in enumerate(systems):
            key = getattr(s, "short_name", None) or f"system_{i}"
            result[key] = s
    return result


def resolve_descriptor_path(desc_dir: str, system_name: str) -> Optional[str]:
    direct = Path(desc_dir) / f"{system_name}.npy"
    if direct.exists():
        return str(direct)
    base = Path(desc_dir) / f"{Path(system_name).name}.npy"
    if base.exists():
        return str(base)
    return None


def list_set_dirs(system_dir: str) -> List[str]:
    p = Path(system_dir)
    if not p.exists():
        return []
    return [x.name for x in sorted(p.iterdir()) if x.is_dir() and x.name.startswith("set.")]


def load_real_atom_types(system_dir: str) -> Optional[np.ndarray]:
    set_dirs = list_set_dirs(system_dir)
    if not set_dirs:
        return None
    chunks: List[np.ndarray] = []
    for s in set_dirs:
        rt = Path(system_dir) / s / "real_atom_types.npy"
        if not rt.exists():
            return None
        chunks.append(np.load(rt))
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def frame_atom_types(system: dpdata.System, frame_idx: int, real_types: Optional[np.ndarray]) -> np.ndarray:
    if real_types is not None:
        return real_types[frame_idx]
    return np.asarray(system["atom_types"])


def make_fingerprint(
    system: dpdata.System,
    frame_idx: int,
    real_types: Optional[np.ndarray] = None,
    coord_decimals: int = 4,
) -> StructureFingerprint:
    atom_names = system["atom_names"]
    atom_types = frame_atom_types(system, frame_idx, real_types)
    coords = np.asarray(system["coords"][frame_idx])
    symbols = np.array([atom_names[t] for t in atom_types], dtype="U8")
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], symbols))
    sorted_symbols = tuple(symbols[order].tolist())
    sorted_coords = np.round(coords[order], coord_decimals)
    return StructureFingerprint(sorted_symbols, sorted_coords.tobytes())


def frame_count(system: dpdata.System) -> int:
    return len(system)


def shape_is_descriptor_like(desc: np.ndarray, n_frames: int) -> bool:
    if desc.ndim < 2:
        return False
    return desc.shape[0] == n_frames


def iter_system_dirs(data_dir: str, preferred_names: Optional[Iterable[str]] = None) -> Dict[str, str]:
    base = Path(data_dir)
    if preferred_names:
        out: Dict[str, str] = {}
        for n in preferred_names:
            p = base / n
            if p.is_dir():
                out[n] = str(p)
        return out
    names = discover_system_names(data_dir)
    return {n: str(base / n) for n in names}
