"""Explicit DeepMD LMDB writer with post-write identity checks."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import dpdata

from dpeva.io.dataset import DatasetLoadError
from dpeva.io.lmdb_index import read_lmdb_frame_index


def write_lmdb(
    systems: Iterable,
    output_path,
    *,
    type_map: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write one source system per input element to a DeepMD LMDB.

    This is deliberately an explicit conversion API, not a default workflow
    output format. ``dump_systems`` preserves the input system partition and
    frame order in ``frame_system_ids``; the resulting file is read back and
    checked before success is returned.
    """

    if isinstance(systems, dpdata.MultiSystems):
        ordered_systems = list(systems)
    else:
        ordered_systems = list(systems)
    if not ordered_systems:
        raise ValueError("write_lmdb requires at least one system")
    try:
        from dpdata.formats.deepmd.lmdb import dump_systems
    except ImportError as exc:  # pragma: no cover - dependency contract guard
        raise DatasetLoadError("Writing deepmd/lmdb requires dpdata>=1.1") from exc
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"LMDB output already exists: {target}")

    expected_nlocs = []
    for index, system in enumerate(ordered_systems):
        coords = system.data.get("coords")
        atom_types = system.data.get("atom_types")
        if coords is None or atom_types is None:
            raise ValueError(f"system[{index}] must expose coords and atom_types")
        expected_nlocs.append(int(len(atom_types)))

    try:
        dump_systems(
            ordered_systems,
            os.fspath(target),
            type_map=type_map,
            overwrite=overwrite,
        )
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to write LMDB {target!s}: {exc}"
        ) from exc

    index = read_lmdb_frame_index(target)
    expected_frames = sum(len(system) for system in ordered_systems)
    if len(index.frames) != expected_frames:
        raise DatasetLoadError(
            f"LMDB write validation failed: wrote {len(index.frames)} frames, "
            f"expected {expected_frames}"
        )
    expected_system_ids = list(range(len(ordered_systems)))
    actual_system_ids = sorted({frame.system_id for frame in index.frames})
    if actual_system_ids != expected_system_ids:
        raise DatasetLoadError(
            f"LMDB write validation failed: system ids {actual_system_ids}, "
            f"expected {expected_system_ids}"
        )
    expected_groups = Counter(expected_nlocs)
    actual_groups = Counter(
        {
            nloc: len(frames)
            for nloc, frames in index.nloc_groups.items()
        }
    )
    if actual_groups != expected_groups:
        raise DatasetLoadError(
            f"LMDB write validation failed: nloc groups {actual_groups}, "
            f"expected {expected_groups}"
        )
    if not index.has_system_identity:
        raise DatasetLoadError(
            "LMDB write validation failed: source-system identity metadata missing"
        )
    return {
        "path": str(target),
        "nframes": len(index.frames),
        "nsystems": len(actual_system_ids),
        "frame_system_ids": [
            frame.system_id for frame in index.frames
        ],
    }
