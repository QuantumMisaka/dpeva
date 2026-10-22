"""Explicit DeepMD LMDB writer with post-write identity checks."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import dpdata
import numpy as np

from dpeva.io.dataset import DatasetLoadError
from dpeva.io.lmdb_index import read_lmdb_frame_index


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _discard(path: Path) -> None:
    """Remove a staging/backup artifact without masking the write outcome."""

    try:
        _remove_path(path)
    except OSError:
        pass


def _inject_real_atom_names(system, type_map: list[str] | None) -> str | None:
    """Temporarily supply the names required by dpdata's mixed-type writer."""

    data = system.data
    if data.get("real_atom_types") is None or data.get("real_atom_names") is not None:
        return None
    names = type_map or data.get("type_map") or data.get("atom_names")
    if not names:
        raise DatasetLoadError(
            "LMDB mixed-type source has real_atom_types but no real_atom_names "
            "and no usable type_map/atom_names"
        )
    data["real_atom_names"] = list(names)
    return "real_atom_names"


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

    expected_nlocs: list[int] = []
    injected_names: list[tuple[object, str | None]] = []
    try:
        for index, system in enumerate(ordered_systems):
            coords = system.data.get("coords")
            atom_types = system.data.get("atom_types")
            if coords is None or atom_types is None:
                raise ValueError(f"system[{index}] must expose coords and atom_types")
            injected = _inject_real_atom_names(system, type_map)
            injected_names.append((system, injected))
            real_types = system.data.get("real_atom_types")
            if real_types is not None:
                real_array = np.asarray(real_types)
                if real_array.ndim == 2 and real_array.shape[0] == len(system):
                    expected_nlocs.extend(
                        int(value) for value in np.sum(real_array >= 0, axis=1)
                    )
                    continue
            expected_nlocs.extend([int(len(atom_types))] * len(system))
    except Exception:
        for system, attribute in injected_names:
            if attribute is not None:
                system.data.pop(attribute, None)
        raise

    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    backup = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        dump_systems(
            ordered_systems,
            os.fspath(staging),
            type_map=type_map,
            overwrite=True,
        )
        index = read_lmdb_frame_index(staging)
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

        if target.exists():
            backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
            os.replace(target, backup)
        os.replace(staging, target)
        staging = None
    except BaseException as exc:
        # KeyboardInterrupt/SystemExit must not leave the previous target
        # stranded under a hidden backup name.
        if staging is not None:
            _discard(staging)
        if backup is not None and not target.exists() and backup.exists():
            try:
                os.replace(backup, target)
                backup = None
            except OSError:
                pass
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, DatasetLoadError):
            raise
        raise DatasetLoadError(f"Failed to write LMDB {target!s}: {exc}") from exc
    finally:
        if staging is not None:
            _discard(staging)
        if backup is not None and target.exists():
            _discard(backup)
        for system, attribute in injected_names:
            if attribute is not None:
                system.data.pop(attribute, None)
    return {
        "path": str(target),
        "nframes": len(index.frames),
        "nsystems": len(actual_system_ids),
        "frame_system_ids": [
            frame.system_id for frame in index.frames
        ],
    }
