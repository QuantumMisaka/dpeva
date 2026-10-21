"""Load DeepMD LMDB frames grouped by their source ``frame_system_ids``."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import List

import numpy as np

from dpeva.io.dataset import DatasetLoadError, is_lmdb_path
from dpeva.io.lmdb_index import read_lmdb_frame_index


def load_lmdb_systems_by_id(path) -> List:
    """Load one ``dpdata.System`` per LMDB source-system id.

    ``dpdata.MultiSystems.from_file`` groups LMDB frames by composition. That is
    the right behavior for generic readers, but it destroys the original
    source-system partition needed by ``dpeva clean``. This loader uses the
    ``frame_system_ids`` metadata and the pinned dpdata 1.1 LMDB codec to keep
    both the partition and frame order.
    """

    if not is_lmdb_path(path):
        raise DatasetLoadError(f"Not a DeepMD LMDB dataset: {path!r}")
    try:
        from dpdata.formats.deepmd.lmdb import LMDBFormat
    except ImportError as exc:  # pragma: no cover - dependency contract guard
        raise DatasetLoadError(
            "Loading LMDB source systems requires dpdata>=1.1"
        ) from exc
    try:
        from dpdata.system import LabeledSystem, System
    except ImportError as exc:  # pragma: no cover - dependency contract guard
        raise DatasetLoadError("Loading LMDB source systems requires dpdata") from exc

    index = read_lmdb_frame_index(path)
    if not index.has_system_identity:
        raise DatasetLoadError(
            f"LMDB {path!r} has no frame_system_ids/system_info metadata; "
            "refusing to guess the original system partition for clean"
        )

    formatter = LMDBFormat()
    try:
        frames, metadata = formatter._read_all_frames(
            os.fspath(path), max_frames=None
        )
        raw_type_map = metadata.get("type_map")
        if raw_type_map:
            names = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in raw_type_map
            ]
            formatter._validate_frame_types(frames, names, metadata)
        else:
            names = formatter._infer_type_names(frames)
            formatter._validate_frame_types(frames, names, metadata)
        formatter._normalize_reference_fields(frames)
        specs = formatter._resolve_read_specs(frames, metadata)
        renamed = formatter._rename_frame_fields(frames, specs)
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to read LMDB source systems at {path!r}: {exc}"
        ) from exc

    if len(renamed) != len(index.frames):
        raise DatasetLoadError(
            f"LMDB {path!r} exposed {len(renamed)} frames but index has "
            f"{len(index.frames)}"
        )

    grouped: dict[int, list[int]] = defaultdict(list)
    for frame in index.frames:
        grouped[frame.system_id].append(frame.storage_index)

    dp_specs = {spec.dp_name: spec for spec in specs.values()}
    datasets = []
    system_ids = sorted(grouped)
    base_name = os.path.basename(os.path.normpath(os.fspath(path)))
    for system_id in system_ids:
        storage_indices = grouped[system_id]
        canonical_frames = [
            formatter._canonicalize_frame(renamed[index_value], dp_specs)
            for index_value in storage_indices
        ]
        formatter._validate_group_schema(
            canonical_frames, list(range(len(canonical_frames)))
        )
        composition = tuple(
            np.bincount(
                canonical_frames[0]["atom_types"], minlength=len(names)
            ).tolist()
        )
        present = [index_value for index_value, count in enumerate(composition) if count]
        atom_names = [names[index_value] for index_value in present]
        remap = np.full(len(names), -1, dtype=int)
        remap[present] = np.arange(len(present))
        atom_types = remap[canonical_frames[0]["atom_types"]]
        data = {
            "atom_names": atom_names,
            "atom_numbs": np.bincount(
                atom_types, minlength=len(atom_names)
            ).tolist(),
            "atom_types": atom_types,
            "orig": np.zeros(3),
        }
        formatter._aggregate_group(
            data,
            canonical_frames,
            list(range(len(canonical_frames))),
            dp_specs,
        )
        datasets.append((system_id, data))

    formatter._register_specs([data for _, data in datasets], dp_specs)
    systems = []
    for system_id, data in datasets:
        system_cls = LabeledSystem if "energies" in data else System
        system = system_cls(data=data)
        system.target_name = f"{base_name}#sys{system_id}"
        systems.append(system)
    return systems
