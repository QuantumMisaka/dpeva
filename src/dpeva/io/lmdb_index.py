"""Read-only frame index for DeepMD LMDB detail attribution."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import lmdb
import msgpack

from dpeva.io.dataset import DatasetLoadError, is_lmdb_path

DEFAULT_FRAME_FMT = "012d"


@dataclass(frozen=True)
class LmdbFrameRef:
    """One LMDB frame and its stable identity in the source dataset."""

    storage_index: int
    system_id: int
    system_frame_index: int
    nloc: int
    dataname: str


@dataclass(frozen=True)
class LmdbFrameIndex:
    """Frame identity and ``dp test`` ordering for one LMDB dataset."""

    path: str
    frames: tuple[LmdbFrameRef, ...]
    test_order: tuple[LmdbFrameRef, ...]
    nloc_groups: Mapping[int, tuple[LmdbFrameRef, ...]]

    @property
    def dataname_list(self) -> list[list[object]]:
        """Return ``[dataname, frame_index, nloc]`` in ``dp test -d`` order."""

        return [
            [frame.dataname, frame.system_frame_index, frame.nloc]
            for frame in self.test_order
        ]

    @property
    def datanames_nframe(self) -> dict[str, int]:
        """Return frame counts keyed by the synthesized system dataname."""

        counts: dict[str, int] = {}
        for frame in self.test_order:
            counts[frame.dataname] = counts.get(frame.dataname, 0) + 1
        return counts


def _read_metadata(env: lmdb.Environment) -> dict:
    with env.begin() as txn:
        raw = txn.get(b"__metadata__")
    if raw is None:
        raise DatasetLoadError("LMDB file is missing __metadata__")
    try:
        metadata = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception as exc:
        raise DatasetLoadError(f"Cannot decode LMDB __metadata__: {exc}") from exc
    if not isinstance(metadata, dict):
        raise DatasetLoadError("LMDB __metadata__ must decode to a mapping")
    return metadata


def _coerce_sequence(value, field: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise DatasetLoadError(f"LMDB metadata field {field!r} must be a sequence")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise DatasetLoadError(
            f"LMDB metadata field {field!r} must contain integers"
        ) from exc


def _natoms_from_system_info(info: dict, index: int) -> int:
    raw_natoms = info.get("natoms")
    if isinstance(raw_natoms, dict):
        raise DatasetLoadError(
            f"LMDB system_info[{index}].natoms is an encoded array; "
            "cannot derive a stable nloc"
        )
    if not isinstance(raw_natoms, (list, tuple)):
        raise DatasetLoadError(
            f"LMDB system_info[{index}].natoms must be a sequence"
        )
    try:
        nloc = sum(int(item) for item in raw_natoms)
    except (TypeError, ValueError) as exc:
        raise DatasetLoadError(
            f"LMDB system_info[{index}].natoms must contain integers"
        ) from exc
    if nloc <= 0:
        raise DatasetLoadError(f"LMDB system_info[{index}] has non-positive nloc")
    return nloc


def _frames_from_system_info(
    system_info,
    nframes: int,
) -> tuple[list[int], list[int]]:
    if not isinstance(system_info, list):
        system_info = [system_info] if isinstance(system_info, dict) else []
    nlocs: list[int] = []
    system_ids: list[int] = []
    for index, info in enumerate(system_info):
        if not isinstance(info, dict):
            raise DatasetLoadError(f"LMDB system_info[{index}] must be a mapping")
        try:
            system_nframes = int(info["nframes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetLoadError(
                f"LMDB system_info[{index}] is missing a valid nframes"
            ) from exc
        if system_nframes < 0:
            raise DatasetLoadError(
                f"LMDB system_info[{index}] has negative nframes"
            )
        nloc = _natoms_from_system_info(info, index)
        nlocs.extend([nloc] * system_nframes)
        system_ids.extend([index] * system_nframes)
    if len(nlocs) != nframes:
        raise DatasetLoadError(
            f"LMDB system_info describes {len(nlocs)} frames, but metadata "
            f"declares {nframes}"
        )
    return nlocs, system_ids


def _scan_frame_nlocs(
    env: lmdb.Environment, nframes: int, frame_fmt: str
) -> list[int]:
    nlocs: list[int] = []
    with env.begin() as txn:
        for index in range(nframes):
            key = format(index, frame_fmt).encode("ascii")
            raw = txn.get(key)
            if raw is None:
                raise DatasetLoadError(
                    f"LMDB frame {index} is missing while scanning nloc metadata"
                )
            try:
                frame = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            except Exception as exc:
                raise DatasetLoadError(
                    f"Cannot decode LMDB frame {index} while scanning nloc metadata: {exc}"
                ) from exc
            atom_types = frame.get("atom_types") if isinstance(frame, dict) else None
            if isinstance(atom_types, dict):
                shape = atom_types.get("shape") or atom_types.get(b"shape")
                if not shape:
                    raise DatasetLoadError(
                        f"LMDB frame {index} atom_types has no shape"
                    )
                nlocs.append(int(shape[0]))
            elif isinstance(atom_types, (list, tuple)):
                nlocs.append(len(atom_types))
            else:
                raise DatasetLoadError(
                    f"LMDB frame {index} has no decodable atom_types"
                )
    return nlocs


def read_lmdb_frame_index(path) -> LmdbFrameIndex:
    """Read LMDB frame identities without loading coordinates or labels.

    The index prefers ``frame_nlocs`` / ``frame_system_ids`` metadata written by
    current dpdata writers. Older files fall back to ``system_info`` and, only
    when neither exists, a frame scan.
    """

    if not is_lmdb_path(path):
        raise DatasetLoadError(f"Not a DeepMD LMDB dataset: {path!r}")

    resolved_path = os.fspath(path)
    try:
        env = lmdb.open(resolved_path, readonly=True, lock=False, subdir=True)
    except lmdb.Error as exc:
        raise DatasetLoadError(
            f"Cannot open LMDB dataset {resolved_path!r}: {exc}"
        ) from exc
    try:
        metadata = _read_metadata(env)
        try:
            nframes = int(metadata["nframes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetLoadError("LMDB metadata has no valid nframes") from exc
        if nframes <= 0:
            raise DatasetLoadError("LMDB metadata must declare at least one frame")
        frame_fmt = str(metadata.get("frame_idx_fmt", DEFAULT_FRAME_FMT))

        frame_nlocs = _coerce_sequence(metadata.get("frame_nlocs"), "frame_nlocs")
        frame_system_ids = _coerce_sequence(
            metadata.get("frame_system_ids"), "frame_system_ids"
        )
        system_nlocs: list[int] | None = None
        system_ids: list[int] | None = None
        system_info = metadata.get("system_info")
        if system_info:
            system_nlocs, system_ids = _frames_from_system_info(system_info, nframes)

        if frame_nlocs is None:
            if system_nlocs is not None:
                frame_nlocs = system_nlocs
            else:
                frame_nlocs = _scan_frame_nlocs(env, nframes, frame_fmt)
        if len(frame_nlocs) != nframes:
            raise DatasetLoadError(
                f"LMDB frame_nlocs has {len(frame_nlocs)} entries, "
                f"but metadata declares {nframes}"
            )
        if any(nloc <= 0 for nloc in frame_nlocs):
            raise DatasetLoadError("LMDB frame_nlocs must all be positive")

        if system_nlocs is not None and frame_nlocs != system_nlocs:
            raise DatasetLoadError(
                "LMDB frame_nlocs conflicts with system_info-derived nloc values"
            )

        if frame_system_ids is None:
            if system_ids is not None:
                frame_system_ids = system_ids
            else:
                frame_system_ids = [0] * nframes
        if len(frame_system_ids) != nframes:
            raise DatasetLoadError(
                f"LMDB frame_system_ids has {len(frame_system_ids)} entries, "
                f"but metadata declares {nframes}"
            )
        if any(system_id < 0 for system_id in frame_system_ids):
            raise DatasetLoadError("LMDB frame_system_ids must be non-negative")
        if system_ids is not None and frame_system_ids != system_ids:
            raise DatasetLoadError(
                "LMDB frame_system_ids conflicts with system_info-derived identities"
            )
    finally:
        env.close()

    nloc_by_system: dict[int, int] = {}
    for system_id, nloc in zip(frame_system_ids, frame_nlocs):
        prior = nloc_by_system.setdefault(system_id, nloc)
        if prior != nloc:
            raise DatasetLoadError(
                f"LMDB system_id {system_id} has multiple nloc values "
                f"({prior}, {nloc}); dp test detail rows are not contiguous"
            )

    base = os.path.basename(os.path.normpath(resolved_path))
    system_frame_offsets: dict[int, int] = defaultdict(int)
    frames: list[LmdbFrameRef] = []
    for storage_index, (system_id, nloc) in enumerate(
        zip(frame_system_ids, frame_nlocs)
    ):
        system_frame_index = system_frame_offsets[system_id]
        system_frame_offsets[system_id] += 1
        frames.append(
            LmdbFrameRef(
                storage_index=storage_index,
                system_id=system_id,
                system_frame_index=system_frame_index,
                nloc=nloc,
                dataname=f"{base}#sys{system_id}",
            )
        )

    test_order = tuple(sorted(frames, key=lambda frame: (frame.nloc, frame.storage_index)))
    grouped: dict[int, list[LmdbFrameRef]] = defaultdict(list)
    for frame in test_order:
        grouped[frame.nloc].append(frame)
    nloc_groups = {nloc: tuple(grouped[nloc]) for nloc in sorted(grouped)}
    return LmdbFrameIndex(
        path=resolved_path,
        frames=tuple(frames),
        test_order=test_order,
        nloc_groups=nloc_groups,
    )
