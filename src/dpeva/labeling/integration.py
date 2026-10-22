import ctypes
import errno
import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, List

import dpdata
import numpy as np
from dpdata.data_type import Axis

from dpeva.constants import DEFAULT_LABELING_INTEGRATION_OUTPUT_FORMAT
from dpeva.io.dataset import load_systems
from dpeva.io.lmdb_writer import write_lmdb
from dpeva.run.dataset import (
    DatasetIntersectionSummary,
    DatasetManifest,
    DatasetParent,
    DatasetValidationResult,
    validate_lineage_counts,
)


logger = logging.getLogger(__name__)


class PublicationError(RuntimeError):
    """Raised when the final bundle cannot be published safely."""


class DataIntegrationManager:
    def __init__(self, deduplicate: bool = False, output_format: str = DEFAULT_LABELING_INTEGRATION_OUTPUT_FORMAT):
        self.deduplicate = deduplicate
        self.output_format = output_format

    def integrate(
        self,
        new_labeled_data_path: Path,
        merged_output_path: Path,
        existing_training_data_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        if merged_output_path.exists():
            raise FileExistsError(
                f"Merged output path already exists; refusing overwrite: {merged_output_path}"
            )

        merged = dpdata.MultiSystems()
        input_frames = []
        existing_count = 0
        new_count = 0
        compatibility_issues = 0
        existing_frames = 0
        new_frames = 0

        reference_atom_names = None
        reference_type_map = None
        if existing_training_data_path is not None:
            if not existing_training_data_path.exists():
                raise FileNotFoundError(
                    f"Existing training data path not found: {existing_training_data_path}"
                )
            existing_systems = load_systems(str(existing_training_data_path), fmt="auto")
            existing_count = len(existing_systems)
            existing_frames = self._count_total_frames(existing_systems)
            for idx, system in enumerate(existing_systems):
                reference_atom_names = self._ensure_compatible_atom_order(
                    system=system,
                    reference_atom_names=reference_atom_names,
                    source=f"existing[{idx}]",
                )
                reference_type_map = self._ensure_compatible_type_map(
                    system=system,
                    reference_type_map=reference_type_map,
                    source=f"existing[{idx}]",
                )
                input_frames.extend(self._iter_frames(system))

        if not new_labeled_data_path.exists():
            raise FileNotFoundError(f"New labeled data path not found: {new_labeled_data_path}")

        new_systems = load_systems(str(new_labeled_data_path), fmt="auto")
        new_count = len(new_systems)
        new_frames = self._count_total_frames(new_systems)
        for idx, system in enumerate(new_systems):
            try:
                reference_atom_names = self._ensure_compatible_atom_order(
                    system=system,
                    reference_atom_names=reference_atom_names,
                    source=f"new[{idx}]",
                )
                reference_type_map = self._ensure_compatible_type_map(
                    system=system,
                    reference_type_map=reference_type_map,
                    source=f"new[{idx}]",
                )
            except ValueError:
                compatibility_issues += 1
                raise
            input_frames.extend(self._iter_frames(system))

        pre_dedup_merged = self._build_multisystems(input_frames)
        before_system_count = len(pre_dedup_merged)
        kept_frames, overlap_frame_count = self._analyze_frames(input_frames)
        if overlap_frame_count and not self.deduplicate:
            raise ValueError(
                "unexplained duplicate/intersection detected; enable integration_deduplicate "
                "to persist removal evidence"
            )
        if self.deduplicate:
            input_frames = kept_frames
        merged = self._build_multisystems(input_frames)
        after_system_count = len(merged)
        filtered_system_count = before_system_count - after_system_count
        merged_frames_before_dedup = existing_frames + new_frames
        merged_frames_after_dedup = self._count_total_frames(merged)
        filtered_frames = merged_frames_before_dedup - merged_frames_after_dedup
        if filtered_frames < 0:
            raise ValueError(
                "integration frame count conflict: merged output exceeds source frame count"
            )

        parent_specs = []
        source_entries = []
        if existing_training_data_path is not None:
            parent_specs.append(
                DatasetParent(dataset_id="existing-training", frame_count=existing_frames)
            )
            source_entries.append("existing-training")
        parent_specs.append(DatasetParent(dataset_id="new-labeled", frame_count=new_frames))
        source_entries.append("new-labeled")
        manifest = DatasetManifest(
            dataset_id=f"integration-{hashlib.sha256(str(merged_output_path).encode()).hexdigest()[:12]}",
            parents=parent_specs,
            transformation="merge",
            frame_count=merged_frames_after_dedup,
            removed_frame_count=filtered_frames,
            system_count=after_system_count,
            type_map=list(reference_type_map or reference_atom_names or []),
            format=self.output_format,
            source_entries=source_entries,
            intersection_summary=DatasetIntersectionSummary(
                method=(
                    "frame-identity-v1"
                    if self.deduplicate and (overlap_frame_count or filtered_frames)
                    else "not-run"
                ),
                overlap_frame_count=overlap_frame_count,
                removed_frame_count=filtered_frames,
                evidence_ref=(
                    "in-memory:frame-identity-v1"
                    if self.deduplicate and (overlap_frame_count or filtered_frames)
                    else None
                ),
            ),
            validation_result=DatasetValidationResult(
                status="passed",
                counts_reconciled=True,
                sources_declared=True,
                intersections_explained=True,
                type_map_compatible=True,
            ),
        )
        validate_lineage_counts(manifest)

        merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{merged_output_path.name}.staging-",
                dir=merged_output_path.parent,
            )
        )
        try:
            if self.output_format == "deepmd/lmdb":
                write_lmdb(
                    list(merged),
                    staging_path / "dataset.lmdb",
                    type_map=list(reference_type_map or reference_atom_names or []),
                )
            else:
                merged.to(self.output_format, str(staging_path))
            content_identity = self._hash_exported_dataset(staging_path)
            manifest = manifest.model_copy(
                update={
                    "content_identity": f"sha256:{content_identity}",
                    "content_identity_strength": "exported-files-sha256",
                }
            )
            validate_lineage_counts(manifest)

            manifest_payload = manifest.model_dump(mode="json")
            manifest_bytes = self._json_bytes(manifest_payload)
            generation = hashlib.sha256(manifest_bytes).hexdigest()[:16]
            immutable_manifest_name = f"dataset-manifest-{generation}.json"
            self._write_immutable_json(staging_path / immutable_manifest_name, manifest_bytes)
            self._write_bytes_atomic(staging_path / "dataset-manifest.json", manifest_bytes)

            summary = {
                "existing_system_count": existing_count,
                "new_system_count": new_count,
                "merged_system_count_before_dedup": before_system_count,
                "merged_system_count_after_dedup": after_system_count,
                "filtered_system_count": filtered_system_count,
                "existing_frame_count": existing_frames,
                "new_frame_count": new_frames,
                "merged_frame_count_before_dedup": merged_frames_before_dedup,
                "merged_frame_count_after_dedup": merged_frames_after_dedup,
                "filtered_frame_count": filtered_frames,
                "deduplicate_enabled": self.deduplicate,
                "output_format": self.output_format,
                "reference_atom_names": reference_atom_names,
                "reference_type_map": reference_type_map,
                "compatibility_issues": compatibility_issues,
                "output_path": str(merged_output_path),
                "dataset_manifest_path": immutable_manifest_name,
                "dataset_manifest_generation": generation,
                "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }
            self._write_json_atomic(staging_path / "integration_summary.json", summary)
            self._rename_noreplace(staging_path, merged_output_path)
            staging_path = None
        except Exception:
            if staging_path is not None and staging_path.exists():
                shutil.rmtree(staging_path)
            raise
        logger.info(
            "Integration summary: existing=%s, new=%s, merged_before_de-dup=%s, merged_after_de-dup=%s",
            existing_count,
            new_count,
            before_system_count,
            after_system_count,
        )
        logger.info(
            "Integration frame summary: existing=%s, new=%s, merged_before_de-dup=%s, merged_after_de-dup=%s",
            existing_frames,
            new_frames,
            merged_frames_before_dedup,
            merged_frames_after_dedup,
        )
        logger.info(f"Integrated dataset exported to {merged_output_path}")
        return summary

    @staticmethod
    def _ensure_compatible_atom_order(system, reference_atom_names, source: str):
        atom_names = list(system.data.get("atom_names", []))
        if not atom_names:
            raise ValueError(f"System atom_names is empty: {source}")
        raw_type_map = system.data.get("type_map")
        if raw_type_map is not None and list(raw_type_map) != atom_names:
            raise ValueError(
                f"Incompatible type_map ordering at {source}: {list(raw_type_map)} != {atom_names}"
            )
        if reference_atom_names is None:
            return atom_names
        if set(atom_names) != set(reference_atom_names):
            raise ValueError(
                f"Incompatible atom_names at {source}: {atom_names} != {reference_atom_names}"
            )
        if atom_names != reference_atom_names:
            DataIntegrationManager._reorder_system_to_reference(system, atom_names, reference_atom_names, source)
        return reference_atom_names

    @staticmethod
    def _ensure_compatible_type_map(system, reference_type_map, source: str):
        """Validate the type map used by a system and return the canonical map."""
        atom_names = list(system.data.get("atom_names", []))
        raw_type_map = system.data.get("type_map")
        if raw_type_map is None:
            candidate = atom_names
        else:
            candidate = list(raw_type_map)
            if not candidate or set(candidate) != set(atom_names):
                raise ValueError(
                    f"Incompatible type_map at {source}: {candidate} != {atom_names}"
                )
            if candidate != atom_names:
                raise ValueError(
                    f"Incompatible type_map ordering at {source}: {candidate} != {atom_names}"
                )

        if reference_type_map is None:
            return candidate
        if candidate != reference_type_map:
            raise ValueError(
                f"Incompatible type_map at {source}: {candidate} != {reference_type_map}"
            )
        return reference_type_map

    @staticmethod
    def _json_bytes(payload: dict[str, object]) -> bytes:
        return (json.dumps(payload, indent=4, sort_keys=True) + "\n").encode("utf-8")

    @classmethod
    def _write_immutable_json(cls, path: Path, payload: bytes) -> None:
        """Publish a generation artifact without replacing prior evidence."""
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(f"immutable manifest already exists with different content: {path.name}")
            return
        cls._write_bytes_atomic(path, payload)

    @staticmethod
    def _write_bytes_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _hash_exported_dataset(output_path: Path) -> str:
        """Hash exported data files, excluding JSON evidence and temp files."""
        digest = hashlib.sha256()
        files = sorted(
            path
            for path in output_path.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() != ".json"
        )
        for path in files:
            digest.update(path.relative_to(output_path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _rename_noreplace(source: Path, target: Path) -> None:
        """Atomically rename a directory without replacing a competitor."""
        if os.name != "posix" or not hasattr(ctypes.CDLL(None), "renameat2"):
            raise PublicationError(
                "safe atomic publication requires Linux renameat2(RENAME_NOREPLACE)"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                f"Merged output path appeared during publication: {target}"
            )
        if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise PublicationError(
                "safe atomic publication requires Linux "
                "renameat2(RENAME_NOREPLACE)"
            )
        raise OSError(error, os.strerror(error), str(target))

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        """Publish a JSON artifact with a same-directory atomic replacement."""
        DataIntegrationManager._write_bytes_atomic(
            path, DataIntegrationManager._json_bytes(payload)
        )

    @staticmethod
    def _reorder_system_to_reference(system, atom_names: List[str], reference_atom_names: List[str], source: str):
        order_map = {name: idx for idx, name in enumerate(atom_names)}
        reorder_indices = [order_map[name] for name in reference_atom_names]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(reorder_indices)}
        data = system.data
        data["atom_names"] = list(reference_atom_names)
        if "type_map" in data and len(data["type_map"]) == len(atom_names):
            data["type_map"] = [data["type_map"][i] for i in reorder_indices]
        if "atom_numbs" in data and len(data["atom_numbs"]) == len(atom_names):
            data["atom_numbs"] = [data["atom_numbs"][i] for i in reorder_indices]
        if "atom_types" in data:
            atom_types_array = np.asarray(data["atom_types"], dtype=int)
            unique_old = set(np.unique(atom_types_array).tolist())
            unknown_indices = unique_old.difference(old_to_new.keys())
            if unknown_indices:
                raise ValueError(f"Incompatible atom_types at {source}: {sorted(unknown_indices)}")
            remapped = np.vectorize(old_to_new.get)(atom_types_array)
            if isinstance(data["atom_types"], np.ndarray):
                data["atom_types"] = remapped.astype(data["atom_types"].dtype, copy=False)
            else:
                data["atom_types"] = remapped.tolist()

    @classmethod
    def _iter_frames(cls, system):
        """Yield one-frame dpdata systems before MultiSystems can coalesce them."""
        nframes = cls._count_frames(system)
        if nframes <= 1:
            yield system
            return
        sub_system = getattr(system, "sub_system", None)
        if not callable(sub_system):
            raise ValueError("multi-frame input does not expose dpdata System.sub_system")
        for frame_index in range(nframes):
            yield sub_system(frame_index)

    @staticmethod
    def _build_multisystems(frames):
        """Build a real pre/post merge so system counts reflect dpdata coalescing."""
        merged = dpdata.MultiSystems()
        for frame in frames:
            merged.append(frame)
        return merged

    @classmethod
    def _analyze_frames(cls, systems):
        """Return unique frames and reject structural duplicates with label conflicts."""
        seen: dict[str, str] = {}
        kept = []
        overlap = 0
        for frame in systems:
            if np.asarray(frame.data.get("coords", [])).size == 0:
                continue
            identity = cls._frame_identity(frame)
            labels = cls._frame_label_identity(frame)
            previous_labels = seen.get(identity)
            if previous_labels is None:
                seen[identity] = labels
                kept.append(frame)
                continue
            overlap += 1
            if previous_labels != labels:
                raise ValueError(
                    "conflicting labels for duplicate frame identity; refusing to deduplicate"
                )
        return kept, overlap

    @classmethod
    def _frame_identity(cls, frame) -> str:
        """Hash frame structure independent of atom storage/type-map order."""
        data = frame.data
        canonical = cls._canonical_frame_arrays(frame)
        payload = {
            "elements": canonical["elements"],
            "real_elements": canonical["real_elements"],
            "coords": cls._array_identity(canonical["coords"]),
            "cells": cls._array_identity(data.get("cells", [])),
            "pbc": not bool(getattr(frame, "nopbc", data.get("nopbc", False))),
        }
        return hashlib.sha256(cls._canonical_json(payload)).hexdigest()

    @classmethod
    def _frame_label_identity(cls, frame) -> str:
        """Hash labels after applying the frame's canonical atom permutation."""
        structural = {
            "atom_names", "type_map", "atom_numbs", "atom_types", "orig", "cells", "coords",
            "nopbc", "real_atom_names", "real_atom_types",
        }
        canonical = cls._canonical_frame_arrays(frame)
        labels = {
            key: cls._array_identity(
                cls._canonical_label_array(frame, key, value, canonical["permutation"])
            )
            for key, value in sorted(frame.data.items())
            if key not in structural
        }
        return hashlib.sha256(cls._canonical_json(labels)).hexdigest()

    @classmethod
    def _canonical_frame_arrays(cls, frame) -> Dict[str, object]:
        """Return atom-name and coordinate arrays in a stable element/geometry order."""
        data = frame.data
        atom_names = list(data.get("atom_names", []))
        atom_types = np.asarray(data.get("atom_types", []), dtype=int).reshape(-1)
        coords = np.asarray(data.get("coords", []), dtype=float)
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords[0]
        if coords.ndim != 2:
            raise ValueError(f"frame coords must resolve to a 2-D array, got {coords.shape}")
        if atom_types.size != coords.shape[0]:
            raise ValueError(
                f"frame atom_types/coords mismatch: {atom_types.size} vs {coords.shape[0]}"
            )
        elements = [
            atom_names[int(type_index)]
            if 0 <= int(type_index) < len(atom_names)
            else f"Type_{int(type_index)}"
            for type_index in atom_types
        ]
        permutation = np.asarray(
            sorted(
                range(len(elements)),
                key=lambda index: (elements[index], tuple(coords[index]), index),
            ),
            dtype=int,
        )
        canonical_elements = [elements[index] for index in permutation]

        real_types = data.get("real_atom_types")
        real_names = list(data.get("real_atom_names", []))
        if real_types is None:
            real_elements = canonical_elements
        else:
            real_array = np.asarray(real_types, dtype=int)
            if real_array.ndim == 2 and real_array.shape[0] == 1:
                real_array = real_array[0]
            if real_array.ndim != 1 or real_array.size != len(elements):
                raise ValueError(
                    "real_atom_types must resolve to one integer per atom"
                )
            real_array = real_array[permutation]
            real_elements = [
                real_names[int(type_index)]
                if 0 <= int(type_index) < len(real_names)
                else f"RealType_{int(type_index)}"
                for type_index in real_array
            ]

        return {
            "elements": canonical_elements,
            "real_elements": real_elements,
            "coords": coords[permutation],
            "permutation": permutation,
        }

    @classmethod
    def _canonical_label_array(cls, frame, key: str, value, permutation):
        """Apply the canonical atom permutation to one label array when applicable."""
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 0 or array.dtype.kind in {"O", "U", "S"}:
            return value
        n_atoms = len(permutation)
        atom_axis = None
        for data_type in getattr(type(frame), "DTYPES", ()):
            if data_type.name == key and data_type.shape is not None:
                if Axis.NATOMS in data_type.shape:
                    atom_axis = data_type.shape.index(Axis.NATOMS)
                break
        if atom_axis is None:
            candidate_axes = [
                axis for axis, size in enumerate(array.shape) if size == n_atoms
            ]
            if len(candidate_axes) == 1:
                atom_axis = candidate_axes[0]
        if atom_axis is not None and array.shape[atom_axis] == n_atoms:
            return np.take(array, permutation, axis=atom_axis)
        return value

    @staticmethod
    def _canonical_json(payload) -> bytes:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _array_identity(value, *, integer: bool = False):
        if value is None:
            return None
        array = np.asarray(value)
        if integer:
            array = np.asarray(array, dtype=np.int64)
        elif np.issubdtype(array.dtype, np.floating):
            array = np.asarray(array, dtype=np.float64)
        elif np.issubdtype(array.dtype, np.integer):
            array = np.asarray(array, dtype=np.int64)
        if array.dtype.kind in {"O", "U", "S"}:
            return {"shape": list(array.shape), "values": array.tolist()}
        contiguous = np.ascontiguousarray(array)
        return {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }

    @staticmethod
    def _count_frames(system) -> int:
        get_nframes = getattr(system, "get_nframes", None)
        if callable(get_nframes):
            return int(get_nframes())
        coords = np.array(system.data.get("coords", []), dtype=float)
        if coords.ndim >= 1:
            return int(coords.shape[0])
        return 0

    @classmethod
    def _count_total_frames(cls, systems) -> int:
        return sum(cls._count_frames(system) for system in systems)
