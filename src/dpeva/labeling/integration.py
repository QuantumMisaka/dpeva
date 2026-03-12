import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List

import dpdata
import numpy as np

from dpeva.constants import DEFAULT_LABELING_INTEGRATION_OUTPUT_FORMAT
from dpeva.io.dataset import load_systems


logger = logging.getLogger(__name__)


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
        merged = dpdata.MultiSystems()
        existing_count = 0
        new_count = 0
        compatibility_issues = 0

        reference_atom_names = None
        if existing_training_data_path is not None and existing_training_data_path.exists():
            existing_systems = load_systems(str(existing_training_data_path), fmt="auto")
            existing_count = len(existing_systems)
            for idx, system in enumerate(existing_systems):
                reference_atom_names = self._ensure_compatible_atom_order(
                    system=system,
                    reference_atom_names=reference_atom_names,
                    source=f"existing[{idx}]",
                )
                merged.append(system)

        if not new_labeled_data_path.exists():
            raise FileNotFoundError(f"New labeled data path not found: {new_labeled_data_path}")

        new_systems = load_systems(str(new_labeled_data_path), fmt="auto")
        new_count = len(new_systems)
        for idx, system in enumerate(new_systems):
            try:
                reference_atom_names = self._ensure_compatible_atom_order(
                    system=system,
                    reference_atom_names=reference_atom_names,
                    source=f"new[{idx}]",
                )
            except ValueError:
                compatibility_issues += 1
                raise
            merged.append(system)

        before_dedup = len(merged)
        if self.deduplicate:
            merged = self._deduplicate(merged)
        after_dedup = len(merged)
        filtered_count = before_dedup - after_dedup

        merged_output_path.mkdir(parents=True, exist_ok=True)
        merged.to(self.output_format, str(merged_output_path))
        summary = {
            "existing_system_count": existing_count,
            "new_system_count": new_count,
            "merged_system_count_before_dedup": before_dedup,
            "merged_system_count_after_dedup": after_dedup,
            "filtered_system_count": filtered_count,
            "deduplicate_enabled": self.deduplicate,
            "output_format": self.output_format,
            "reference_atom_names": reference_atom_names,
            "compatibility_issues": compatibility_issues,
            "output_path": str(merged_output_path),
        }
        with open(merged_output_path / "integration_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(
            "Integration summary: existing=%s, new=%s, merged_before_dedup=%s, merged_after_dedup=%s",
            existing_count,
            new_count,
            before_dedup,
            after_dedup,
        )
        logger.info(f"Integrated dataset exported to {merged_output_path}")
        return summary

    @staticmethod
    def _ensure_compatible_atom_order(system, reference_atom_names, source: str):
        atom_names = list(system.data.get("atom_names", []))
        if not atom_names:
            raise ValueError(f"System atom_names is empty: {source}")
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

    def _deduplicate(self, systems: dpdata.MultiSystems) -> dpdata.MultiSystems:
        deduped = dpdata.MultiSystems()
        seen = set()
        for system in systems:
            coords = np.array(system.data.get("coords", []), dtype=float)
            if coords.size == 0:
                deduped.append(system)
                continue
            signature = hashlib.sha1(coords.tobytes()).hexdigest()
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(system)
        return deduped
