import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from dpeva.inference.visualizer import InferenceVisualizer
from dpeva.io.dataset import load_systems


class DatasetAnalysisManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, dataset_dir: Path, output_dir: Path) -> Dict[str, Any]:
        systems = load_systems(str(dataset_dir), fmt="auto")
        if not systems:
            raise ValueError(f"No valid systems found in dataset_dir: {dataset_dir}")

        energy_per_atom: List[float] = []
        force_norm: List[float] = []
        virial_trace_per_atom: List[float] = []
        pressure_gpa: List[float] = []
        frame_records: List[Dict[str, Any]] = []

        for sys_idx, system in enumerate(systems):
            system_name = getattr(system, "target_name", getattr(system, "short_name", f"sys_{sys_idx}"))
            atom_count = int(len(system["atom_types"]))
            nframes = int(system.get_nframes())

            energies = np.array(system.data.get("energies", []), dtype=float).reshape(-1) if "energies" in system.data else np.array([], dtype=float)
            forces = np.array(system.data.get("forces", []), dtype=float) if "forces" in system.data else np.array([], dtype=float)
            virials = np.array(system.data.get("virials", []), dtype=float) if "virials" in system.data else np.array([], dtype=float)
            cells = np.array(system.data.get("cells", []), dtype=float) if "cells" in system.data else np.array([], dtype=float)

            for frame_idx in range(nframes):
                epa = None
                if energies.size > frame_idx:
                    epa = float(energies[frame_idx] / atom_count)
                    energy_per_atom.append(epa)

                if forces.size > 0 and frame_idx < forces.shape[0]:
                    frame_force = forces[frame_idx]
                    force_norm.extend(np.linalg.norm(frame_force, axis=1).tolist())

                vtrace = None
                pressure = None
                if virials.size > 0 and frame_idx < virials.shape[0]:
                    virial_frame = np.array(virials[frame_idx], dtype=float).reshape(3, 3)
                    vtrace = float(np.trace(virial_frame) / atom_count)
                    virial_trace_per_atom.append(vtrace)
                    if cells.size > 0 and frame_idx < cells.shape[0]:
                        cell = np.array(cells[frame_idx], dtype=float).reshape(3, 3)
                        volume = abs(np.linalg.det(cell))
                        if volume > 1e-12:
                            pressure = float(-np.trace(virial_frame) / (3.0 * volume) * 160.21766208)
                            pressure_gpa.append(pressure)

                frame_records.append(
                    {
                        "system_name": system_name,
                        "sys_idx": sys_idx,
                        "frame_idx": frame_idx,
                        "n_atoms": atom_count,
                        "energy_per_atom": epa,
                        "virial_trace_per_atom": vtrace,
                        "pressure_gpa": pressure,
                    }
                )

        viz = InferenceVisualizer(str(output_dir))
        if energy_per_atom:
            viz.plot_distribution(np.array(energy_per_atom), "Dataset Energy Per Atom", "eV/atom", color="purple")
        if force_norm:
            viz.plot_distribution(np.array(force_norm), "Dataset Force Magnitude", "eV/Å", color="orange")
        if virial_trace_per_atom:
            viz.plot_distribution(np.array(virial_trace_per_atom), "Dataset Virial Trace Per Atom", "eV/atom", color="red")
        if pressure_gpa:
            viz.plot_distribution(np.array(pressure_gpa), "Dataset Pressure", "GPa", color="teal")

        def series_stats(values: List[float]) -> Dict[str, Any]:
            if not values:
                return {"count": 0}
            return pd.Series(values).describe().to_dict()

        summary = {
            "dataset_dir": str(dataset_dir),
            "n_systems": len(systems),
            "n_frames": len(frame_records),
            "energy_per_atom": series_stats(energy_per_atom),
            "force_magnitude": series_stats(force_norm),
            "virial_trace_per_atom": series_stats(virial_trace_per_atom),
            "pressure_gpa": series_stats(pressure_gpa),
        }

        with open(output_dir / "dataset_stats.json", "w") as f:
            json.dump(summary, f, indent=4, default=self._json_default)

        pd.DataFrame(frame_records).to_csv(output_dir / "dataset_frame_summary.csv", index=False)

        return summary

    @staticmethod
    def _json_default(value: Any):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)
