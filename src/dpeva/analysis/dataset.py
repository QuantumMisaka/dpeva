import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from dpeva.constants import (
    FILENAME_DATASET_ELEMENT_PRESENCE_PNG,
    FILENAME_DATASET_ELEMENT_RATIO_PNG,
    FILENAME_DATASET_FRAME_SUMMARY_CSV,
    FILENAME_DATASET_STATS_JSON,
)
from dpeva.analysis.managers import describe_analysis_plot_level
from dpeva.inference.visualizer import InferenceVisualizer
from dpeva.inference.stats import StatsCalculator
from dpeva.io.dataset import load_systems


class DatasetAnalysisManager:
    """Analyze dataset-only physical statistics and generate visualization artifacts."""

    def __init__(
        self,
        ref_energies: Dict[str, float] | None = None,
        enable_cohesive_energy: bool = True,
        allow_ref_energy_lstsq_completion: bool = False,
    ):
        """Initialize dataset analysis manager."""
        self.logger = logging.getLogger(__name__)
        self.ref_energies = ref_energies or {}
        self.enable_cohesive_energy = enable_cohesive_energy
        self.allow_ref_energy_lstsq_completion = allow_ref_energy_lstsq_completion

    def analyze(self, dataset_dir: Path, output_dir: Path, plot_level: str = "full") -> Dict[str, Any]:
        """Compute dataset statistics and export plots/summary files."""
        systems = load_systems(str(dataset_dir), fmt="auto")
        if not systems:
            raise ValueError(f"No valid systems found in dataset_dir: {dataset_dir}")
        full_plot_enabled = plot_level == "full"
        self.logger.info(
            f"Dataset analysis plot scope ({plot_level}): {describe_analysis_plot_level(plot_level, mode='dataset')}"
        )

        energy_per_atom: List[float] = []
        force_norm: List[float] = []
        virial_trace_per_atom: List[float] = []
        pressure_gpa: List[float] = []
        cohesive_energy_per_atom: List[float] = []
        frame_records: List[Dict[str, Any]] = []
        element_count_by_atom: Counter = Counter()
        system_element_presence_count: Counter = Counter()
        frame_element_presence_count: Counter = Counter()
        frame_atom_counts: List[Dict[str, int]] = []
        frame_atom_nums: List[int] = []

        for sys_idx, system in enumerate(systems):
            system_name = getattr(system, "target_name", getattr(system, "short_name", f"sys_{sys_idx}"))
            atom_count = int(len(system["atom_types"]))
            nframes = int(system.get_nframes())
            atom_names = []
            if "atom_names" in system.data:
                atom_names = list(system.data.get("atom_names", []))
            else:
                try:
                    atom_names = list(system["atom_names"])
                except Exception:
                    atom_names = []
            atom_types = list(system["atom_types"])
            elements = [atom_names[t] for t in atom_types] if atom_names else []
            element_counts_this_system = Counter(elements)
            for elem, elem_count in element_counts_this_system.items():
                element_count_by_atom[elem] += int(elem_count) * nframes
                system_element_presence_count[elem] += 1
                frame_element_presence_count[elem] += nframes

            energies = np.array(system.data.get("energies", []), dtype=float).reshape(-1) if "energies" in system.data else np.array([], dtype=float)
            forces = np.array(system.data.get("forces", []), dtype=float) if "forces" in system.data else np.array([], dtype=float)
            virials = np.array(system.data.get("virials", []), dtype=float) if "virials" in system.data else np.array([], dtype=float)
            cells = np.array(system.data.get("cells", []), dtype=float) if "cells" in system.data else np.array([], dtype=float)

            for frame_idx in range(nframes):
                epa = None
                if energies.size > frame_idx:
                    epa = float(energies[frame_idx] / atom_count)
                    energy_per_atom.append(epa)
                    frame_atom_counts.append(dict(element_counts_this_system))
                    frame_atom_nums.append(atom_count)

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

        if full_plot_enabled and self.enable_cohesive_energy and energy_per_atom and frame_atom_counts and frame_atom_nums:
            calc = StatsCalculator(
                energy_per_atom=np.array(energy_per_atom, dtype=float),
                force_flat=np.array([], dtype=float),
                atom_counts_list=frame_atom_counts,
                atom_num_list=frame_atom_nums,
                ref_energies=self.ref_energies,
                enable_cohesive_energy=True,
                allow_ref_energy_lstsq_completion=self.allow_ref_energy_lstsq_completion,
            )
            cohesive = calc.compute_relative_energy(calc.e_pred)
            if cohesive is not None:
                cohesive_energy_per_atom = np.asarray(cohesive, dtype=float).tolist()
                viz.plot_distribution(
                    np.asarray(cohesive_energy_per_atom, dtype=float),
                    "Dataset Cohesive Energy",
                    "eV/atom",
                    color="#7e22ce",
                )
            else:
                self.logger.warning("Dataset cohesive energy plotting skipped due to unavailable composition/reference conditions.")

        self._plot_element_statistics(viz, output_dir, summary_data={
            "element_ratio_by_atom": self._compute_ratio_by_atom(element_count_by_atom),
            "system_element_presence": self._compute_system_presence(system_element_presence_count, len(systems)),
            "frame_element_presence": self._compute_frame_presence(frame_element_presence_count, len(frame_records)),
            "n_frames": len(frame_records),
        })

        def series_stats(values: List[float]) -> Dict[str, Any]:
            """Return pandas describe dictionary for a numeric series."""
            if not values:
                return {"count": 0}
            return pd.Series(values).describe().to_dict()

        summary = {
            "dataset_dir": str(dataset_dir),
            "n_systems": len(systems),
            "n_frames": len(frame_records),
            "element_categories": sorted(element_count_by_atom.keys()),
            "element_count_by_atom": dict(element_count_by_atom),
            "element_ratio_by_atom": self._compute_ratio_by_atom(element_count_by_atom),
            "system_element_presence": self._compute_system_presence(system_element_presence_count, len(systems)),
            "frame_element_presence": self._compute_frame_presence(frame_element_presence_count, len(frame_records)),
            "energy_per_atom": series_stats(energy_per_atom),
            "force_magnitude": series_stats(force_norm),
            "virial_trace_per_atom": series_stats(virial_trace_per_atom),
            "pressure_gpa": series_stats(pressure_gpa),
        }
        if cohesive_energy_per_atom:
            summary["cohesive_energy_per_atom"] = series_stats(cohesive_energy_per_atom)

        with open(output_dir / FILENAME_DATASET_STATS_JSON, "w") as f:
            json.dump(summary, f, indent=4, default=self._json_default)

        pd.DataFrame(frame_records).to_csv(output_dir / FILENAME_DATASET_FRAME_SUMMARY_CSV, index=False)

        return summary

    def _plot_element_statistics(self, viz: InferenceVisualizer, output_dir: Path, summary_data: Dict[str, Any]):
        """Plot element-ratio pie and per-element frame-presence mini pies."""
        dpi = getattr(viz, "dpi", 150)
        if not isinstance(dpi, (int, float)):
            dpi = 150
        ratios = summary_data.get("element_ratio_by_atom", {}) or {}
        frame_presence = summary_data.get("frame_element_presence", {}) or {}
        n_frames = int(summary_data.get("n_frames", 0))
        if ratios:
            elems = list(ratios.keys())
            vals = [ratios[e] for e in elems]
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            colors = sns.color_palette("Set2", n_colors=len(elems))
            
            n_elems = len(elems)
            if n_elems <= 3:
                pct_fontsize = 14
                title_fontsize = 18
            elif n_elems <= 6:
                pct_fontsize = 12
                title_fontsize = 16
            else:
                pct_fontsize = 10
                title_fontsize = 14

            wedges, _, autotexts = ax.pie(
                vals,
                colors=colors,
                autopct="%1.1f%%",
                startangle=120,
                wedgeprops={"edgecolor": "white", "linewidth": 1.0},
                textprops={"fontsize": pct_fontsize},
            )
            for wedge, elem, value, pct_text in zip(wedges, elems, vals, autotexts):
                pct_text.set_text(f"{elem}\n{value * 100:.1f}%")
                pct_text.set_fontsize(pct_fontsize + 1)
                pct_text.set_weight("semibold")
            ax.set_title("Dataset Element Ratio by Atom", fontsize=title_fontsize, weight="bold")
            ax.axis("equal")
            fig.tight_layout()
            fig.savefig(output_dir / FILENAME_DATASET_ELEMENT_RATIO_PNG, dpi=dpi)
            plt.close(fig)

        if frame_presence and n_frames > 0:
            elems = list(frame_presence.keys())
            cols = 2
            rows = int(np.ceil(len(elems) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(8.5, 3.8 * rows))
            axes_arr = np.atleast_1d(axes).flatten()
            colors = sns.color_palette("tab10", n_colors=max(len(elems), 3))
            
            n_elems_pres = len(elems)
            if n_elems_pres <= 4:
                center_fontsize = 13
                subtitle_fontsize = 14
                suptitle_fontsize = 18
            elif n_elems_pres <= 8:
                center_fontsize = 11
                subtitle_fontsize = 12
                suptitle_fontsize = 16
            else:
                center_fontsize = 10
                subtitle_fontsize = 11
                suptitle_fontsize = 14

            for idx, elem in enumerate(elems):
                ax = axes_arr[idx]
                present_count = int(frame_presence[elem]["frame_count"])
                present_ratio = float(frame_presence[elem]["frame_ratio"])
                absent_count = max(n_frames - present_count, 0)
                ax.pie(
                    [present_count, absent_count],
                    colors=[colors[idx % len(colors)], "#e5e7eb"],
                    startangle=90,
                    counterclock=False,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.0},
                )
                ax.text(
                    0.0,
                    0.0,
                    f"{elem}\n{present_ratio * 100:.1f}%\n{present_count}/{n_frames}",
                    ha="center",
                    va="center",
                    fontsize=center_fontsize,
                    weight="semibold",
                )
                ax.set_title(f"{elem} Frame Presence", fontsize=subtitle_fontsize, weight="normal")
                ax.axis("equal")
            for j in range(len(elems), len(axes_arr)):
                axes_arr[j].axis("off")
            fig.suptitle("Dataset Element Frame Presence", fontsize=suptitle_fontsize, weight="bold", y=0.99)
            fig.tight_layout()
            fig.savefig(output_dir / FILENAME_DATASET_ELEMENT_PRESENCE_PNG, dpi=dpi)
            plt.close(fig)

    @staticmethod
    def _json_default(value: Any):
        """Convert NumPy types to JSON-serializable Python types."""
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    @staticmethod
    def _compute_ratio_by_atom(element_count_by_atom: Counter) -> Dict[str, float]:
        """Compute element ratio by total atom count."""
        total_atoms = float(sum(element_count_by_atom.values()))
        if total_atoms <= 0:
            return {}
        return {
            elem: float(count) / total_atoms
            for elem, count in sorted(element_count_by_atom.items())
        }

    @staticmethod
    def _compute_system_presence(system_element_presence_count: Counter, n_systems: int) -> Dict[str, Dict[str, float]]:
        """Compute element presence ratio across systems."""
        if n_systems <= 0:
            return {}
        return {
            elem: {
                "system_count": int(count),
                "system_ratio": float(count) / float(n_systems)
            }
            for elem, count in sorted(system_element_presence_count.items())
        }

    @staticmethod
    def _compute_frame_presence(frame_element_presence_count: Counter, n_frames: int) -> Dict[str, Dict[str, float]]:
        """Compute element presence ratio across frames."""
        if n_frames <= 0:
            return {}
        return {
            elem: {
                "frame_count": int(count),
                "frame_ratio": float(count) / float(n_frames)
            }
            for elem, count in sorted(frame_element_presence_count.items())
        }
