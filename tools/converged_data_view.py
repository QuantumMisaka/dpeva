#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converged Data View Script
==========================

This script processes, visualizes, and cleans converged calculation data (dpdata/npy/mixed format).
It uses an Object-Oriented approach to handle data flow, metric calculation, and anomaly detection.

Classes:
    - Config: Configuration parameters.
    - DPDataLoader: Loads dpdata systems and converts to DataFrame.
    - DataProcessor: Computes derived metrics (cohesive energy, pressure, etc.).
    - DataVisualizer: Generates plots.
    - DataCleaner: Identifies anomalies and filters data.
    - DataExporter: Exports data and statistics.
    - ConvergedDataViewApp: Main application controller.
    
Author:
    - Quantum Misaka via Trae SOLO
"""

import sys
import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import lstsq

import dpdata
from ase import Atoms
from ase.io import write


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """
    Configuration parameters for the data processing pipeline.
    """
    input_npy_dir: Path = Path("./converged_data/all_conv_dpdata_mixed/").resolve()
    output_dir: Path = Path("./outputs/conveged_data_view").resolve()
    data_format: str = "deepmd/npy/mixed"
    
    # Plotting
    show_figures: bool = False
    save_figures: bool = True
    matplotlib_backend: str = "Agg"
    
    # Thresholds for anomaly detection
    energy_thr: float = 0.0      # eV/atom (cohesive energy)
    force_thr: float = 40.0      # eV/Å
    stress_thr: float = 40.0    # GPa
    max_atoms_thr: int = 255     # Maximum number of atoms

    # Data Splitting
    test_ratio: float = 0.8      # Ratio of data to use for testing
    
    # Export Formats
    export_formats: List[str] = field(default_factory=lambda: [
        "extxyz",
        "deepmd/npy",
        "deepmd/npy/mixed"
        ])

    # Reference energies (E0) for cohesive energy calculation
    # If provided, these values override the Least Squares calculation
    ref_energies_override: Dict[str, float] = field(default_factory=lambda: {
        "Fe": -3215.2791,
        "C": -156.0795,
        "O": -444.6670,
        "H": -13.5410,
    })

    @property
    def anomaly_output_dir(self) -> Path:
        return self.output_dir / "anomalies"

    def setup_logging(self) -> Path:
        """Sets up logging configuration."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.output_dir / "run.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding="utf-8", mode="w"),
            ],
            force=True # Reset any existing handlers
        )
        return log_file


# =============================================================================
# Utilities
# =============================================================================

class Utils:
    """Helper static methods."""
    
    @staticmethod
    def sanitize_formula(formula: str) -> str:
        """Removes special characters from chemical formula for safe filenames."""
        return "".join(ch for ch in formula if ch.isalnum() or ch in ("+", "-", "_"))

    @staticmethod
    def build_atoms_from_dpdata(conv_data: dpdata.MultiSystems, si: int, fi: int) -> Atoms:
        """Converts a specific frame in dpdata to an ASE Atoms object."""
        return conv_data[si][fi].to_ase_structure()[0]


# =============================================================================
# Data Loading
# =============================================================================

class DPDataLoader:
    """
    Handles loading of dpdata Systems and conversion to raw DataFrame.
    """
    
    def __init__(self, config: Config):
        self.config = config

    def load_systems(self) -> dpdata.MultiSystems:
        """Loads MultiSystems from the specified directory."""
        if not self.config.input_npy_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.config.input_npy_dir}")
        
        logging.info(f"加载 {self.config.data_format} 数据: {self.config.input_npy_dir}")
        try:
            conv_data = dpdata.MultiSystems.from_file(
                str(self.config.input_npy_dir),
                fmt=self.config.data_format
            )
            logging.info(str(conv_data))
            return conv_data
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def build_initial_dataframe(self, conv_data: dpdata.MultiSystems) -> pd.DataFrame:
        """Iterates through systems and frames to build a pandas DataFrame."""
        logging.info("开始构建基础数据帧...")
        all_data = []
        
        for si, s in enumerate(conv_data):
            atom_names = s["atom_names"]
            atom_types = s["atom_types"]
            elements = [atom_names[t] for t in atom_types]
            n_frames = s.get_nframes()
            
            for fi in range(n_frames):
                # Extract raw data
                energy = float(s["energies"][fi])
                forces = s["forces"][fi]
                virial = s["virials"][fi]
                cells = s["cells"][fi]
                
                # Basic calculations
                volume = float(np.abs(np.linalg.det(cells)))
                stress_tensor = virial / volume if volume > 1e-6 else np.zeros((3, 3))
                spins = s.data["spins"][fi] if "spins" in s.data else None
                
                all_data.append({
                    "energy": energy,
                    "num_atoms": int(len(elements)),
                    "elements": list(elements),
                    "forces": np.array(forces),
                    "stress_tensor": np.array(stress_tensor),
                    "volume": volume,
                    "spins": None if spins is None else np.array(spins),
                    "sys_idx": int(si),
                    "frame_idx": int(fi),
                })
                
        df = pd.DataFrame(all_data)
        logging.info(f"数据帧构建完成，总帧数: {len(df)}")
        return df


# =============================================================================
# Data Processing
# =============================================================================

class DataProcessor:
    """
    Handles computation of derived metrics and statistics.
    """
    
    def __init__(self, config: Config):
        self.config = config

    def get_unique_elements(self, df: pd.DataFrame) -> List[str]:
        """Returns a sorted list of unique elements present in the DataFrame."""
        return sorted(list(set([e for row in df["elements"] for e in row])))

    def compute_cohesive_energy(self, df: pd.DataFrame) -> Tuple[Dict[str, float], List[str]]:
        """
        Computes cohesive energy per atom.
        Uses reference energies from config if available, otherwise fits them via Least Squares.
        """
        unique_elements = self.get_unique_elements(df)
        ref_override = self.config.ref_energies_override
        
        use_override = False
        E0_dict = {}

        # Check if we can use override values
        if isinstance(ref_override, dict):
            if all(e in ref_override for e in unique_elements) and \
               all(isinstance(ref_override[e], (int, float)) for e in unique_elements):
                E0_dict = {e: float(ref_override[e]) for e in unique_elements}
                use_override = True
                logging.info("使用外部配置的参考能量 (E0)")

        # If not, use Least Squares fitting
        if not use_override:
            logging.info("未提供完整参考能量，使用最小二乘法拟合 E0")
            element_map = {e: i for i, e in enumerate(unique_elements)}
            A = np.zeros((len(df), len(unique_elements)))
            b = df["energy"].values
            
            for idx, row in df.iterrows():
                counts = Counter(row["elements"])
                for elem, count in counts.items():
                    A[idx, element_map[elem]] = count
            
            E0_values, residuals, rank, s_vals = lstsq(A, b)
            E0_dict = {elem: float(val) for elem, val in zip(unique_elements, E0_values)}

        logging.info("最终使用的 E0 值:")
        for elem in unique_elements:
            logging.info(f"  {elem}: {E0_dict.get(elem)}")

        # Apply calculation
        df["ref_energy"] = df["elements"].apply(lambda x: sum([E0_dict.get(e, 0) for e in x]))
        df["cohesive_energy"] = df["energy"] - df["ref_energy"]
        df["cohesive_energy_per_atom"] = df["cohesive_energy"] / df["num_atoms"]
        
        return E0_dict, unique_elements

    def add_derived_metrics(self, df: pd.DataFrame):
        """Adds pressure, max force magnitude, and energy per atom to the DataFrame."""
        
        # Energy per atom
        if "energy_per_atom" not in df.columns:
            df["energy_per_atom"] = df["energy"] / df["num_atoms"]
            
        # Pressure (GPa)
        if "pressure_gpa" not in df.columns:
            pressures = []
            for stress in df["stress_tensor"]:
                # hydrostatic pressure = trace(stress) / 3
                p = float(np.trace(stress) / 3.0)
                pressures.append(p)
            # Convert to GPa (1 eV/A^3 approx 160.2 GPa)
            df["pressure_gpa"] = np.array(pressures) * 160.21766208
            
        # Max Force Magnitude
        if "max_force_magnitude" not in df.columns:
            df["max_force_magnitude"] = [
                float(np.linalg.norm(f, axis=1).max()) for f in df["forces"]
            ]
            
        return df

    def analyze_spins_stats(self, df: pd.DataFrame):
        """Logs statistics about spin data."""
        total = len(df)
        with_spins = int(df["spins"].apply(lambda x: x is not None).sum())
        logging.info(f"包含自旋的帧数: {with_spins}/{total}")
        
        if with_spins == 0:
            return

        shape_counts = Counter()
        mismatch = 0
        
        for _, row in df.iterrows():
            spins = row["spins"]
            if spins is None:
                continue
                
            elements = row["elements"]
            if hasattr(spins, "shape"):
                shape_counts[tuple(spins.shape)] += 1
            
            # Check dimensions
            if hasattr(spins, "ndim") and spins.ndim == 2 and spins.shape[1] == 3:
                mags = np.linalg.norm(spins, axis=1)
            else:
                mags = np.abs(spins)
                
            if len(elements) != len(mags):
                mismatch += 1

        logging.info(f"自旋数组形状统计: {dict(shape_counts)}")
        logging.info(f"元素与自旋长度不匹配的帧数: {mismatch}")


# =============================================================================
# Visualization
# =============================================================================

class DataVisualizer:
    """
    Handles all plotting operations.
    """
    
    def __init__(self, config: Config):
        self.config = config
        matplotlib.use(self.config.matplotlib_backend)
        sns.set_theme(style="whitegrid", context="talk")

    def _save_and_show(self, fig: plt.Figure, filename: str):
        """Helper to save and/or show figures based on config."""
        if self.config.save_figures:
            out_path = self.config.output_dir / filename
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            logging.info(f"图像已保存: {out_path}")
        
        if self.config.show_figures:
            plt.show()
        
        plt.close(fig)

    def plot_distribution(self, data: pd.Series, title: str, xlabel: str, 
                          filename: str, color: str = None):
        """Generic method to plot a histogram/KDE distribution."""
        logging.info(f"绘制分布图: {title}")
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        self._save_and_show(fig, filename)

    def plot_force_distribution(self, df: pd.DataFrame):
        """Plots distribution of all atomic force magnitudes."""
        logging.info("绘制原子受力模长分布图")
        all_force_mags = []
        for forces in df["forces"]:
            mags = np.linalg.norm(forces, axis=1)
            all_force_mags.extend(mags)
        
        self.plot_distribution(
            pd.Series(all_force_mags), 
            "Atomic Force Magnitude Distribution",
            "Force (eV/Å)",
            "force_magnitude.png",
            color="orange"
        )

    def plot_spins_kde(self, df: pd.DataFrame, unique_elements: List[str]):
        """Plots KDE of spin magnitudes by element."""
        logging.info("绘制元素自旋 KDE 图")
        spins_by_element = {e: [] for e in unique_elements}
        
        has_spins = False
        for _, row in df.iterrows():
            if row["spins"] is None:
                continue
            has_spins = True
            spins = row["spins"]
            elements = row["elements"]
            
            if hasattr(spins, "ndim") and spins.ndim == 2 and spins.shape[1] == 3:
                mags = np.linalg.norm(spins, axis=1)
            else:
                mags = np.abs(spins)
                
            for elem, mag in zip(elements, mags):
                spins_by_element[elem].append(float(mag))
        
        if not has_spins:
            logging.info("无自旋数据，跳过绘制。")
            return

        fig = plt.figure(figsize=(12, 8))
        for elem in unique_elements:
            if len(spins_by_element[elem]) > 0:
                sns.kdeplot(spins_by_element[elem], label=elem, fill=True, alpha=0.3)
        
        plt.title("Atomic Spin Magnitude Distribution by Element")
        plt.xlabel("Magnetic Moment ($\\mu_B$)")
        plt.ylabel("Density")
        plt.legend()
        self._save_and_show(fig, "spins_kde.png")

    def plot_spins_facets(self, df: pd.DataFrame, unique_elements: List[str]):
        """Plots faceted histograms for spins by element."""
        logging.info("绘制元素自旋分面图")
        # Reuse extraction logic (simplified for brevity, could be a helper method)
        spins_by_element = {e: [] for e in unique_elements}
        has_spins = False
        for _, row in df.iterrows():
            if row["spins"] is None: continue
            has_spins = True
            spins = row["spins"]
            elements = row["elements"]
            mags = np.linalg.norm(spins, axis=1) if (spins.ndim == 2 and spins.shape[1] == 3) else np.abs(spins)
            for elem, mag in zip(elements, mags):
                spins_by_element[elem].append(float(mag))
        
        if not has_spins: return

        n = len(unique_elements)
        cols = 4 if n >= 4 else n if n > 0 else 1
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
        # Ensure axes is always 2D array
        if rows == 1 and cols == 1: axes = np.array([[axes]])
        elif rows == 1 or cols == 1: axes = np.array(axes).reshape(rows, cols)
        
        for i, elem in enumerate(unique_elements):
            r, c = i // cols, i % cols
            ax = axes[r][c]
            data = spins_by_element[elem]
            if len(data) > 0:
                sns.histplot(data, bins=50, stat="density", alpha=0.3, ax=ax)
                try:
                    sns.kdeplot(data, fill=True, alpha=0.3, ax=ax)
                except Exception:
                    pass # KDE might fail for sparse data
            ax.set_title(elem)
            ax.set_xlabel("Magnetic Moment ($\\mu_B$)")
            ax.set_ylabel("Density")
            
        # Hide empty subplots
        for j in range(n, rows * cols):
            r, c = j // cols, j % cols
            axes[r][c].axis("off")
            
        plt.tight_layout()
        self._save_and_show(fig, "spins_facets.png")

    def plot_atoms_distribution(self, df: pd.DataFrame):
        """Plots distribution of number of atoms per frame."""
        logging.info("绘制原子数量分布图")
        self.plot_distribution(
            df["num_atoms"], 
            "Number of Atoms Distribution", 
            "Number of Atoms", 
            "num_atoms_distribution.png", 
            color="teal"
        )

    def plot_all(self, df: pd.DataFrame, unique_elements: List[str]):
        """Orchestrates all plotting tasks."""
        self.plot_distribution(
            df["energy_per_atom"], 
            "Absolute Energy per Atom Distribution", 
            "Energy (eV/atom)", 
            "energy_per_atom.png"
        )
        self.plot_distribution(
            df["cohesive_energy_per_atom"], 
            "Cohesive Energy per Atom Distribution", 
            "Cohesive Energy (eV/atom)", 
            "cohesive_energy_per_atom.png", 
            color="green"
        )
        self.plot_force_distribution(df)
        self.plot_distribution(
            df["pressure_gpa"], 
            "Hydrostatic Stress (Pressure) Distribution", 
            "Stress (GPa)", 
            "pressure_gpa.png", 
            color="purple"
        )
        self.plot_spins_kde(df, unique_elements)
        self.plot_spins_facets(df, unique_elements)
        self.plot_atoms_distribution(df)


# =============================================================================
# Anomaly Detection & Cleaning
# =============================================================================

class DataCleaner:
    """
    Identifies anomalies and creates clean subsets of data.
    """
    
    def __init__(self, config: Config):
        self.config = config

    def compute_anomaly_masks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates boolean masks for anomalies based on thresholds."""
        coh = df["cohesive_energy_per_atom"] > self.config.energy_thr
        force = df["max_force_magnitude"] > self.config.force_thr
        stress = np.abs(df["pressure_gpa"]) > self.config.stress_thr
        atoms_count = df["num_atoms"] > self.config.max_atoms_thr
        
        # Update DataFrame with anomaly flags
        df["is_coh_anom"] = coh
        df["is_force_anom"] = force
        df["is_stress_anom"] = stress
        df["is_atoms_anom"] = atoms_count
        
        # Bitmask: 1=Coh, 2=Force, 4=Stress, 8=Atoms
        bitmask = (coh.astype(int) * 1) + (force.astype(int) * 2) + (stress.astype(int) * 4) + (atoms_count.astype(int) * 8)
        df["anomaly_bitmask"] = bitmask
        df["is_anom_any"] = bitmask > 0
        
        counts = {
            "cohesive": int(coh.sum()),
            "force": int(force.sum()),
            "stress": int(stress.sum()),
            "atoms": int(atoms_count.sum()),
        }
        
        # Breakdown of overlaps
        breakdown_map = {
            1: "cohesive", 2: "force", 4: "stress", 8: "atoms",
            3: "cohesive+force", 5: "cohesive+stress", 6: "force+stress",
            7: "cohesive+force+stress",
            9: "cohesive+atoms", 10: "force+atoms", 12: "stress+atoms",
            11: "cohesive+force+atoms", 13: "cohesive+stress+atoms", 14: "force+stress+atoms",
            15: "cohesive+force+stress+atoms",
        }
        vc = df.loc[df["is_anom_any"], "anomaly_bitmask"].value_counts().sort_index()
        overlap_breakdown = [
            {"bitmask": int(k), "label": breakdown_map.get(int(k), str(int(k))), "count": int(v)} 
            for k, v in vc.items()
        ]
        
        union_total = int(df["is_anom_any"].sum())
        logging.info(f"异常帧计数: cohesive={counts['cohesive']} force={counts['force']} stress={counts['stress']} atoms={counts['atoms']} union_total={union_total}")
        
        return {
            "counts": counts,
            "union_total": union_total,
            "overlap_breakdown": overlap_breakdown,
        }

    def get_clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Returns the cleaned DataFrame and the number of removed frames."""
        mask_bad = df["is_anom_any"]
        cleaned = df[~mask_bad].copy()
        return cleaned, int(mask_bad.sum())


# =============================================================================
# Data Export
# =============================================================================

class DataExporter:
    """
    Handles file export operations (CSV, JSON, extxyz, deepmd).
    """
    
    def __init__(self, config: Config):
        self.config = config

    def export_statistics(self, df: pd.DataFrame, output_dir: Path, suffix: str = ""):
        """Calculates and saves statistical summaries."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base Stats
        base_cols = ["num_atoms", "energy_per_atom", "cohesive_energy_per_atom", 
                     "pressure_gpa", "max_force_magnitude"]
        base_cols = [c for c in base_cols if c in df.columns]
        
        desc = df[base_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
        desc.to_csv(output_dir / f"summary_describe_base{suffix}.csv")
        desc.to_json(output_dir / f"summary_describe_base{suffix}.json", orient="split", indent=2)
        logging.info(f"统计摘要已保存至 {output_dir}")

    def _export_extxyz_batch(self, conv_data: dpdata.MultiSystems, df_subset: pd.DataFrame, 
                             out_dir: Path, name_fn):
        """Helper to export a batch of extxyz files."""
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for idx, row in df_subset.iterrows():
            si, fi = int(row["sys_idx"]), int(row["frame_idx"])
            atoms = Utils.build_atoms_from_dpdata(conv_data, si, fi)
            formula = Utils.sanitize_formula(atoms.get_chemical_formula())
            
            filename = out_dir / name_fn(idx, row, formula)
            try:
                write(str(filename), atoms, format="extxyz")
                count += 1
            except Exception as e:
                logging.warning(f"Failed to write extxyz: {filename} -> {e}")
        return count

    def export_anomalies(self, conv_data: dpdata.MultiSystems, df: pd.DataFrame, stats: Dict):
        """Exports anomaly structures and indices."""
        out_dir = self.config.anomaly_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("开始导出异常帧数据...")
        
        # 1. Cohesive Anomalies
        self._export_extxyz_batch(
            conv_data, 
            df[df["is_coh_anom"]].sort_values("cohesive_energy_per_atom", ascending=False),
            out_dir / "outliers_cohesive_extxyz",
            lambda i, r, f: f"{i:05d}_cepa{r['cohesive_energy_per_atom']:.3f}_{f}.extxyz"
        )
        
        # 2. Force Anomalies
        self._export_extxyz_batch(
            conv_data,
            df[df["is_force_anom"]].sort_values("max_force_magnitude", ascending=False),
            out_dir / "outliers_force_extxyz",
            lambda i, r, f: f"{i:05d}_fmax{r['max_force_magnitude']:.2f}_{f}.extxyz"
        )
        
        # 3. Stress Anomalies
        self._export_extxyz_batch(
            conv_data,
            df[df["is_stress_anom"]].reindex(df[df["is_stress_anom"]]["pressure_gpa"].abs().sort_values(ascending=False).index),
            out_dir / "outliers_stress_extxyz",
            lambda i, r, f: f"{i:05d}_stress{np.abs(r['pressure_gpa']):.2f}_{f}.extxyz"
        )
        
        # 4. Atoms Anomalies
        self._export_extxyz_batch(
            conv_data,
            df[df["is_atoms_anom"]].sort_values("num_atoms", ascending=False),
            out_dir / "outliers_atoms_extxyz",
            lambda i, r, f: f"{i:05d}_atoms{r['num_atoms']}_{f}.extxyz"
        )

        # 5. Indices and Summary
        self._export_anomaly_indices(conv_data, df, out_dir, stats)

    def _export_anomaly_indices(self, conv_data: dpdata.MultiSystems, df: pd.DataFrame, 
                                out_dir: Path, stats: Dict):
        """Exports CSV indices of anomalies and a JSON summary."""
        def build_rows(sub):
            rows = []
            for _, row in sub.iterrows():
                try:
                    si, fi = int(row["sys_idx"]), int(row["frame_idx"])
                    atoms = Utils.build_atoms_from_dpdata(conv_data, si, fi)
                    formula = Utils.sanitize_formula(atoms.get_chemical_formula())
                    bm = int(row.get("anomaly_bitmask", 0))
                    srcs = []
                    if bm & 1: srcs.append("cohesive")
                    if bm & 2: srcs.append("force")
                    if bm & 4: srcs.append("stress")
                    if bm & 8: srcs.append("atoms")
                    
                    rows.append({
                        "sys_idx": si, "frame_idx": fi, "num_atoms": len(atoms), "formula": formula,
                        "cohesive_energy_per_atom": float(row.get("cohesive_energy_per_atom", 0.0)),
                        "max_force_magnitude": float(row.get("max_force_magnitude", 0.0)),
                        "pressure_gpa": float(row.get("pressure_gpa", 0.0)),
                        "sources": ",".join(srcs), "bitmask": bm
                    })
                except Exception:
                    pass
            return rows

        # Export CSVs
        for name, mask_col in [("cohesive", "is_coh_anom"), ("force", "is_force_anom"), 
                               ("stress", "is_stress_anom"), ("atoms", "is_atoms_anom"), ("union", "is_anom_any")]:
            target = df[df[mask_col]]
            if not target.empty:
                out_csv_path = out_dir / f"outliers_{name}_index.csv"
                pd.DataFrame(build_rows(target)).to_csv(out_csv_path, index=False)
                logging.info(f"异常索引已保存: {out_csv_path}")

        # Export Overlap Breakdown
        with open(out_dir / "anomalies_overlap_breakdown.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["bitmask", "label", "count"])
            writer.writeheader()
            for item in stats["overlap_breakdown"]:
                writer.writerow(item)

        # Export Summary JSON
        summary = {
            "thresholds": {
                "cohesive_energy": self.config.energy_thr,
                "force": self.config.force_thr,
                "stress": self.config.stress_thr,
                "max_atoms": self.config.max_atoms_thr,
            },
            "stats": stats
        }
        with open(out_dir / "anomalies_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info("异常索引与汇总已导出")

    def export_cleaned_data(self, conv_data: dpdata.MultiSystems, df_clean: pd.DataFrame, output_dir: Path):
        """Exports the cleaned dataset in extxyz and deepmd formats."""
        logging.info("开始导出清洗数据集...")
        
        # 1. Single ExtXYZ
        if "extxyz" in self.config.export_formats:
            atoms_list = []
            for _, row in df_clean.iterrows():
                si, fi = int(row["sys_idx"]), int(row["frame_idx"])
                atoms = Utils.build_atoms_from_dpdata(conv_data, si, fi)
                atoms.info["system_index"] = si
                atoms.info["frame_index"] = fi
                atoms_list.append(atoms)
                
            cleaned_xyz_dir = output_dir / "cleaned_extxyz"
            cleaned_xyz_dir.mkdir(parents=True, exist_ok=True)
            try:
                write(str(cleaned_xyz_dir / "cleaned_all.extxyz"), atoms_list, format="extxyz")
                logging.info(f"清洗数据 ExtXYZ 导出完成: {len(atoms_list)} 帧")
            except Exception as e:
                logging.error(f"ExtXYZ 导出失败: {e}")

        # 2. DeepMD Formats
        deepmd_formats = [f for f in self.config.export_formats if f.startswith("deepmd")]
        if deepmd_formats:
            systems = dpdata.MultiSystems()
            for _, row in df_clean.iterrows():
                si, fi = int(row["sys_idx"]), int(row["frame_idx"])
                systems.append(conv_data[si][fi])
            
            if "deepmd/npy" in self.config.export_formats:
                out_npy = output_dir / "cleaned_deepmd_npy"
                out_npy.mkdir(parents=True, exist_ok=True)
                try:
                    systems.to("deepmd/npy", str(out_npy))
                    logging.info(f"DeepMD (npy) 导出完成: {out_npy}")
                except Exception as e:
                    logging.error(f"DeepMD (npy) 导出失败: {e}")
            
            if "deepmd/npy/mixed" in self.config.export_formats:
                out_mixed = output_dir / "cleaned_deepmd_mixed"
                out_mixed.mkdir(parents=True, exist_ok=True)
                try:
                    systems.to("deepmd/npy/mixed", str(out_mixed))
                    logging.info(f"DeepMD (mixed) 导出完成: {out_mixed}")
                except Exception as e:
                    logging.error(f"DeepMD (mixed) 导出失败: {e}")


# =============================================================================
# Main Application
# =============================================================================

class ConvergedDataViewApp:
    """
    Main application controller that orchestrates the data pipeline.
    """
    
    def __init__(self):
        self.config = Config()
        self.config.setup_logging()
        
        self.loader = DPDataLoader(self.config)
        self.processor = DataProcessor(self.config)
        self.visualizer = DataVisualizer(self.config)
        self.cleaner = DataCleaner(self.config)
        self.exporter = DataExporter(self.config)

    def run(self):
        logging.info("=== 启动数据处理流程 ===")
        logging.info(f"输入: {self.config.input_npy_dir}")
        logging.info(f"输出: {self.config.output_dir}")

        try:
            # 1. Load Data
            conv_data = self.loader.load_systems()
            df = self.loader.build_initial_dataframe(conv_data)
            
            # 2. Process Metrics
            self.processor.compute_cohesive_energy(df)
            self.processor.add_derived_metrics(df)
            self.processor.analyze_spins_stats(df)
            
            unique_elements = self.processor.get_unique_elements(df)
            
            # 3. Initial Visualization & Stats
            self.exporter.export_statistics(df, self.config.output_dir)
            self.visualizer.plot_all(df, unique_elements)
            
            # 4. Anomaly Detection
            anomaly_stats = self.cleaner.compute_anomaly_masks(df)
            self.exporter.export_anomalies(conv_data, df, anomaly_stats)
            
            # 5. Cleaned Data Generation
            df_clean, removed_count = self.cleaner.get_clean_dataframe(df)
            logging.info(f"清洗完成: 保留 {len(df_clean)} 帧, 移除 {removed_count} 异常帧")
            
            if not df_clean.empty:
                # 6. Cleaned Data Visualization & Export
                clean_output_dir = self.config.output_dir
                self.exporter.export_statistics(df_clean, clean_output_dir / "cleaned_stats", suffix="_cleaned")
                
                # Plot cleaned data (update output dir temporarily or pass path)
                # Here we reuse visualizer but note that config.output_dir is global for it.
                # To avoid mess, we can just save cleaned stats plots to a subdir.
                original_out = self.config.output_dir
                self.config.output_dir = original_out / "cleaned_stats"
                self.config.output_dir.mkdir(exist_ok=True)
                self.visualizer.plot_all(df_clean, unique_elements)
                self.config.output_dir = original_out  # Restore
                
                self.exporter.export_cleaned_data(conv_data, df_clean, self.config.output_dir)

                # 7. Train/Test Split & Export
                if self.config.test_ratio > 0:
                    logging.info(f"开始划分数据集 (Test Ratio: {self.config.test_ratio})...")
                    df_valid = df_clean.sample(frac=self.config.test_ratio, random_state=42)
                    df_train = df_clean.drop(df_valid.index)
                    
                    logging.info(f"训练集大小: {len(df_train)} 帧")
                    logging.info(f"验证集大小: {len(df_valid)} 帧")
                    
                    if not df_train.empty:
                        self.exporter.export_cleaned_data(conv_data, df_train, self.config.output_dir / "train")
                    if not df_valid.empty:
                        self.exporter.export_cleaned_data(conv_data, df_valid, self.config.output_dir / "valid")
            else:
                logging.warning("清洗后数据为空，跳过清洗数据导出。")

            logging.info("=== 流程执行完毕 ===")
            
        except Exception as e:
            logging.critical(f"程序执行出错: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    app = ConvergedDataViewApp()
    app.run()
