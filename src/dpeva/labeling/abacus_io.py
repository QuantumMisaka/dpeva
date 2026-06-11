"""Minimal ABACUS INPUT/KPT/STRU writers for DPEVA Labeling.

The STRU/KPT/INPUT formatting in this module follows the subset used by
ATST-Tools' vendored abacuslite implementation at
``atst-tools/src/atst_tools/external/ASE_interface/abacuslite/io/generalio.py``
from reference commit ``7812291``. This is not a full abacuslite replacement;
it only covers the DPEVA Labeling prepare path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.units import Angstrom, Bohr


ATOM_MASS = dict(zip(chemical_symbols, atomic_masses.tolist()))


def _as_path(path: str | Path) -> Path:
    return Path(path)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def write_input_file(path: str | Path, parameters: Mapping[str, Any]) -> str:
    """Write an ABACUS ``INPUT`` file from ordered key-value parameters."""
    output = _as_path(path)
    with output.open("w", encoding="ascii") as handle:
        handle.write("INPUT_PARAMETERS\n")
        for key, value in parameters.items():
            if value is None:
                continue
            handle.write(f"{key} {_format_value(value)}\n")
        handle.write("\n")
    return output.resolve().as_posix()


def write_kpt_file(path: str | Path, kpoints: Sequence[int]) -> str:
    """Write the Gamma-centered KPT mesh used by the current Labeling path."""
    if len(kpoints) != 3:
        raise ValueError(f"kpoints must contain 3 integers, got {kpoints!r}")
    output = _as_path(path)
    mesh = [int(value) for value in kpoints]
    with output.open("w", encoding="ascii") as handle:
        handle.write("K_POINTS\n")
        handle.write("0\n")
        handle.write("Gamma\n")
        handle.write(f"{mesh[0]} {mesh[1]} {mesh[2]} 0 0 0\n")
    return output.resolve().as_posix()


def _species_order(symbols: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(symbols))


def _validate_species_map(
    mapping: Mapping[str, str] | None,
    symbols: Sequence[str],
    label: str,
) -> Mapping[str, str]:
    if mapping is None:
        return {}
    missing = [symbol for symbol in _species_order(symbols) if symbol not in mapping]
    if missing:
        raise ValueError(f"Missing {label} mapping for {', '.join(missing)}")
    return mapping


def _magmom_entry(value: np.ndarray) -> str:
    if abs(float(np.linalg.norm(value))) <= 1e-10:
        return ""
    if len(value) == 1:
        return f" mag {value[0]}"
    return f" mag {value[0]} {value[1]} {value[2]}"


def write_stru_file(
    path: str | Path,
    atoms: Atoms,
    pp_map: Mapping[str, str],
    orb_map: Mapping[str, str] | None = None,
) -> str:
    """Write an ABACUS ``STRU`` file for the current Labeling subset.

    Species are grouped by first occurrence, matching the ATST-Tools vendored
    abacuslite snapshot. Mobility and velocity are currently fixed to the
    abacuslite defaults used by that writer.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ase.Atoms instance")

    symbols = atoms.get_chemical_symbols()
    pp_map = _validate_species_map(pp_map, symbols, "pseudopotential")
    orb_map = _validate_species_map(orb_map, symbols, "orbital") if orb_map else None

    output = _as_path(path)
    species_order = _species_order(symbols)
    positions = atoms.get_positions()
    magmoms = np.asarray(atoms.get_initial_magnetic_moments()).reshape(len(atoms), -1)

    with output.open("w", encoding="ascii") as handle:
        handle.write("ATOMIC_SPECIES\n")
        for symbol in species_order:
            handle.write(f"{symbol} {ATOM_MASS[symbol]} {pp_map[symbol]} \n")

        if orb_map is not None:
            handle.write("\nNUMERICAL_ORBITAL\n")
            for symbol in species_order:
                handle.write(f"{orb_map[symbol]}\n")

        handle.write("\nLATTICE_CONSTANT\n")
        handle.write(f"{Angstrom / Bohr}\n")

        handle.write("\nLATTICE_VECTORS\n")
        for vector in np.asarray(atoms.get_cell()).tolist():
            handle.write(f"{vector[0]} {vector[1]} {vector[2]}\n")

        handle.write("\nATOMIC_POSITIONS\n")
        handle.write("Cartesian\n")

        for symbol in species_order:
            indices = [index for index, current in enumerate(symbols) if current == symbol]
            handle.write(f"\n{symbol}\n")
            handle.write("0.0\n")
            handle.write(f"{len(indices)}\n")
            for index in indices:
                coord = positions[index]
                handle.write(f"{coord[0]} {coord[1]} {coord[2]}")
                handle.write(" m 1 1 1")
                handle.write(" v 0.0 0.0 0.0")
                handle.write(_magmom_entry(magmoms[index]))
                handle.write("\n")

    return output.resolve().as_posix()
