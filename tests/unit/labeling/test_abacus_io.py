from pathlib import Path

import pytest
from ase import Atoms

from dpeva.labeling.abacus_io import (
    write_input_file,
    write_kpt_file,
    write_stru_file,
)


def test_write_input_file_writes_abacus_input_parameters(tmp_path: Path) -> None:
    output = tmp_path / "INPUT"

    write_input_file(
        output,
        {
            "ecutwfc": 100,
            "gamma_only": 1,
            "pseudo_dir": "/data/pp",
            "basis_dir": "/data/orb",
            "orbital_dir": "/data/orb",
        },
    )

    assert output.read_text(encoding="ascii") == (
        "INPUT_PARAMETERS\n"
        "ecutwfc 100\n"
        "gamma_only 1\n"
        "pseudo_dir /data/pp\n"
        "basis_dir /data/orb\n"
        "orbital_dir /data/orb\n"
        "\n"
    )


def test_write_kpt_file_writes_gamma_centered_mesh(tmp_path: Path) -> None:
    output = tmp_path / "KPT"

    write_kpt_file(output, [3, 2, 1])

    assert output.read_text(encoding="ascii") == "K_POINTS\n0\nGamma\n3 2 1 0 0 0\n"


def test_write_stru_file_uses_atst_species_order_and_magmoms(tmp_path: Path) -> None:
    output = tmp_path / "STRU"
    atoms = Atoms(
        symbols=["C", "Pt", "H", "C", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_initial_magnetic_moments([1.0, 2.0, 0.0, 3.0, 0.0])

    write_stru_file(
        output,
        atoms,
        pp_map={"C": "C.upf", "Pt": "Pt.upf", "H": "H.upf"},
        orb_map={"C": "C.orb", "Pt": "Pt.orb", "H": "H.orb"},
    )

    content = output.read_text(encoding="ascii")
    assert "ATOMIC_SPECIES\nC" in content
    assert content.index("\nC\n") < content.index("\nPt\n") < content.index("\nH\n")
    assert "NUMERICAL_ORBITAL\nC.orb\nPt.orb\nH.orb\n" in content
    assert "0.0 0.0 0.0 m 1 1 1 v 0.0 0.0 0.0 mag 1.0\n" in content
    assert "0.0 0.0 3.0 m 1 1 1 v 0.0 0.0 0.0 mag 3.0\n" in content


def test_write_stru_file_reports_missing_pseudopotential_map(tmp_path: Path) -> None:
    atoms = Atoms("FeO", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="Missing pseudopotential mapping for O"):
        write_stru_file(tmp_path / "STRU", atoms, pp_map={"Fe": "Fe.upf"})
