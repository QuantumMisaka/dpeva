from pathlib import Path

from ase import Atoms

from dpeva.labeling.abacus_io import (
    write_input_file,
    write_kpt_file,
    write_stru_file,
)


GOLDEN_DIR = Path(__file__).parent / "golden"


def test_abacus_writer_matches_golden_files(tmp_path: Path) -> None:
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

    write_input_file(
        tmp_path / "INPUT",
        {
            "ecutwfc": 100,
            "gamma_only": 1,
            "pseudo_dir": "/data/pp",
            "basis_dir": "/data/orb",
            "orbital_dir": "/data/orb",
            "none_value": None,
        },
    )
    write_kpt_file(tmp_path / "KPT", [3, 2, 1])
    write_stru_file(
        tmp_path / "STRU",
        atoms,
        pp_map={"C": "C.upf", "Pt": "Pt.upf", "H": "H.upf"},
        orb_map={"C": "C.orb", "Pt": "Pt.orb", "H": "H.orb"},
    )

    for filename in ("INPUT", "KPT", "STRU"):
        assert (tmp_path / filename).read_text(encoding="ascii") == (
            GOLDEN_DIR / filename
        ).read_text(encoding="ascii")
