"""Collection export routing for LMDB source systems."""

from pathlib import Path

import dpdata
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version

from dpeva.io.collection import CollectionIOManager
from dpeva.io.lmdb_writer import write_lmdb


requires_lmdb_writer = pytest.mark.skipif(
    Version(getattr(dpdata, "__version__", "0")) < Version("1.1"),
    reason="deepmd/lmdb writing requires dpdata>=1.1",
)


def _write_npy_system(root: Path, name: str, *, atom_type: int) -> object:
    system_dir = root / name
    (system_dir / "set.000").mkdir(parents=True)
    (system_dir / "type.raw").write_text(f"{atom_type} {atom_type}\n")
    (system_dir / "type_map.raw").write_text("H\nO\n")
    np.save(system_dir / "set.000" / "coord.npy", np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))
    np.save(system_dir / "set.000" / "box.npy", np.eye(3).reshape(1, 9) * 10.0)
    np.save(system_dir / "set.000" / "energy.npy", np.array([-1.0]))
    np.save(system_dir / "set.000" / "force.npy", np.zeros((1, 6)))
    np.save(system_dir / "set.000" / "virial.npy", np.zeros((1, 9)))
    return dpdata.LabeledSystem(str(system_dir), fmt="deepmd/npy")


@requires_lmdb_writer
def test_collection_exports_lmdb_source_systems(tmp_path: Path) -> None:
    first = _write_npy_system(tmp_path, "first", atom_type=0)
    second = _write_npy_system(tmp_path, "second", atom_type=1)
    lmdb_path = tmp_path / "collection.lmdb"
    write_lmdb([first, second], lmdb_path, type_map=["H", "O"])

    manager = CollectionIOManager(str(tmp_path), "savedir")
    manager.ensure_dirs()
    df_final = pd.DataFrame(
        {
            "dataname": [
                "collection.lmdb#sys0-0",
                "collection.lmdb#sys1-0",
            ]
        }
    )

    counts = manager.export_dpdata(
        str(lmdb_path),
        df_final,
        ["collection.lmdb#sys0", "collection.lmdb#sys1"],
    )

    assert counts[2] == 2
    assert counts[3] == 0
