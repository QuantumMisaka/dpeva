"""Source-system-preserving LMDB loader for the clean workflow."""

from pathlib import Path

import dpdata
import lmdb
import msgpack
import numpy as np
import pytest
from packaging.version import Version

from dpeva.io.dataset import DatasetLoadError
from dpeva.io.lmdb_systems import load_lmdb_systems_by_id

requires_lmdb_writer = pytest.mark.skipif(
    Version(getattr(dpdata, "__version__", "0")) < Version("1.1"),
    reason="deepmd/lmdb writing requires dpdata>=1.1",
)


def _write_system(root: Path, name: str, coords: np.ndarray, energies: np.ndarray):
    system_dir = root / name
    (system_dir / "set.000").mkdir(parents=True)
    nframes, natoms, _ = coords.shape
    (system_dir / "type.raw").write_text(" ".join("0" for _ in range(natoms)))
    (system_dir / "type_map.raw").write_text("Fe")
    np.save(system_dir / "set.000" / "coord.npy", coords.reshape(nframes, -1))
    np.save(
        system_dir / "set.000" / "box.npy",
        np.tile(np.eye(3).reshape(1, 9), (nframes, 1)) * 10.0,
    )
    np.save(system_dir / "set.000" / "energy.npy", energies)
    np.save(
        system_dir / "set.000" / "force.npy",
        np.zeros((nframes, natoms * 3), dtype=float),
    )
    np.save(system_dir / "set.000" / "virial.npy", np.zeros((nframes, 9)))
    return dpdata.LabeledSystem(str(system_dir), fmt="deepmd/npy")


def _write_metadata_only(root: Path) -> Path:
    target = root / "meta.lmdb"
    env = lmdb.open(str(target), map_size=1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            txn.put(
                b"__metadata__",
                msgpack.packb(
                    {"nframes": 2, "frame_nlocs": [2, 2]},
                    use_bin_type=True,
                ),
            )
    finally:
        env.close()
    return target


@requires_lmdb_writer
def test_missing_system_identity_is_rejected(tmp_path: Path) -> None:
    target = _write_metadata_only(tmp_path)

    with pytest.raises(DatasetLoadError, match="no frame_system_ids/system_info"):
        load_lmdb_systems_by_id(target)


@requires_lmdb_writer
def test_loader_preserves_same_composition_system_partition(tmp_path: Path) -> None:
    from dpdata.formats.deepmd.lmdb import dump_systems

    first = _write_system(
        tmp_path,
        "first",
        np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ]
        ),
        np.array([-1.0, -2.0]),
    )
    second = _write_system(
        tmp_path,
        "second",
        np.array(
            [
                [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[6.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
            ]
        ),
        np.array([-3.0, -4.0]),
    )
    target = tmp_path / "same-composition.lmdb"
    dump_systems([first, second], str(target), type_map=["Fe"])

    systems = load_lmdb_systems_by_id(target)

    assert len(systems) == 2
    assert [system.target_name for system in systems] == [
        "same-composition.lmdb#sys0",
        "same-composition.lmdb#sys1",
    ]
    assert [len(system) for system in systems] == [2, 2]
    assert systems[0]["energies"].tolist() == [-1.0, -2.0]
    assert systems[1]["energies"].tolist() == [-3.0, -4.0]
    assert np.allclose(systems[0]["coords"][0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(systems[1]["coords"][0], [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
