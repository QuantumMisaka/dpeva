"""Explicit LMDB writer and self-check tests."""

from pathlib import Path

import dpdata
import numpy as np
import pytest
from packaging.version import Version

from dpeva.io.lmdb_index import read_lmdb_frame_index
from dpeva.io.lmdb_writer import write_lmdb

requires_lmdb_writer = pytest.mark.skipif(
    Version(getattr(dpdata, "__version__", "0")) < Version("1.1"),
    reason="deepmd/lmdb writing requires dpdata>=1.1",
)


def _system(root: Path, name: str, coords: np.ndarray, energies: np.ndarray):
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


@requires_lmdb_writer
def test_write_lmdb_preserves_system_partition(tmp_path: Path) -> None:
    first = _system(
        tmp_path,
        "first",
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        np.array([-1.0]),
    )
    second = _system(
        tmp_path,
        "second",
        np.array([[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
        np.array([-2.0]),
    )
    target = tmp_path / "out.lmdb"

    summary = write_lmdb([first, second], target, type_map=["Fe"])

    index = read_lmdb_frame_index(target)
    assert summary["nframes"] == 2
    assert summary["nsystems"] == 2
    assert [frame.system_id for frame in index.frames] == [0, 1]
    assert index.has_system_identity


@requires_lmdb_writer
def test_write_lmdb_validates_multiframe_nloc_groups(tmp_path: Path) -> None:
    first = _system(
        tmp_path,
        "first",
        np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            ]
        ),
        np.array([-1.0, -1.1]),
    )
    second = _system(
        tmp_path,
        "second",
        np.array(
            [
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [3.5, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ),
        np.array([-2.0, -2.1]),
    )
    target = tmp_path / "out.lmdb"

    summary = write_lmdb([first, second], target, type_map=["Fe"])

    index = read_lmdb_frame_index(target)
    assert summary["nframes"] == 4
    assert [frame.storage_index for frame in index.frames] == [0, 1, 2, 3]
    assert {
        nloc: len(frames) for nloc, frames in index.nloc_groups.items()
    } == {2: 2, 3: 2}


@requires_lmdb_writer
def test_write_lmdb_refuses_existing_target(tmp_path: Path) -> None:
    system = _system(
        tmp_path,
        "one",
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        np.array([-1.0]),
    )
    target = tmp_path / "out.lmdb"
    write_lmdb([system], target, type_map=["Fe"])

    with pytest.raises(FileExistsError):
        write_lmdb([system], target, type_map=["Fe"])


def test_write_lmdb_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one system"):
        write_lmdb([], tmp_path / "empty.lmdb")
