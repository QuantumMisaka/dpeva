"""Frame-index and dp-test detail attribution for DeepMD LMDB."""

from pathlib import Path

import numpy as np
import pytest

import dpdata
import lmdb
import msgpack
from packaging.version import Version

from dpeva.io.dataproc import DPTestResultParser
from dpeva.io.dataset import DatasetLoadError
from dpeva.io.lmdb_index import read_lmdb_frame_index

requires_lmdb_writer = pytest.mark.skipif(
    Version(getattr(dpdata, "__version__", "0")) < Version("1.1"),
    reason="deepmd/lmdb writing requires dpdata>=1.1",
)


def _write_npy_system(root: Path, name: str, atom_types: list[int], energies: list[float]):
    system_dir = root / name
    (system_dir / "set.000").mkdir(parents=True)
    (system_dir / "type.raw").write_text(" ".join(str(t) for t in atom_types))
    (system_dir / "type_map.raw").write_text("Fe C")
    nframes = len(energies)
    natoms = len(atom_types)
    np.save(
        system_dir / "set.000" / "coord.npy",
        (np.arange(nframes * natoms * 3, dtype=float).reshape(nframes, natoms * 3) * 0.1),
    )
    np.save(
        system_dir / "set.000" / "box.npy",
        np.tile(np.eye(3).reshape(1, 9), (nframes, 1)) * 10.0,
    )
    np.save(system_dir / "set.000" / "energy.npy", np.asarray(energies, dtype=float))
    np.save(
        system_dir / "set.000" / "force.npy",
        np.zeros((nframes, natoms * 3), dtype=float),
    )
    np.save(
        system_dir / "set.000" / "virial.npy",
        np.zeros((nframes, 9), dtype=float),
    )
    return dpdata.LabeledSystem(str(system_dir), fmt="deepmd/npy")


def _write_lmdb(root: Path) -> Path:
    from dpdata.formats.deepmd.lmdb import dump_systems

    systems = [
        _write_npy_system(root, "Fe2", [0, 0], [-1.0, -1.1]),
        _write_npy_system(root, "Fe2C1", [0, 0, 1], [-2.0, -2.1]),
    ]
    target = root / "mixed.lmdb"
    dump_systems(systems, str(target), type_map=["Fe", "C"])
    return target


@requires_lmdb_writer
def test_frame_index_orders_nloc_then_storage_order(tmp_path: Path) -> None:
    target = _write_lmdb(tmp_path)

    index = read_lmdb_frame_index(target)

    assert [
        (frame.storage_index, frame.system_id, frame.system_frame_index, frame.nloc)
        for frame in index.test_order
    ] == [
        (0, 0, 0, 2),
        (1, 0, 1, 2),
        (2, 1, 0, 3),
        (3, 1, 1, 3),
    ]
    assert index.dataname_list == [
        ["mixed.lmdb#sys0", 0, 2],
        ["mixed.lmdb#sys0", 1, 2],
        ["mixed.lmdb#sys1", 0, 3],
        ["mixed.lmdb#sys1", 1, 3],
    ]
    assert index.datanames_nframe == {
        "mixed.lmdb#sys0": 2,
        "mixed.lmdb#sys1": 2,
    }


@requires_lmdb_writer
def test_parser_consumes_lmdb_detail_sections(tmp_path: Path) -> None:
    target = _write_lmdb(tmp_path)
    header_e = f"# {target} [nloc=2]: data_e pred_e"
    header_f = f"# {target} [nloc=2]: data_fx data_fy data_fz pred_fx pred_fy pred_fz"
    header_e2 = f"# {target} [nloc=3]: data_e pred_e"
    header_f2 = f"# {target} [nloc=3]: data_fx data_fy data_fz pred_fx pred_fy pred_fz"
    (tmp_path / "results.e_peratom.out").write_text(
        "\n".join([header_e, "0 0", "0 0", header_e2, "0 0", "0 0"]) + "\n"
    )
    (tmp_path / "results.f.out").write_text(
        "\n".join(
            [
                header_f,
                *(["0 0 0 0 0 0"] * 4),
                header_f2,
                *(["0 0 0 0 0 0"] * 6),
            ]
        )
        + "\n"
    )

    parsed = DPTestResultParser(str(tmp_path), head="results").parse()

    assert parsed["dataname_list"] == [
        ["mixed.lmdb#sys0", 0, 2],
        ["mixed.lmdb#sys0", 1, 2],
        ["mixed.lmdb#sys1", 0, 3],
        ["mixed.lmdb#sys1", 1, 3],
    ]
    assert parsed["datanames_nframe"] == {
        "mixed.lmdb#sys0": 2,
        "mixed.lmdb#sys1": 2,
    }


@requires_lmdb_writer
def test_parser_rejects_wrong_lmdb_group_order(tmp_path: Path) -> None:
    target = _write_lmdb(tmp_path)
    (tmp_path / "results.e_peratom.out").write_text(
        "\n".join(
            [
                f"# {target} [nloc=3]: data_e pred_e",
                "0 0",
                "0 0",
                f"# {target} [nloc=2]: data_e pred_e",
                "0 0",
                "0 0",
            ]
        )
        + "\n"
    )

    with pytest.raises(DatasetLoadError, match="do not match"):
        DPTestResultParser(str(tmp_path), head="results").parse()


def _write_metadata_lmdb(root: Path, metadata: dict) -> Path:
    target = root / "meta.lmdb"
    env = lmdb.open(str(target), map_size=1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
    finally:
        env.close()
    return target


def test_metadata_index_uses_frame_nlocs_and_system_ids(tmp_path: Path) -> None:
    target = _write_metadata_lmdb(
        tmp_path,
        {
            "nframes": 4,
            "frame_nlocs": [3, 2, 2, 3],
            "frame_system_ids": [0, 1, 1, 0],
        },
    )

    index = read_lmdb_frame_index(target)

    assert [
        (frame.storage_index, frame.system_id, frame.nloc)
        for frame in index.test_order
    ] == [(1, 1, 2), (2, 1, 2), (0, 0, 3), (3, 0, 3)]


def test_metadata_index_falls_back_to_system_info(tmp_path: Path) -> None:
    target = _write_metadata_lmdb(
        tmp_path,
        {
            "nframes": 4,
            "system_info": [
                {"natoms": [2, 1], "nframes": 2},
                {"natoms": [1, 0], "nframes": 2},
            ],
        },
    )

    index = read_lmdb_frame_index(target)

    assert [
        (frame.storage_index, frame.system_id, frame.nloc)
        for frame in index.test_order
    ] == [(2, 1, 1), (3, 1, 1), (0, 0, 3), (1, 0, 3)]
    assert index.datanames_nframe == {
        "meta.lmdb#sys1": 2,
        "meta.lmdb#sys0": 2,
    }


def test_metadata_index_rejects_system_with_multiple_nlocs(tmp_path: Path) -> None:
    target = _write_metadata_lmdb(
        tmp_path,
        {
            "nframes": 2,
            "frame_nlocs": [2, 3],
            "frame_system_ids": [0, 0],
        },
    )

    with pytest.raises(DatasetLoadError, match="multiple nloc"):
        read_lmdb_frame_index(target)
