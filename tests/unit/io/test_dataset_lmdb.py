"""LMDB input contract for :mod:`dpeva.io.dataset`.

The behaviour asserted here mirrors the verified DeepMD/dpdata facts recorded in
``docs/superpowers/plans/2026-09-20-lmdb-format-compatibility.md`` §1:

* LMDB must only ever be read through ``dpdata.MultiSystems`` — a single-system
  read silently returns the first composition group of a multi-composition file.
* ``deepmd/lmdb`` is the supported format name; the bare ``lmdb`` alias is
  normalized because dpdata 1.0.x maps it to a different legacy plugin.
* A zero-system load is an error unless a caller explicitly opts into ``warn``.
"""

from unittest.mock import MagicMock, patch

import pytest

from dpeva.io.dataset import (
    DatasetLoadError,
    LMDB_FORMAT,
    detect_dataset_kind,
    is_lmdb_detail_label,
    is_lmdb_path,
    load_systems,
    normalize_format,
)


def _make_lmdb_dir(root, name="set.lmdb", mdb_content: bytes = b""):
    path = root / name
    path.mkdir()
    (path / "data.mdb").write_bytes(mdb_content)
    (path / "lock.mdb").write_bytes(b"")
    return path


class TestLmdbDetection:
    def test_suffix_and_data_mdb_are_both_accepted(self, tmp_path):
        assert is_lmdb_path(str(tmp_path / "payload.lmdb"))
        assert is_lmdb_path(str(_make_lmdb_dir(tmp_path)))
        assert not is_lmdb_path(str(tmp_path / "plain_directory"))
        assert not is_lmdb_path(None)

    def test_dataset_kind_separates_lmdb_from_npy_layouts(self, tmp_path):
        lmdb = _make_lmdb_dir(tmp_path)
        single = tmp_path / "Fe6O0C2"
        single.mkdir()
        (single / "type.raw").write_text("0 1")
        container = tmp_path / "pools"
        container.mkdir()

        assert detect_dataset_kind(lmdb) == "lmdb"
        assert detect_dataset_kind(single) == "single_system"
        assert detect_dataset_kind(container) == "container"
        assert detect_dataset_kind(tmp_path / "absent") == "missing"

    def test_legacy_alias_is_normalized_to_canonical_name(self):
        assert normalize_format("lmdb") == LMDB_FORMAT
        assert normalize_format("deepmd/npy") == "deepmd/npy"

    def test_detail_label_recognizes_lmdb_headers(self):
        assert is_lmdb_detail_label("/data/g3.lmdb [nloc=9]")
        assert is_lmdb_detail_label("/data/g3.lmdb")
        assert not is_lmdb_detail_label("/data/cleaned_v1/Fe6O0C2")


class TestLmdbRouting:
    def test_lmdb_never_enters_the_single_system_path(self, tmp_path):
        lmdb = _make_lmdb_dir(tmp_path)
        sentinel = MagicMock(name="lmdb-group")
        sentinel.__len__ = lambda self: 3

        with patch("dpeva.io.dataset._load_lmdb_systems", return_value=[sentinel]) as loader, \
             patch("dpeva.io.dataset._load_single_path") as single:
            systems = load_systems(str(lmdb), fmt="auto")

        assert systems == [sentinel]
        loader.assert_called_once()
        single.assert_not_called()

    def test_single_system_helper_refuses_lmdb(self, tmp_path):
        from dpeva.io.dataset import _load_single_path

        lmdb = _make_lmdb_dir(tmp_path)
        with pytest.raises(DatasetLoadError, match="single-system path"):
            _load_single_path(str(lmdb), "sys")

    def test_target_systems_is_rejected_for_lmdb(self, tmp_path):
        lmdb = _make_lmdb_dir(tmp_path)
        with pytest.raises(DatasetLoadError, match="cannot be applied to LMDB input"):
            load_systems(str(lmdb), fmt="auto", target_systems=["Fe6O0C2"])

    def test_zero_system_policy(self, tmp_path):
        empty = _make_lmdb_dir(tmp_path)
        with patch("dpeva.io.dataset._load_lmdb_systems", return_value=[]):
            with pytest.raises(DatasetLoadError, match="No systems could be loaded"):
                load_systems(str(empty), fmt="auto")
            assert load_systems(str(empty), fmt="auto", on_empty="warn") == []


def _dpdata_supports_lmdb() -> bool:
    try:
        from packaging.version import Version
    except ImportError:  # pragma: no cover
        return False
    import dpdata

    try:
        return Version(getattr(dpdata, "__version__", "0")) >= Version("1.1")
    except Exception:  # pragma: no cover - unparsable local build
        return False


@pytest.mark.skipif(
    not _dpdata_supports_lmdb(),
    reason="deepmd/lmdb reading requires dpdata>=1.1",
)
class TestLmdbRoundTrip:
    """Real-file checks; skipped on environments that cannot read LMDB at all."""

    @staticmethod
    def _write_npy_system(root, name, atom_types, energy):
        """Write a minimal labelled deepmd/npy system directory."""
        import dpdata
        import numpy as np

        system_dir = root / name
        (system_dir / "set.000").mkdir(parents=True)
        (system_dir / "type.raw").write_text(" ".join(str(t) for t in atom_types))
        # deepmd/npy only names elements when type_map.raw is present; without it
        # dpdata falls back to synthetic "Type_0" names.
        (system_dir / "type_map.raw").write_text("Fe C")
        natoms = len(atom_types)
        np.save(
            system_dir / "set.000" / "coord.npy",
            np.arange(natoms * 3, dtype=float).reshape(1, natoms * 3) * 0.1,
        )
        np.save(system_dir / "set.000" / "box.npy", np.tile(np.eye(3), (1, 1)).reshape(1, 9) * 10.0)
        np.save(system_dir / "set.000" / "energy.npy", np.asarray([energy], dtype=float))
        np.save(system_dir / "set.000" / "force.npy", np.zeros((1, natoms * 3)))
        np.save(system_dir / "set.000" / "virial.npy", np.zeros((1, 9)))
        return dpdata.LabeledSystem(str(system_dir), fmt="deepmd/npy")

    @classmethod
    def _write_dataset(cls, tmp_path):
        target = tmp_path / "tiny.lmdb"
        from dpdata.formats.deepmd.lmdb import dump_systems

        # Two different compositions: a single-system read would return only the
        # first group, so the frame total below detects that regression.
        systems = [
            cls._write_npy_system(tmp_path, "Fe2C0", [0, 0], energy=-1.0),
            cls._write_npy_system(tmp_path, "Fe1C1", [0, 1], energy=-2.0),
        ]
        dump_systems(systems, str(target), type_map=["Fe", "C"])
        assert (target / "data.mdb").is_file()
        return target

    def test_reads_every_frame_of_every_group(self, tmp_path):
        target = self._write_dataset(tmp_path)

        systems = load_systems(str(target), fmt="auto")

        assert sum(len(system) for system in systems) == 2
        assert all(system.target_name.startswith("tiny.lmdb[") for system in systems)

    def test_legacy_alias_still_reads_through_canonical_format(self, tmp_path):
        target = self._write_dataset(tmp_path)

        systems = load_systems(str(target), fmt="lmdb")

        assert sum(len(system) for system in systems) == 2

    def test_frame_count_mismatch_is_rejected(self, tmp_path):
        target = self._write_dataset(tmp_path)

        with patch("dpeva.io.dataset._lmdb_stored_frame_count", return_value=99):
            with pytest.raises(DatasetLoadError, match="frame-count mismatch"):
                load_systems(str(target), fmt="auto")
