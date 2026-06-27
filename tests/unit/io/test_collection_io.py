import pytest
import os
import h5py
import numpy as np
import pandas as pd

from dpeva.io.collection import CollectionIOManager

@pytest.fixture
def mock_project(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    return d

@pytest.fixture
def manager(mock_project):
    return CollectionIOManager(str(mock_project), "root_save")

def test_ensure_dirs(manager):
    manager.ensure_dirs()
    assert os.path.exists(manager.view_savedir)
    assert os.path.exists(manager.dpdata_savedir)
    assert os.path.exists(manager.df_savedir)

def test_load_descriptors_flat(manager, tmp_path):
    desc_dir = tmp_path / "desc"
    desc_dir.mkdir()
    
    # Create dummy npy
    data = np.random.rand(5, 10, 3) # 5 frames, 10 atoms, 3 dims? No, descriptors are usually (N_frames, N_atoms * D) or (N_frames, N_atoms, D)
    # The code expects (N_frames, N_atoms, D) or similar for mean pooling
    data = np.random.rand(5, 10, 4) 
    np.save(desc_dir / "sys1.npy", data)
    
    names, stru = manager.load_descriptors(str(desc_dir), "test")
    
    assert len(names) == 5
    assert stru.shape == (5, 4) # After mean pooling
    assert names[0] == "sys1-0"

def test_load_atomic_features(manager, tmp_path):
    desc_dir = tmp_path / "desc"
    desc_dir.mkdir()
    
    data = np.random.rand(5, 10, 4)
    np.save(desc_dir / "sys1.npy", data)
    
    df = pd.DataFrame({"dataname": ["sys1-0", "sys1-2"]})
    
    X, n = manager.load_atomic_features(str(desc_dir), df)
    
    assert len(X) == 2
    assert X[0].shape == (10, 4)
    assert n[0] == 10

def _write_embedding_hdf5(path, system_name="sys1"):
    descriptor = np.arange(24, dtype=np.float32).reshape(3, 2, 4) + 1.0
    atomic_feature = np.arange(18, dtype=np.float32).reshape(3, 2, 3) + 1.0
    with h5py.File(path, "w") as h5:
        group = h5.create_group(system_name)
        group.attrs["system"] = system_name
        group.attrs["nframes"] = 3
        group.create_dataset("descriptor", data=descriptor)
        group.create_dataset("atomic_feature", data=atomic_feature)
        group.create_dataset("structural_feature", data=atomic_feature.sum(axis=1))
        group.create_dataset("atom_types", data=np.zeros((3, 2), dtype=np.int32))
    return descriptor, atomic_feature

def test_load_descriptors_from_hdf5_file(manager, tmp_path):
    h5_path = tmp_path / "embedding.hdf5"
    descriptor, _ = _write_embedding_hdf5(h5_path)

    names, stru = manager.load_descriptors(str(h5_path), "test")

    expected = descriptor.mean(axis=1)
    expected = expected / (np.linalg.norm(expected, axis=1, keepdims=True) + 1e-12)
    assert names == ["sys1-0", "sys1-1", "sys1-2"]
    np.testing.assert_allclose(stru, expected)


def test_load_descriptors_from_hdf5_atomic_feature_dataset(manager, tmp_path):
    h5_path = tmp_path / "embedding.hdf5"
    _, atomic_feature = _write_embedding_hdf5(h5_path)

    names, stru = manager.load_descriptors(
        str(h5_path),
        "test fitting last layer",
        hdf5_dataset="atomic_feature",
    )

    expected = atomic_feature.mean(axis=1)
    expected = expected / (np.linalg.norm(expected, axis=1, keepdims=True) + 1e-12)
    assert names == ["sys1-0", "sys1-1", "sys1-2"]
    assert stru.shape == (3, 3)
    np.testing.assert_allclose(stru, expected)


def test_load_descriptors_from_hdf5_directory(manager, tmp_path):
    desc_dir = tmp_path / "desc"
    desc_dir.mkdir()
    _write_embedding_hdf5(desc_dir / "embedding.hdf5")

    names, stru = manager.load_descriptors(str(desc_dir), "test")

    assert names == ["sys1-0", "sys1-1", "sys1-2"]
    assert stru.shape == (3, 4)


def test_load_descriptors_from_nested_hdf5_directory_keeps_pool_prefix(manager, tmp_path):
    desc_dir = tmp_path / "desc"
    pool_dir = desc_dir / "poolA"
    pool_dir.mkdir(parents=True)
    _write_embedding_hdf5(pool_dir / "embedding.hdf5")

    names, stru = manager.load_descriptors(str(desc_dir), "test")

    assert names == ["poolA/sys1-0", "poolA/sys1-1", "poolA/sys1-2"]
    assert stru.shape == (3, 4)


def test_load_atomic_features_from_hdf5(manager, tmp_path):
    h5_path = tmp_path / "embedding.hdf5"
    descriptor, _ = _write_embedding_hdf5(h5_path)
    df = pd.DataFrame({"dataname": ["sys1-0", "sys1-2"]})

    X, n = manager.load_atomic_features(str(h5_path), df)

    assert len(X) == 2
    np.testing.assert_array_equal(X[0], descriptor[0])
    np.testing.assert_array_equal(X[1], descriptor[2])
    assert n == [2, 2]


def test_load_atomic_features_from_hdf5_atomic_feature_dataset(manager, tmp_path):
    h5_path = tmp_path / "embedding.hdf5"
    _, atomic_feature = _write_embedding_hdf5(h5_path)
    df = pd.DataFrame({"dataname": ["sys1-0", "sys1-2"]})

    X, n = manager.load_atomic_features(
        str(h5_path),
        df,
        hdf5_dataset="atomic_feature",
    )

    assert len(X) == 2
    np.testing.assert_array_equal(X[0], atomic_feature[0])
    np.testing.assert_array_equal(X[1], atomic_feature[2])
    assert n == [2, 2]


def test_load_feature_sums_from_hdf5_atomic_feature(manager, tmp_path):
    h5_path = tmp_path / "embedding.hdf5"
    _, atomic_feature = _write_embedding_hdf5(h5_path)

    names, feature_sums, atom_counts = manager.load_feature_sums(
        str(h5_path),
        "training LLPR features",
        dataset="atomic_feature",
        normalization="sum",
    )

    assert names == ["sys1-0", "sys1-1", "sys1-2"]
    np.testing.assert_array_equal(feature_sums, atomic_feature.sum(axis=1))
    np.testing.assert_array_equal(atom_counts, np.array([2.0, 2.0, 2.0]))
