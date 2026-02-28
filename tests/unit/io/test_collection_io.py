import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

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
