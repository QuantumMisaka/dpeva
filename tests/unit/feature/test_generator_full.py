
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dpeva.feature.generator import DescriptorGenerator

class TestDescriptorGenerator:
    
    @pytest.fixture
    def generator(self):
        # Mock _DEEPMD_AVAILABLE to True for testing
        with patch("dpeva.feature.generator._DEEPMD_AVAILABLE", True):
            with patch("dpeva.feature.generator.DeepPot") as mock_dp:
                gen = DescriptorGenerator("model.pb")
                gen.model = mock_dp.return_value
                return gen

    def test_init_no_deepmd(self):
        """Test initialization when DeepMD is missing."""
        with patch("dpeva.feature.generator._DEEPMD_AVAILABLE", False):
            gen = DescriptorGenerator("model.pb")
            assert gen.model is None

    def test_descriptor_from_model(self, generator):
        """Test descriptor calculation for single system."""
        # Mock dpdata.System data
        sys = MagicMock()
        sys.data = {
            "coords": np.random.rand(10, 3),
            "cells": np.random.rand(3, 3),
            "atom_names": ["H", "O"],
            "atom_types": [0, 1]
        }
        
        generator.model.get_type_map.return_value = ["H", "O"]
        generator.model.eval_descriptor.return_value = np.random.rand(10, 128)
        
        desc = generator._descriptor_from_model(sys)
        
        assert desc.shape == (10, 128)
        generator.model.eval_descriptor.assert_called()

    def test_compute_descriptors(self, generator, tmp_path):
        """Test full descriptor computation."""
        data_path = tmp_path / "data"
        data_path.mkdir()
        
        # Mock load_systems
        with patch("dpeva.feature.generator.load_systems") as mock_load:
            sys_mock = MagicMock()
            sys_mock.__len__.return_value = 10
            sys_mock.data = {"nopbc": False}
            
            # Mock _get_desc_by_batch
            with patch.object(generator, '_get_desc_by_batch') as mock_batch:
                mock_batch.return_value = [np.random.rand(10, 128)]
                
                mock_load.return_value = [sys_mock]
                
                # Test atomic mode
                desc = generator.compute_descriptors(str(data_path), output_mode="atomic")
                assert desc.shape == (10, 128)
                
                # Test structural mode
                mock_batch.return_value = [np.random.rand(10, 128)] # Reset iterator/mock
                desc = generator.compute_descriptors(str(data_path), output_mode="structural")
                assert desc.shape == (10,) # Mean pooled
