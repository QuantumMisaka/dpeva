
import pytest
from unittest.mock import MagicMock, patch
import json
from dpeva.labeling.generator import AbacusGenerator

class TestAbacusGenerator:
    
    @pytest.fixture
    def generator(self):
        config = {
            "dft_params": {"ecutwfc": 100},
            "pp_map": {"Fe": "Fe.upf"},
            "orb_map": {"Fe": "Fe.orb"},
            "pp_dir": "/tmp/pp",
            "orb_dir": "/tmp/orb",
            "mag_map": {"Fe": 5.0},
            "kpt_criteria": 25,
            "vacuum_thickness": 10.0
        }
        return AbacusGenerator(config)

    @patch("dpeva.labeling.generator.write_input")
    @patch("dpeva.labeling.generator.write_abacus")
    def test_gen_001_cluster_identification(self, mock_write_abacus, mock_write_input, generator, mock_atoms_cluster, tmp_path):
        """
        GEN-001: Test cluster identification and K-point generation.
        """
        output_dir = tmp_path / "task_0"
        
        # Act
        # analyze is called internally if stru_type not provided
        stru_type = generator.generate(mock_atoms_cluster, output_dir, "task_0")
        
        # Assert
        assert stru_type in ["cluster", "cubic_cluster"]
        
        # Verify KPT file content (should be Gamma only for cluster)
        kpt_file = output_dir / "KPT"
        assert kpt_file.exists()
        content = kpt_file.read_text()
        assert "1 1 1 0 0 0" in content
        
        # Verify write_input called with correct parameters
        mock_write_input.assert_called_once()
        call_args = mock_write_input.call_args[1]
        params = call_args['parameters']
        assert params['gamma_only'] == 1
        
        # Verify write_abacus called
        mock_write_abacus.assert_called_once()

    @patch("dpeva.labeling.generator.write_input")
    @patch("dpeva.labeling.generator.write_abacus")
    def test_gen_002_layer_identification(self, mock_write_abacus, mock_write_input, generator, mock_atoms_layer, tmp_path):
        """
        GEN-002: Test layer identification and dipole correction.
        """
        output_dir = tmp_path / "task_1"
        
        # Mock analyzer to return deterministic results for layer
        # Vacuum in Z (index 2) -> [False, False, True]
        generator.analyzer.analyze = MagicMock(return_value=(mock_atoms_layer, "layer", [False, False, True]))
        
        # Act
        stru_type = generator.generate(mock_atoms_layer, output_dir, "task_1")
        
        # Assert
        assert stru_type == "layer"
        
        # Verify write_input parameters for layer
        mock_write_input.assert_called_once()
        params = mock_write_input.call_args[1]['parameters']
        
        # Check dipole correction flags
        assert params.get('efield_flag') == 1
        assert params.get('dip_cor_flag') == 1
        # Vacuum in Z means efield_dir should be 2
        assert params.get('efield_dir') == 2

    @patch("dpeva.labeling.generator.write_input")
    @patch("dpeva.labeling.generator.write_abacus")
    def test_gen_003_metadata_injection(self, mock_write_abacus, mock_write_input, generator, mock_atoms_bulk, tmp_path):
        """
        GEN-003: Verify task_meta.json generation with system_name.
        """
        output_dir = tmp_path / "task_meta"
        
        # Act
        generator.generate(
            mock_atoms_bulk, 
            output_dir, 
            "sys1_0", 
            dataset_name="DS1",
            system_name="sys1"
        )
        
        # Assert
        meta_file = output_dir / "task_meta.json"
        assert meta_file.exists()
        
        with open(meta_file) as f:
            meta = json.load(f)
            
        assert meta["dataset_name"] == "DS1"
        assert meta["system_name"] == "sys1"
        assert meta["task_name"] == "sys1_0"
        assert meta["frame_idx"] == 0
        assert meta["stru_type"] is not None # Should be analyzed

    @patch("dpeva.labeling.generator.write_input")
    @patch("dpeva.labeling.generator.write_abacus")
    def test_gen_004_magmom_setting(self, mock_write_abacus, mock_write_input, generator, mock_atoms_bulk, tmp_path):
        """
        GEN-004: Verify magnetic moments setting.
        """
        output_dir = tmp_path / "task_mag"
        
        # Act
        generator.generate(mock_atoms_bulk, output_dir, "task_mag")
        
        # Assert
        # Check if atoms object passed to write_abacus has magmoms
        mock_write_abacus.assert_called_once()
        atoms_arg = mock_write_abacus.call_args[0][1] # 2nd positional arg is atoms
        
        magmoms = atoms_arg.get_initial_magnetic_moments()
        # mock_atoms_bulk has 2 Fe atoms. Config has Fe: 5.0
        assert all(m == 5.0 for m in magmoms)
