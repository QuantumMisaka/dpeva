import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch, call
from dpeva.feature.generator import DescriptorGenerator

@pytest.fixture
def mock_deep_pot():
    # Use create=True because DeepPot might not exist if import failed
    with patch("dpeva.feature.generator.DeepPot", create=True) as mock_cls:
        yield mock_cls

class TestDescriptorGenerator:
    
    @pytest.fixture
    def generator(self):
        # Mock _DEEPMD_AVAILABLE to True for testing
        with patch("dpeva.feature.generator._DEEPMD_AVAILABLE", True):
            # Patch DeepPot on the module where it is used/imported
            # In dpeva.feature.generator, DeepPot is imported inside the try-except block
            # or at module level if available.
            # We need to patch dpeva.feature.generator.DeepPot if it exists there, 
            # or mock the import.
            
            # Since the code does `from deepmd.infer.deep_pot import DeepPot`,
            # `DeepPot` becomes a name in `dpeva.feature.generator` namespace ONLY IF import succeeds.
            # If import fails, it is not defined.
            
            # We can use `patch.dict` on sys.modules to mock `deepmd.infer.deep_pot`
            # OR we can just set the attribute on the module manually if it's missing,
            # but `patch` might complain if it doesn't exist.
            
            # Better approach: Mock the class where it is DEFINED if we can, 
            # or use `create=True` in patch to create it if missing.
            
            with patch("dpeva.feature.generator.DeepPot", create=True) as mock_dp:
                gen = DescriptorGenerator("model.pb")
                gen.model = mock_dp.return_value
                yield gen

    @patch("dpeva.feature.generator._DEEPMD_AVAILABLE", True)
    def test_init(self, mock_deep_pot):
        generator = DescriptorGenerator(
            model_path="model.pt",
            head="MyHead",
            batch_size=500,
            omp_threads=4
        )
        assert generator.model_path.endswith("model.pt")
        assert generator.head == "MyHead"
        assert generator.batch_size == 500
        assert generator.omp_threads == 4
        assert os.environ['OMP_NUM_THREADS'] == '4'
        mock_deep_pot.assert_called_with(os.path.abspath("model.pt"), head="MyHead")

    @patch("dpeva.feature.generator.load_systems")
    @patch("dpeva.feature.generator._DEEPMD_AVAILABLE", True)
    def test_compute_descriptors(self, mock_load_systems, mock_deep_pot):
        """Verify compute_descriptors uses load_systems and handles list."""
        generator = DescriptorGenerator(
            model_path="model.pt",
            head="OC20M"
        )
        generator.model = MagicMock()
        
        # Mock load_systems return
        mock_sys1 = MagicMock()
        mock_sys1.data = {'nopbc': False}
        mock_sys1.__len__.return_value = 10
        
        mock_sys2 = MagicMock()
        mock_sys2.data = {'nopbc': False}
        mock_sys2.__len__.return_value = 5
        
        mock_load_systems.return_value = [mock_sys1, mock_sys2]
        
        # Mock _get_desc_by_batch
        with patch.object(generator, "_get_desc_by_batch") as mock_get_batch:
            # Return list of arrays
            mock_get_batch.side_effect = [
                [np.zeros((10, 32))], # sys1
                [np.zeros((5, 32))]   # sys2
            ]
            
            result = generator.compute_descriptors("dummy_path", output_mode="atomic")
            
            # Verification
            mock_load_systems.assert_called_with("dummy_path")
            assert mock_get_batch.call_count == 2
            assert result.shape == (15, 32)

    @patch("dpeva.feature.generator.load_systems")
    @patch("dpeva.feature.generator._DEEPMD_AVAILABLE", True)
    def test_compute_descriptors_structural(self, mock_load_systems, mock_deep_pot):
        """Verify structural output mode (mean pooling)."""
        generator = DescriptorGenerator(model_path="model.pt")
        generator.model = MagicMock()
        
        mock_sys = MagicMock()
        mock_sys.data = {'nopbc': False}
        mock_sys.__len__.return_value = 2
        mock_load_systems.return_value = [mock_sys]
        
        # 2 frames, 3 atoms each, 4 dim descriptor
        # Shape: (2, 3, 4) -> flattened to (6, 4) by DeepPot? 
        # Wait, DeepPot returns (N_frames * N_atoms, dim) usually?
        # Or (N_frames, N_atoms, dim)?
        # dpeva usually flattens.
        # Let's assume _get_desc_by_batch returns [ (N_atoms_in_batch, dim) ]
        
        # Let's say 2 frames, 3 atoms per frame. Total 6 atoms.
        raw_desc = np.ones((6, 4))
        
        with patch.object(generator, "_get_desc_by_batch") as mock_get_batch:
            mock_get_batch.return_value = [raw_desc]
            
            # This logic in generator.py:
            # desc = np.concatenate(desc_list, axis=0)
            # if output_mode == "structural":
            #    desc = np.mean(desc, axis=1)
            
            # Wait, if desc is (TotalAtoms, Dim), axis=1 mean gives (TotalAtoms,). That's not structural descriptor.
            # Structural descriptor usually means Mean over atoms for each frame.
            # But `eval_descriptor` return shape depends on DeepMD version.
            # Usually (N_frames, N_atoms, Dim) if formatted?
            # Or (N_total_atoms, Dim)?
            
            # In `generator.py`:
            # predict = self.model.eval_descriptor(coords, cells, atypes)
            # return predict
            
            # If `predict` is (N_frames, N_atoms, Dim), then `np.mean(desc, axis=1)` makes sense -> (N_frames, Dim).
            # If `predict` is (N_total_atoms, Dim), then axis=1 mean is meaningless.
            
            # DeepPot.eval_descriptor documentation says:
            # Returns: descriptor: numpy.ndarray, shape (n_frames, n_atoms, n_outputs)
            
            # So `np.concatenate` on axis 0 (batch dim) preserves (N_frames, N_atoms, Dim).
            # So `np.mean(desc, axis=1)` gives (N_frames, Dim). Correct.
            
            # Setup mock return to be (2, 3, 4)
            mock_get_batch.return_value = [np.ones((2, 3, 4))]
            
            result = generator.compute_descriptors("dummy", output_mode="structural")
            
            assert result.shape == (2, 4)
            assert np.all(result == 1)
