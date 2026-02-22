
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from dpeva.uncertain.visualization import UQVisualizer

class TestUQVisualizer:
    
    @pytest.fixture
    def visualizer(self, tmp_path):
        return UQVisualizer(str(tmp_path))

    @pytest.fixture
    def mock_plt(self):
        with patch("dpeva.uncertain.visualization.plt") as mock:
            # Setup get_xlim return values for any ax created
            mock.gca.return_value.get_xlim.return_value = (0, 1)
            mock.gca.return_value.get_ylim.return_value = (0, 1)
            yield mock

    @pytest.fixture
    def mock_sns(self):
        with patch("dpeva.uncertain.visualization.sns") as mock:
            yield mock

    def test_filter_uq(self, visualizer):
        """Test UQ filtering logic."""
        data = np.array([-0.1, 0.5, 1.5, 2.5])
        
        filtered, mask = visualizer._filter_uq(data, "test")
        
        assert len(filtered) == 2
        assert np.all(filtered >= 0)
        assert np.all(filtered <= 2)
        assert np.array_equal(mask, [False, True, True, False])

    def test_plot_uq_distribution(self, visualizer, mock_plt, mock_sns):
        """Test plotting UQ distribution."""
        uq_qbc = np.random.rand(100)
        uq_rnd = np.random.rand(100)
        
        visualizer.plot_uq_distribution(uq_qbc, uq_rnd)
        
        assert mock_plt.figure.called
        assert mock_sns.kdeplot.call_count == 2
        assert mock_plt.savefig.called
        assert mock_plt.close.called

    def test_plot_uq_with_trust_range(self, visualizer, mock_plt, mock_sns):
        """Test plotting UQ with trust range."""
        uq_data = np.random.rand(100)
        
        visualizer.plot_uq_with_trust_range(uq_data, "Label", "out.png", 0.1, 0.2)
        
        assert mock_plt.axvline.call_count == 2
        assert mock_plt.axvspan.call_count == 3
        assert mock_plt.savefig.called

    def test_plot_uq_vs_error(self, visualizer, mock_plt):
        """Test plotting UQ vs Error."""
        uq_qbc = np.random.rand(100)
        uq_rnd = np.random.rand(100)
        err = np.random.rand(100)
        
        visualizer.plot_uq_vs_error(uq_qbc, uq_rnd, err)
        
        assert mock_plt.scatter.call_count == 2
        assert mock_plt.savefig.called

    def test_plot_uq_diff_parity(self, visualizer, mock_plt):
        """Test plotting UQ diff parity."""
        uq_qbc = np.random.rand(100)
        uq_rnd = np.random.rand(100)
        err = np.random.rand(100)
        
        visualizer.plot_uq_diff_parity(uq_qbc, uq_rnd, err)
        
        # 2 plots generated
        assert mock_plt.figure.call_count == 2
        assert mock_plt.savefig.call_count == 2

    def test_plot_uq_fdiff_scatter(self, visualizer, mock_plt, mock_sns):
        """Test 2D scatter plot."""
        df = pd.DataFrame({
            "uq_qbc_for": np.random.rand(100),
            "uq_rnd_for_rescaled": np.random.rand(100),
            "diff_maxf_0_frame": np.random.rand(100)
        })
        
        visualizer.plot_uq_fdiff_scatter(df, "strict", 0.1, 0.2, 0.1, 0.2)
        
        assert mock_sns.scatterplot.called
        assert mock_plt.savefig.called

    def test_draw_boundary(self, visualizer, mock_plt):
        """Test boundary drawing logic for different schemes."""
        schemes = ["strict", "circle_lo", "tangent_lo", "crossline_lo", "loose"]
        
        for scheme in schemes:
            visualizer._draw_boundary(scheme, 0.1, 0.2, 0.1, 0.2)
            
        # Just check that plot commands are issued
        assert mock_plt.plot.called
