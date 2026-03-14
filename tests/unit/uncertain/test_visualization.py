
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from dpeva.uncertain.visualization import UQVisualizer
from dpeva.constants import (
    FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER,
    FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER_TRUNCATED,
)

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
        scatter_kwargs = mock_sns.scatterplot.call_args.kwargs
        assert scatter_kwargs["x"] == "uq_qbc_for"
        assert scatter_kwargs["y"] == "uq_rnd_for_rescaled"
        assert scatter_kwargs["hue"] == "diff_maxf_0_frame"
        assert scatter_kwargs["palette"] == "Reds"
        mock_plt.title.assert_called_with("UQ-QbC and UQ-RND vs Max Force Diff", fontsize=14)
        mock_plt.xlabel.assert_called_with("UQ-QbC Value", fontsize=12)
        mock_plt.ylabel.assert_called_with("UQ-RND-rescaled Value", fontsize=12)
        mock_plt.legend.assert_called_with(title="Max Force Diff", fontsize=10)
        save_path = mock_plt.savefig.call_args.args[0]
        assert save_path.endswith(os.path.join(str(visualizer.save_dir), FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER))

    def test_plot_uq_fdiff_scatter_with_truncation(self, visualizer, mock_plt, mock_sns):
        df = pd.DataFrame(
            {
                "uq_qbc_for": [0.1, 2.5],
                "uq_rnd_for_rescaled": [0.2, 2.6],
                "diff_maxf_0_frame": [0.3, 0.9],
            }
        )
        visualizer.plot_uq_fdiff_scatter(df, "strict", 0.1, 0.2, 0.1, 0.2)
        assert mock_sns.scatterplot.call_count >= 2
        save_paths = [call.args[0] for call in mock_plt.savefig.call_args_list]
        assert any(path.endswith(os.path.join(str(visualizer.save_dir), FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER)) for path in save_paths)
        assert any(path.endswith(os.path.join(str(visualizer.save_dir), FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER_TRUNCATED)) for path in save_paths)

    def test_draw_boundary(self, visualizer, mock_plt):
        """Test boundary drawing logic for different schemes."""
        schemes = ["strict", "circle_lo", "tangent_lo", "crossline_lo", "loose"]
        
        for scheme in schemes:
            visualizer._draw_boundary(scheme, 0.1, 0.2, 0.1, 0.2)
            
        # Just check that plot commands are issued
        assert mock_plt.plot.called

    def test_plot_uq_identity_scatter_missing_identity(self, visualizer, mock_plt, mock_sns):
        df = pd.DataFrame({"uq_qbc_for": [0.1], "uq_rnd_for_rescaled": [0.2]})
        visualizer.plot_uq_identity_scatter(df, "strict", 0.1, 0.2, 0.1, 0.2)
        assert not mock_sns.scatterplot.called

    def test_plot_uq_identity_scatter_with_truncation(self, visualizer, mock_plt, mock_sns):
        df = pd.DataFrame(
            {
                "uq_qbc_for": [0.1, 2.5],
                "uq_rnd_for_rescaled": [0.2, 2.6],
                "uq_identity": ["candidate", "failed"],
            }
        )
        visualizer.plot_uq_identity_scatter(df, "strict", 0.1, 0.2, 0.1, 0.2)
        assert mock_sns.scatterplot.call_count >= 2
        assert mock_plt.savefig.call_count >= 2

    def test_plot_candidate_vs_error(self, visualizer, mock_plt):
        df_uq = pd.DataFrame(
            {
                "uq_qbc_for": [0.1, 0.2],
                "uq_rnd_for_rescaled": [0.2, 0.3],
                "diff_maxf_0_frame": [0.3, 0.4],
            }
        )
        df_candidate = df_uq.iloc[[0]].copy()
        visualizer.plot_candidate_vs_error(df_uq, df_candidate)
        assert mock_plt.savefig.call_count >= 2

    def test_plot_pca_analysis_returns_dataframe(self, visualizer, mock_plt):
        explained_variance = np.array([0.6, 0.3, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002])
        all_features = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        df_uq = pd.DataFrame({"dataname": ["a-0", "a-1", "b-0", "b-1"]})
        out_df = visualizer.plot_pca_analysis(
            explained_variance=explained_variance,
            selected_PC_dim=2,
            all_features=all_features,
            direct_indices=[0, 1],
            random_indices=[2, 3],
            scores_direct=np.array([0.5, 0.6]),
            scores_random=np.array([0.4, 0.3]),
            df_uq=df_uq,
            final_indices=[0, 1],
            n_candidates=2,
            full_features=all_features,
        )
        assert list(out_df.columns) == ["PC1", "PC2"]
