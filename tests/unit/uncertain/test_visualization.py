import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from dpeva.uncertain.visualization import UQVisualizer
from dpeva.constants import (
    FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER,
    FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER_TRUNCATED,
    FILENAME_FINAL_SAMPLED_PCAVIEW,
    FILENAME_FINAL_SAMPLED_PCAVIEW_BY_POOL,
)


class TestUQVisualizer:
    @pytest.fixture
    def visualizer(self, tmp_path):
        return UQVisualizer(str(tmp_path))

    @pytest.fixture
    def mock_plt(self):
        with patch("dpeva.uncertain.visualization.plt") as mock:
            mock.gca.return_value.get_xlim.return_value = (0, 1)
            mock.gca.return_value.get_ylim.return_value = (0, 1)
            mock.gca.return_value.get_legend_handles_labels.return_value = (
                ["handle"],
                ["label"],
            )
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
        mock_plt.title.assert_called_with(
            "Distribution of UQ-force by KDEplot (Truncated [0, 2])",
            fontsize=visualizer.fonts["title"],
        )
        mock_plt.xlabel.assert_called_with(
            "UQ Value", fontsize=visualizer.fonts["label"]
        )
        mock_plt.ylabel.assert_called_with(
            "Density", fontsize=visualizer.fonts["label"]
        )
        mock_plt.legend.assert_called_with(
            title="Series",
            fontsize=visualizer.fonts["legend"],
            title_fontsize=visualizer.fonts["legend_title"],
            frameon=True,
        )
        assert mock_plt.savefig.called
        assert mock_plt.close.called

    def test_plot_uq_with_trust_range(self, visualizer, mock_plt, mock_sns):
        """Test plotting UQ with trust range."""
        uq_data = np.random.rand(100)

        visualizer.plot_uq_with_trust_range(uq_data, "Label", "out.png", 0.1, 0.2)

        assert mock_plt.axvline.call_count == 2
        assert mock_plt.axvspan.call_count == 3
        mock_plt.title.assert_called_with(
            "Distribution of Label by KDEplot (Truncated [0, 2])",
            fontsize=visualizer.fonts["title"],
        )
        mock_plt.xlabel.assert_called_with(
            "Label Value", fontsize=visualizer.fonts["label"]
        )
        mock_plt.ylabel.assert_called_with(
            "Density", fontsize=visualizer.fonts["label"]
        )
        mock_plt.legend.assert_called_with(
            title="Regions",
            fontsize=visualizer.fonts["legend"],
            title_fontsize=visualizer.fonts["legend_title"],
            frameon=True,
        )
        assert mock_plt.savefig.called

    def test_plot_uq_vs_error(self, visualizer, mock_plt):
        """Test plotting UQ vs Error."""
        uq_qbc = np.random.rand(100)
        uq_rnd = np.random.rand(100)
        err = np.random.rand(100)

        visualizer.plot_uq_vs_error(uq_qbc, uq_rnd, err)

        assert mock_plt.scatter.call_count == 2
        assert mock_plt.gca.return_value.xaxis.set_major_locator.called
        assert mock_plt.gca.return_value.yaxis.set_major_locator.called
        mock_plt.legend.assert_called_with(
            title="Series",
            fontsize=visualizer.fonts["legend"],
            title_fontsize=visualizer.fonts["legend_title"],
            frameon=True,
        )
        assert mock_plt.savefig.called

    def test_plot_uq_diff_parity(self, visualizer, mock_plt):
        """Test plotting UQ diff parity."""
        uq_qbc = np.random.rand(100)
        uq_rnd = np.random.rand(100)
        err = np.random.rand(100)

        visualizer.plot_uq_diff_parity(uq_qbc, uq_rnd, err)

        # 2 plots generated
        assert mock_plt.figure.call_count == 2
        assert mock_plt.gca.return_value.xaxis.set_major_locator.call_count >= 2
        assert mock_plt.gca.return_value.yaxis.set_major_locator.call_count >= 2
        assert mock_plt.savefig.call_count == 2

    def test_plot_uq_fdiff_scatter(self, visualizer, mock_plt, mock_sns):
        """Test 2D scatter plot."""
        df = pd.DataFrame(
            {
                "uq_qbc_for": np.random.rand(100),
                "uq_rnd_for_rescaled": np.random.rand(100),
                "diff_maxf_0_frame": np.random.rand(100),
            }
        )

        visualizer.plot_uq_fdiff_scatter(df, "strict", 0.1, 0.2, 0.1, 0.2)

        assert mock_sns.scatterplot.called
        scatter_kwargs = mock_sns.scatterplot.call_args.kwargs
        assert scatter_kwargs["x"] == "uq_qbc_for"
        assert scatter_kwargs["y"] == "uq_rnd_for_rescaled"
        assert scatter_kwargs["hue"] == "diff_maxf_0_frame"
        assert scatter_kwargs["palette"] == "Reds"
        mock_plt.title.assert_called_with(
            "UQ-QbC and UQ-RND vs Max Force Diff",
            fontsize=visualizer.fonts["title"],
        )
        mock_plt.xlabel.assert_called_with(
            "UQ-QbC Value",
            fontsize=visualizer.fonts["label"],
        )
        mock_plt.ylabel.assert_called_with(
            "UQ-RND-rescaled Value",
            fontsize=visualizer.fonts["label"],
        )
        mock_plt.legend.assert_called_with(
            title="Max Force Diff",
            fontsize=visualizer.fonts["legend"],
            title_fontsize=visualizer.fonts["legend_title"],
            frameon=True,
        )
        assert mock_plt.gca.return_value.xaxis.set_major_locator.called
        assert mock_plt.gca.return_value.yaxis.set_major_locator.called
        save_path = mock_plt.savefig.call_args.args[0]
        assert save_path.endswith(
            os.path.join(
                str(visualizer.save_dir), FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER
            )
        )

    def test_plot_uq_fdiff_scatter_with_truncation(
        self, visualizer, mock_plt, mock_sns
    ):
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
        assert any(
            path.endswith(
                os.path.join(
                    str(visualizer.save_dir), FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER
                )
            )
            for path in save_paths
        )
        assert any(
            path.endswith(
                os.path.join(
                    str(visualizer.save_dir),
                    FILENAME_UQ_FORCE_QBC_RND_FDIFF_SCATTER_TRUNCATED,
                )
            )
            for path in save_paths
        )
        mock_plt.gca.return_value.set_xlim.assert_called_with(0, 2.0)
        mock_plt.gca.return_value.set_ylim.assert_called_with(0, 2.0)

    def test_draw_boundary(self, visualizer, mock_plt):
        """Test boundary drawing logic for different schemes."""
        schemes = ["strict", "circle_lo", "tangent_lo", "crossline_lo", "loose"]

        for scheme in schemes:
            visualizer._draw_boundary(scheme, 0.1, 0.2, 0.1, 0.2)

        # Just check that plot commands are issued
        assert mock_plt.plot.called

    def test_plot_uq_identity_scatter_missing_identity(
        self, visualizer, mock_plt, mock_sns
    ):
        df = pd.DataFrame({"uq_qbc_for": [0.1], "uq_rnd_for_rescaled": [0.2]})
        visualizer.plot_uq_identity_scatter(df, "strict", 0.1, 0.2, 0.1, 0.2)
        assert not mock_sns.scatterplot.called

    def test_plot_uq_identity_scatter_with_truncation(
        self, visualizer, mock_plt, mock_sns
    ):
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
        x_locator = mock_plt.gca.return_value.xaxis.set_major_locator.call_args.args[0]
        y_locator = mock_plt.gca.return_value.yaxis.set_major_locator.call_args.args[0]
        assert x_locator._edge.step == 0.25
        assert y_locator._edge.step == 0.25

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
        assert mock_plt.gca.return_value.xaxis.set_major_locator.call_count >= 2
        assert mock_plt.gca.return_value.yaxis.set_major_locator.call_count >= 2
        mock_plt.legend.assert_called_with(
            title="Series",
            fontsize=visualizer.fonts["legend"],
            title_fontsize=visualizer.fonts["legend_title"],
            frameon=True,
        )

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

    def test_plot_pca_analysis_joint_mode_generates_by_pool_plot(
        self, visualizer, mock_plt
    ):
        pca_profile = visualizer._build_pca_scatter_profile()
        pca_fonts = pca_profile["fonts"]
        explained_variance = np.array([0.6, 0.3, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002])
        all_features = np.array(
            [[0.1 * (idx + 1), 0.1 * (idx + 2)] for idx in range(10)], dtype=float
        )
        df_uq = pd.DataFrame(
            {
                "dataname": [f"pool{idx + 1}/sys-{idx}" for idx in range(7)],
            }
        )
        visualizer.plot_pca_analysis(
            explained_variance=explained_variance,
            selected_PC_dim=2,
            all_features=all_features,
            direct_indices=list(range(7)),
            random_indices=[0],
            scores_direct=np.array([0.6, 0.5]),
            scores_random=np.array([0.4, 0.3]),
            df_uq=df_uq,
            final_indices=list(range(7)),
            n_candidates=7,
            full_features=all_features,
        )
        save_paths = [call.args[0] for call in mock_plt.savefig.call_args_list]
        assert any(
            path.endswith(
                os.path.join(str(visualizer.save_dir), FILENAME_FINAL_SAMPLED_PCAVIEW)
            )
            for path in save_paths
        )
        assert any(
            path.endswith(
                os.path.join(
                    str(visualizer.save_dir), FILENAME_FINAL_SAMPLED_PCAVIEW_BY_POOL
                )
            )
            for path in save_paths
        )
        by_pool_save_call = next(
            call
            for call in mock_plt.savefig.call_args_list
            if call.args[0].endswith(
                os.path.join(
                    str(visualizer.save_dir), FILENAME_FINAL_SAMPLED_PCAVIEW_BY_POOL
                )
            )
        )
        legend_kwargs = mock_plt.legend.call_args_list[-1].kwargs
        assert legend_kwargs["loc"] == "center left"
        assert legend_kwargs["bbox_to_anchor"] == (1.02, 0.5)
        assert "ncol" not in legend_kwargs
        assert legend_kwargs["fontsize"] == pca_fonts["legend"]
        assert by_pool_save_call.kwargs == {"dpi": visualizer.dpi}
        assert any(
            call.kwargs.get("fontsize") == pca_fonts["title"]
            for call in mock_plt.title.call_args_list
            if call.args and call.args[0] == "PCA of UQ-DIRECT sampling"
        )
        assert any(
            call.kwargs.get("fontsize") == pca_fonts["title"]
            for call in mock_plt.title.call_args_list
            if call.args and call.args[0] == "DIRECT Coverage Analysis"
        )
        assert any(
            call.kwargs.get("fontsize") == pca_fonts["title"]
            for call in mock_plt.title.call_args_list
            if call.args
            and call.args[0] == "PCA of UQ-DIRECT sampling (Sampled by Pool)"
        )
        assert any(
            call.kwargs.get("fontsize") == pca_fonts["label"]
            for call in mock_plt.xlabel.call_args_list
            if call.args and call.args[0] == "PC1"
        )
        figure_sizes = [call.kwargs.get("figsize") for call in mock_plt.figure.call_args_list]
        assert figure_sizes.count(pca_profile["figure_size"]) == 3
        assert any(
            call.kwargs.get("labelsize") == pca_fonts["tick"]
            for call in mock_plt.gca.return_value.tick_params.call_args_list
        )
        assert mock_plt.gca.return_value.margins.call_count == 4
        for call in mock_plt.gca.return_value.margins.call_args_list:
            assert call.kwargs == {
                "x": pca_profile["axis_margins"][0],
                "y": pca_profile["axis_margins"][1],
            }
        mock_plt.gcf.return_value.subplots_adjust.assert_called_with(right=0.76)

    def test_plot_pca_analysis_joint_mode_handles_non_range_index(
        self, visualizer, mock_plt
    ):
        explained_variance = np.array([0.6, 0.3, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002])
        all_features = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        df_uq = pd.DataFrame({"dataname": ["pool1/a-0", "pool2/b-0"]}, index=[100, 200])
        visualizer.plot_pca_analysis(
            explained_variance=explained_variance,
            selected_PC_dim=2,
            all_features=all_features,
            direct_indices=[0, 1],
            random_indices=[0],
            scores_direct=np.array([0.6, 0.5]),
            scores_random=np.array([0.4, 0.3]),
            df_uq=df_uq,
            final_indices=[0, 1],
            n_candidates=2,
            full_features=all_features,
        )
        save_paths = [call.args[0] for call in mock_plt.savefig.call_args_list]
        assert any(
            path.endswith(
                os.path.join(
                    str(visualizer.save_dir), FILENAME_FINAL_SAMPLED_PCAVIEW_BY_POOL
                )
            )
            for path in save_paths
        )

    def test_apply_linked_major_ticks_uses_shared_locator_step(self, visualizer):
        axis = type(
            "Axis",
            (),
            {
                "get_xlim": lambda self: (0.0, 0.83),
                "get_ylim": lambda self: (0.0, 1.71),
                "xaxis": type(
                    "LocatorHolder",
                    (),
                    {
                        "set_major_locator": lambda self, locator: setattr(
                            self, "locator", locator
                        )
                    },
                )(),
                "yaxis": type(
                    "LocatorHolder",
                    (),
                    {
                        "set_major_locator": lambda self, locator: setattr(
                            self, "locator", locator
                        )
                    },
                )(),
            },
        )()
        step = visualizer._apply_linked_major_ticks(axis)
        assert step == 0.5
        assert axis.xaxis.locator._edge.step == axis.yaxis.locator._edge.step == 0.5

    def test_apply_pca_axis_layout_uses_profile_margins_and_tick_density(
        self, visualizer
    ):
        axis = type(
            "Axis",
            (),
            {
                "get_xlim": lambda self: (0.0, 5.5),
                "get_ylim": lambda self: (0.0, 5.5),
                "margins": lambda self, x, y: setattr(self, "margin_args", (x, y)),
                "xaxis": type(
                    "LocatorHolder",
                    (),
                    {
                        "set_major_locator": lambda self, locator: setattr(
                            self, "locator", locator
                        )
                    },
                )(),
                "yaxis": type(
                    "LocatorHolder",
                    (),
                    {
                        "set_major_locator": lambda self, locator: setattr(
                            self, "locator", locator
                        )
                    },
                )(),
            },
        )()
        profile = visualizer._build_pca_scatter_profile()
        step = visualizer._apply_pca_axis_layout(axis, profile)
        assert axis.margin_args == profile["axis_margins"]
        assert step == 1.0
        assert axis.xaxis.locator._edge.step == axis.yaxis.locator._edge.step == 1.0

    def test_plot_pca_analysis_normal_mode_skips_by_pool_plot(
        self, visualizer, mock_plt
    ):
        pca_profile = visualizer._build_pca_scatter_profile()
        explained_variance = np.array([0.6, 0.3, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002])
        all_features = np.array([[0.1, 0.2], [0.2, 0.3]])
        df_uq = pd.DataFrame({"dataname": ["a-0", "b-0"]})
        visualizer.plot_pca_analysis(
            explained_variance=explained_variance,
            selected_PC_dim=2,
            all_features=all_features,
            direct_indices=[0],
            random_indices=[1],
            scores_direct=np.array([0.6, 0.5]),
            scores_random=np.array([0.4, 0.3]),
            df_uq=df_uq,
            final_indices=[0],
            n_candidates=None,
            full_features=all_features,
        )
        save_paths = [call.args[0] for call in mock_plt.savefig.call_args_list]
        assert any(
            path.endswith(
                os.path.join(str(visualizer.save_dir), FILENAME_FINAL_SAMPLED_PCAVIEW)
            )
            for path in save_paths
        )
        assert not any(
            path.endswith(
                os.path.join(
                    str(visualizer.save_dir), FILENAME_FINAL_SAMPLED_PCAVIEW_BY_POOL
                )
            )
            for path in save_paths
        )
        figure_sizes = [call.kwargs.get("figsize") for call in mock_plt.figure.call_args_list]
        assert figure_sizes.count(pca_profile["figure_size"]) == 3
        assert any(
            call.args == ("PC1",)
            and call.kwargs.get("fontsize") == pca_profile["fonts"]["label"]
            for call in mock_plt.xlabel.call_args_list
        )
        assert any(
            call.args == ("PC2",)
            and call.kwargs.get("fontsize") == pca_profile["fonts"]["label"]
            for call in mock_plt.ylabel.call_args_list
        )
        assert mock_plt.gca.return_value.margins.call_count == 3
