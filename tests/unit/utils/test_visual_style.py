import matplotlib.pyplot as plt

from dpeva.utils.visual_style import (
    get_analysis_parity_profile,
    get_collection_pca_scatter_profile,
    get_legend_layout,
    get_publication_font_hierarchy,
    resolve_linked_tick_step,
    scale_font_hierarchy,
    set_visual_style,
)


def test_get_publication_font_hierarchy_builds_clear_levels():
    fonts = get_publication_font_hierarchy(12)
    assert fonts["title"] > fonts["label"] > fonts["tick"] > fonts["annotation"]
    assert fonts["legend_title"] >= fonts["legend"]


def test_resolve_linked_tick_step_returns_shared_nice_step():
    step = resolve_linked_tick_step((0.0, 0.83), (0.0, 1.71), target_tick_count=6)
    assert step == 0.5


def test_resolve_linked_tick_step_supports_denser_tick_targets():
    sparse_step = resolve_linked_tick_step((0.0, 5.5), (0.0, 5.5), target_tick_count=6)
    dense_step = resolve_linked_tick_step((0.0, 5.5), (0.0, 5.5), target_tick_count=8)
    assert sparse_step == 2.0
    assert dense_step == 1.0


def test_get_legend_layout_balances_dense_items():
    layout = get_legend_layout(item_count=17, max_rows=6, max_cols=4)
    assert layout["ncol"] == 3
    assert layout["loc"] == "upper center"
    assert layout["bbox_to_anchor"] == (0.5, -0.12)
    assert layout["bottom_margin"] > 0.18


def test_scale_font_hierarchy_preserves_semantic_levels():
    base_fonts = get_publication_font_hierarchy(12)
    scaled_fonts = scale_font_hierarchy(
        base_fonts,
        title_scale=1.24,
        label_scale=1.2,
        tick_scale=1.2,
        legend_scale=1.08,
        legend_title_scale=1.08,
    )
    assert scaled_fonts["base"] == base_fonts["base"]
    assert scaled_fonts["title"] == round(base_fonts["title"] * 1.24, 2)
    assert scaled_fonts["label"] == round(base_fonts["label"] * 1.2, 2)
    assert scaled_fonts["tick"] == round(base_fonts["tick"] * 1.2, 2)
    assert scaled_fonts["legend"] == round(base_fonts["legend"] * 1.08, 2)
    assert scaled_fonts["annotation"] == base_fonts["annotation"]


def test_get_collection_pca_scatter_profile_builds_shared_baseline():
    profile = get_collection_pca_scatter_profile(12)
    assert profile["figure_size"] == (11, 9)
    assert profile["axis_margins"] == (0.08, 0.08)
    assert profile["tick_target_count"] == 10
    fonts = profile["fonts"]
    assert "title" in fonts
    assert "label" in fonts


def test_get_analysis_parity_profile_includes_quantity_layout_and_policy():
    profile = get_analysis_parity_profile(12)
    base_enhanced = profile["enhanced"]
    force_enhanced = profile["quantity_overrides"]["force"]["enhanced"]
    energy_enhanced = profile["quantity_overrides"]["energy"]["enhanced"]
    assert base_enhanced["renderer_policy"] == "auto"
    assert base_enhanced["renderer_default"] == "scatter"
    assert force_enhanced["side_panels_enabled"] is False
    assert force_enhanced["error_inset_enabled"] is True
    assert force_enhanced["renderer_default"] == "hexbin"
    assert force_enhanced["main_density_mode"] == "hexbin"
    assert force_enhanced["colorbar_enabled"] is True
    assert force_enhanced["colorbar_title"] == "Counts Per Hexbin"
    assert force_enhanced["colorbar_orientation"] == "vertical"
    assert base_enhanced["top_panel_ylabel"] == "True Density"
    assert base_enhanced["error_panel_ylabel"] == "Error Density"
    assert base_enhanced["error_panel_xlabel_template"] == ""
    assert base_enhanced["wspace"] < 0.07
    assert base_enhanced["hexbin_wspace"] < 0.18
    assert base_enhanced["aligned_sidebar_gap"] <= 0.04
    assert base_enhanced["hexbin_aligned_sidebar_gap"] < 0.03
    assert base_enhanced["scatter_sidebar_width_scale"] >= 0.22
    assert force_enhanced["hexbin_colorbar_width_ratio"] < 0.5
    assert "width_ratios" in energy_enhanced
    assert energy_enhanced["scientific_enabled"] is False


def test_set_visual_style_applies_publication_font_sizes():
    set_visual_style(font_size=12)
    fonts = get_publication_font_hierarchy(12)
    assert plt.rcParams["axes.titlesize"] == fonts["title"]
    assert plt.rcParams["axes.labelsize"] == fonts["label"]
    assert plt.rcParams["legend.fontsize"] == fonts["legend"]
