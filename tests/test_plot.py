"""Tests for the grid plotting helpers in cc_mapping.plot."""

import anndata as ad
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cc_mapping.plot import (
    GridLabelRenderer,
    combine_figures_with_gridspec,
    get_legend,
    plot_row_partitions,
)

matplotlib.use("agg")

RGB_PALETTE = ["#ff0000", "#00ff00", "#0000ff"]  # G1, S, G2


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def embedded_adata():
    """300 cells with a categorical phase, a phase_colors column, and a 2D embedding."""
    rng = np.random.default_rng(0)
    n_cells = 300
    phase = rng.choice(["G1", "S", "G2"], size=n_cells)
    obs = pd.DataFrame(
        {
            "phase": pd.Categorical(phase, categories=["G1", "S", "G2"]),
            "phase_colors": pd.Series(phase)
            .map({"G1": "orange", "S": "orchid", "G2": "dodgerblue"})
            .values,
            "condition": rng.choice(["ctrl", "drug"], size=n_cells),
        },
        index=np.arange(n_cells).astype(str),
    )
    adata = ad.AnnData(
        X=rng.normal(size=(n_cells, 3)),
        obs=obs,
        var=pd.DataFrame(index=["m0", "m1", "m2"]),
    )
    adata.obsm["X_phate"] = rng.normal(size=(n_cells, 2))
    return adata


def _cell_facecolors(fig, cell_idx):
    """Facecolors of the foreground scatter in grid cell `cell_idx` (row-major).

    fig.axes[0] and fig.axes[1] are the column/row label axes.
    """
    return fig.axes[2 + cell_idx].collections[-1].get_facecolors()


# ---------------------------------------------------------------------------
# GridLabelRenderer
# ---------------------------------------------------------------------------


def test_grid_label_renderer_uses_colormap_registry():
    renderer = GridLabelRenderer()
    assert renderer.row_cmap(0) == matplotlib.colormaps["tab20"](0)
    assert renderer.col_cmap(0) == matplotlib.colormaps["Dark2"](0)


# ---------------------------------------------------------------------------
# plot_row_partitions coloring
# ---------------------------------------------------------------------------


def test_row_partitions_colors_by_numeric_feature(embedded_adata):
    fig = plot_row_partitions(
        embedded_adata, obs_search_term="condition", colors=["m0"], unit_size=2
    )
    assert isinstance(fig, plt.Figure)


def test_row_partitions_use_color_string_column_verbatim(embedded_adata):
    fig = plot_row_partitions(
        embedded_adata,
        obs_search_term="condition",
        colors=["phase_colors"],
        plot_background=False,
        unit_size=2,
    )
    all_column = _cell_facecolors(fig, 2)  # columns: ctrl, drug, ALL
    g2 = (embedded_adata.obs["phase"] == "G2").to_numpy()
    np.testing.assert_allclose(
        all_column[g2][:, :3], [mcolors.to_rgb("dodgerblue")] * g2.sum()
    )


def test_row_partitions_color_categorical_column_from_uns_palette(embedded_adata):
    embedded_adata.uns["phase_colors"] = RGB_PALETTE
    fig = plot_row_partitions(
        embedded_adata,
        obs_search_term="condition",
        colors=["phase"],
        plot_background=False,
        unit_size=2,
    )

    # ALL column: every S cell is green
    all_column = _cell_facecolors(fig, 2)
    is_s = (embedded_adata.obs["phase"] == "S").to_numpy()
    np.testing.assert_allclose(all_column[is_s][:, :3], [[0, 1, 0]] * is_s.sum())

    # ctrl column: the subset keeps the same per-category colors
    ctrl = embedded_adata[embedded_adata.obs["condition"] == "ctrl"]
    ctrl_column = _cell_facecolors(fig, 0)
    ctrl_g2 = (ctrl.obs["phase"] == "G2").to_numpy()
    np.testing.assert_allclose(ctrl_column[ctrl_g2][:, :3], [[0, 0, 1]] * ctrl_g2.sum())


def test_row_partitions_color_categorical_column_with_default_palette(embedded_adata):
    fig = plot_row_partitions(
        embedded_adata,
        obs_search_term="condition",
        colors=["phase"],
        plot_background=False,
        unit_size=2,
    )
    all_column = _cell_facecolors(fig, 2)
    colors_per_phase = {
        ph: {tuple(c) for c in all_column[(embedded_adata.obs["phase"] == ph).to_numpy()]}
        for ph in ["G1", "S", "G2"]
    }
    assert all(len(colors) == 1 for colors in colors_per_phase.values())
    assert len({next(iter(colors)) for colors in colors_per_phase.values()}) == 3


# ---------------------------------------------------------------------------
# get_legend
# ---------------------------------------------------------------------------


def test_legend_for_color_string_column(embedded_adata):
    patches, _ = get_legend(embedded_adata, "phase_colors")
    got = {p.get_label(): mcolors.to_hex(p.get_facecolor()) for p in patches}
    assert got == {
        "G1": mcolors.to_hex("orange"),
        "S": mcolors.to_hex("orchid"),
        "G2": mcolors.to_hex("dodgerblue"),
    }


def test_legend_for_categorical_column_matches_plot_colors(embedded_adata):
    embedded_adata.uns["phase_colors"] = RGB_PALETTE
    patches, _ = get_legend(embedded_adata, "phase")
    got = {p.get_label(): mcolors.to_hex(p.get_facecolor()) for p in patches}
    assert got == {"G1": "#ff0000", "S": "#00ff00", "G2": "#0000ff"}


# ---------------------------------------------------------------------------
# combine_figures_with_gridspec
# ---------------------------------------------------------------------------


def _scatter_figure(color):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter([0, 1, 2], [0, 1, 2], s=400, c=color)
    return fig


def _has_color(region, rgb):
    return bool((np.abs(region - np.array(rgb)).sum(axis=-1) < 60).any())


def test_combine_figures_draws_each_source_in_its_own_cell():
    figures = [_scatter_figure("red"), _scatter_figure("blue")]

    combined = combine_figures_with_gridspec(figures, grid_rows=1, grid_cols=2, unit_size=3)

    combined.canvas.draw()
    img = np.asarray(combined.canvas.buffer_rgba())[..., :3].astype(int)
    left, right = img[:, : img.shape[1] // 2], img[:, img.shape[1] // 2 :]
    assert _has_color(left, (255, 0, 0)) and not _has_color(left, (0, 0, 255))
    assert _has_color(right, (0, 0, 255)) and not _has_color(right, (255, 0, 0))


def test_combine_figures_leaves_source_figures_intact():
    source = _scatter_figure("red")

    combine_figures_with_gridspec([source], grid_rows=1, grid_cols=1, unit_size=3)

    assert len(source.axes[0].collections) == 1
