"""Tests for plotting functionality in GMMThresholding and SequentialGMM.

This module tests the plotting methods to ensure they correctly visualize
GMM components, decision boundaries, and categorical assignments with proper
color mappings.

The colour tests use a GMM whose middle component (B) is never the most likely
one: samples below 149.5 are labelled A and samples above it C, so every part of
the plot above 149.5 must be drawn in C's colour, never in B's.
"""

import warnings

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cc_mapping.thresholding import GMMThresholding, SequentialGMM
from tests.helpers import probs_from_winners, use_fake_gaussian_mixture

matplotlib.use("agg")

FEATURE = "marker"
LABELS = ["A", "B", "C"]
FEATURE_VALUES = np.arange(300, dtype=float)
B_NEVER_WINS = [0] * 150 + [2] * 150
MEANS = [75.0, 150.0, 225.0]  # B's mean lies above the threshold, inside C's interval
A_COLOUR, B_COLOUR, C_COLOUR = plt.get_cmap("rainbow")(np.linspace(0, 1, 3))

# TODO: Add tests for:
# - Plotting with collapsed labels
# - Plotting with manual thresholds
# - Color consistency across plot types


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def b_never_wins_adata(monkeypatch):
    use_fake_gaussian_mixture(
        monkeypatch,
        FEATURE_VALUES,
        probs_from_winners(B_NEVER_WINS, n_components=3),
        means=MEANS,
        variance=400.0,
    )
    adata = ad.AnnData(X=FEATURE_VALUES.reshape(-1, 1))
    adata.var_names = [FEATURE]
    adata.obs_names = [f"sample_{i}" for i in range(adata.n_obs)]
    adata.obs["group"] = "to_refine"
    return adata


def _categorized(adata):
    gmm = GMMThresholding(adata=adata, feature=FEATURE, label_obs_save_str="labels")
    gmm.fit(n_components=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the empty-label warning for B is expected
        gmm.categorize_samples(ordered_labels=LABELS)
    return gmm


def _assert_background_along(background, positions):
    """Well away from the 149.5 threshold, the shading is A's colour below and C's above."""
    assert np.allclose(background[positions < 140], A_COLOUR)
    assert np.allclose(background[positions > 160], C_COLOUR)


def test_hist_background_takes_the_colour_of_each_interval_label(b_never_wins_adata):
    ax = _categorized(b_never_wins_adata).plot_hist_distribution_with_boundaries()

    background = ax.images[0].get_array()  # (2, resolution, RGBA); columns run along x
    x = np.linspace(*ax.get_xlim(), background.shape[1])
    _assert_background_along(background[0], x)


def test_gmm_curves_and_mean_lines_take_the_colour_of_the_interval_they_fall_in(
    b_never_wins_adata,
):
    ax = _categorized(b_never_wins_adata).plot_hist_distribution_with_boundaries()

    mean_line_colours = {
        line.get_xdata()[0]: line.get_color()
        for line in ax.lines
        if line.get_linestyle() == "--"
    }
    assert np.allclose(mean_line_colours[75.0], A_COLOUR)
    assert np.allclose(mean_line_colours[150.0], C_COLOUR)
    assert np.allclose(mean_line_colours[225.0], C_COLOUR)

    for curve in ax.collections:
        x = curve.get_offsets()[:, 0]
        colours = curve.get_facecolors()
        assert np.allclose(colours[x < 149.5], A_COLOUR)
        assert np.allclose(colours[x >= 149.5], C_COLOUR)


def test_threshold_line_blends_the_colours_of_the_intervals_on_either_side(
    b_never_wins_adata,
):
    ax = _categorized(b_never_wins_adata).plot_hist_distribution_with_boundaries()

    (threshold_line,) = [line for line in ax.lines if line.get_linestyle() == "-"]
    assert threshold_line.get_xdata()[0] == 149.5
    assert np.allclose(threshold_line.get_color(), (A_COLOUR + C_COLOUR) / 2)


def test_strip_plot_background_takes_the_colour_of_each_interval_label(
    b_never_wins_adata,
):
    gmm = _categorized(b_never_wins_adata)
    fig = gmm.plot_strip_plot_histogram_with_decision_boundaries(scatter_density=False)

    (hist_ax,) = [ax for ax in fig.axes if ax.images]
    # The (resolution, 2, RGBA) background has one row per point along y
    background = hist_ax.images[0].get_array()
    y = np.linspace(*hist_ax.get_ylim(), background.shape[0])
    _assert_background_along(background[:, 0], y)


def test_sequential_hist_plot_uses_the_stored_interval_labels(b_never_wins_adata):
    seq_gmm = SequentialGMM(adata=b_never_wins_adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the empty-label warning for B is expected
        seq_gmm.refine_labels_with_gmm(
            feature=FEATURE,
            obs_label="group",
            value_to_refine="to_refine",
            n_components=3,
            ordered_labels=LABELS,
            operation_name="refine",
        )

    ax = seq_gmm.plot_hist_distribution_with_boundaries("refine")

    background = ax.images[0].get_array()
    x = np.linspace(*ax.get_xlim(), background.shape[1])
    _assert_background_along(background[0], x)
