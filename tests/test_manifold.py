"""Tests for PHATE plotting and hyperparameter search in cc_mapping.manifold."""

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# cc_mapping.manifold needs the optional phate dependency; skip the module without it
pytest.importorskip("phate")

from cc_mapping.manifold import (
    PHATEConfig,
    PHATEHyperparamGrid,
    PHATEVisualizer,
    perform_phate_hyperparameter_search,
)

matplotlib.use("agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def phase_adata():
    """200 cells, 6 markers, categorical phase, a scaled layer and a feature set."""
    rng = np.random.default_rng(0)
    n_cells, n_markers = 200, 6
    phase = rng.choice(["G1", "S", "G2"], size=n_cells)
    X = rng.normal(size=(n_cells, n_markers))
    for k, ph in enumerate(["G1", "S", "G2"]):
        X[phase == ph, k] += 3.0
    obs = pd.DataFrame(
        {"phase": pd.Categorical(phase, categories=["G1", "S", "G2"])},
        index=np.arange(n_cells).astype(str),
    )
    adata = ad.AnnData(
        X=X, obs=obs, var=pd.DataFrame(index=[f"m{i}" for i in range(n_markers)])
    )
    adata.layers["scaled"] = (X - X.mean(axis=0)) / X.std(axis=0)
    adata.var["fs"] = True
    adata.obsm["X_phate"] = rng.normal(size=(n_cells, 2))
    return adata


@pytest.fixture
def small_grid():
    return PHATEHyperparamGrid(
        row_param_name="gamma",
        col_param_name="t",
        constant_param_name="knn",
        row_param_values=[0, 1],
        col_param_values=[5, 10],
        constant_param_values=[5, 10],
    )


BASE_CONFIG = PHATEConfig(n_pca=5, n_jobs=1, random_state=0)


def test_plot_from_adata_colors_by_categorical_column(phase_adata):
    phase_adata.uns["phase_colors"] = ["#ff0000", "#00ff00", "#0000ff"]

    ax = PHATEVisualizer.plot_from_adata(phase_adata, "phase")

    facecolors = ax.collections[0].get_facecolors()
    is_s = (phase_adata.obs["phase"] == "S").to_numpy()
    np.testing.assert_allclose(facecolors[is_s][:, :3], [[0, 1, 0]] * is_s.sum())


def test_hyperparameter_search_draws_legend_on_every_grid(phase_adata, small_grid):
    figures = perform_phate_hyperparameter_search(
        phase_adata,
        "fs",
        "scaled",
        small_grid,
        BASE_CONFIG,
        color_name="phase",
        final_grid_dims=None,
        unit_size=2,
        show_legend=True,
    )

    assert len(figures) == 2
    for fig in figures:
        assert any(ax.get_legend() is not None for ax in fig.axes)


def test_hyperparameter_search_combines_grids_when_final_grid_dims_given(
    phase_adata, small_grid, tmp_path
):
    out = tmp_path / "search.png"

    figures = perform_phate_hyperparameter_search(
        phase_adata,
        "fs",
        "scaled",
        small_grid,
        BASE_CONFIG,
        color_name="phase",
        final_grid_dims=(1, 2),
        unit_size=2,
        save_path=str(out),
    )

    assert len(figures) == 1
    assert out.exists()


def test_hyperparameter_search_does_not_leave_figures_open(phase_adata, small_grid):
    """Returned figures must not stay registered with pyplot, or the next plt.show()
    in a notebook displays them a second time."""
    plt.close("all")

    figures = perform_phate_hyperparameter_search(
        phase_adata,
        "fs",
        "scaled",
        small_grid,
        BASE_CONFIG,
        color_name="phase",
        final_grid_dims=(1, 2),
        unit_size=2,
    )

    assert plt.get_fignums() == []
    assert figures[0].axes
