"""Regression tests for cc_mapping.core.random_forest_feature_selection."""

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cc_mapping.core import random_forest_feature_selection

matplotlib.use("agg")

FAST_RF = {"n_estimators": 50, "min_samples_leaf": 5, "n_jobs": 1}


@pytest.fixture
def informative_adata():
    """600 cells x 10 markers where only marker_0/1/2 separate the three phases."""
    rng = np.random.default_rng(0)
    n_cells, n_markers = 600, 10
    phase = rng.choice(["G1", "S", "G2"], size=n_cells)
    X = rng.normal(size=(n_cells, n_markers))
    for k, ph in enumerate(["G1", "S", "G2"]):
        X[phase == ph, k] += 3.0
    obs = pd.DataFrame(
        {"phase": pd.Categorical(phase)}, index=np.arange(n_cells).astype(str)
    )
    var = pd.DataFrame(index=[f"marker_{i}" for i in range(n_markers)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _select_top3(adata, training_features):
    random_forest_feature_selection(
        adata,
        training_features,
        "phase",
        method="RF_min_3",
        feature_set_name="fs",
        plot=False,
        verbose=False,
        rf_params=FAST_RF,
    )
    return set(adata.var_names[adata.var["fs"]])


def test_selected_features_do_not_depend_on_training_list_order(informative_adata):
    names = list(informative_adata.var_names)
    shuffled = list(np.random.default_rng(1).permutation(names))
    assert shuffled != names

    in_order = _select_top3(informative_adata.copy(), names)
    out_of_order = _select_top3(informative_adata.copy(), shuffled)

    assert in_order == {"marker_0", "marker_1", "marker_2"}
    assert out_of_order == in_order


def test_unknown_training_feature_raises(informative_adata):
    with pytest.raises(ValueError, match="not_a_marker"):
        _select_top3(informative_adata, ["marker_0", "marker_1", "not_a_marker"])


def test_accuracy_plot_is_shown_when_plot_is_true(informative_adata, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(plt.gcf()))

    random_forest_feature_selection(
        informative_adata,
        list(informative_adata.var_names),
        "phase",
        method="RF_min_max",
        plot=True,
        verbose=False,
        rf_params=FAST_RF,
    )

    assert len(shown) == 1
    assert shown[0].axes[0].get_ylabel() == "Accuracy"
    plt.close("all")


def test_accuracy_plot_can_be_saved_without_showing(informative_adata, tmp_path):
    plt.close("all")
    out = tmp_path / "accuracy.png"

    random_forest_feature_selection(
        informative_adata,
        list(informative_adata.var_names),
        "phase",
        method="RF_min_max",
        plot=True,
        show=False,
        save_path=str(out),
        verbose=False,
        rf_params=FAST_RF,
    )

    assert out.exists()
    assert plt.get_fignums() == []
