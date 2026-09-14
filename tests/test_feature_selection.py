"""Tests for cc_mapping.feature_selection as a replacement for
cc_mapping.core.random_forest_feature_selection.

The equivalence tests run both implementations with the same random_state,
rf_params and train/test split, and require the same features in the same
order with the same accuracy curve.
"""

import contextlib
import io
import re
from dataclasses import dataclass

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cc_mapping import core as cc_core
from cc_mapping.feature_selection import (
    RFMinMaxSelector,
    RFTopNSelector,
    prepare_feature_matrix,
)

# The reference: core.random_forest_feature_selection as it was before it became
# a wrapper around the selectors.
from tests.fixtures.legacy_core import random_forest_feature_selection

matplotlib.use("agg")

FAST_RF = {"n_estimators": 50, "min_samples_leaf": 5, "n_jobs": 1}
PHASES = ["G1", "S", "G2"]

# Non-default split settings, passed identically to both implementations.
SHARED = {"random_state": 7, "train_test_split_params": {"test_size": 0.3}}

# With SHARED on graded_adata, core's "increment" cutoff stops at 4 features and
# "jump" at 5 (checked by test_graded_adata_separates_the_cutoff_methods).
THRESHOLD = 0.03
STABLE = 3


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _phase_adata(seed, shifts, n_cells=600):
    """Cells x len(shifts) markers; marker_i is shifted by shifts[i] in one phase."""
    rng = np.random.default_rng(seed)
    phase = rng.choice(PHASES, size=n_cells)
    X = rng.normal(size=(n_cells, len(shifts)))
    for i, shift in enumerate(shifts):
        X[phase == PHASES[i % 3], i] += shift
    obs = pd.DataFrame(
        {"phase": pd.Categorical(phase)}, index=np.arange(n_cells).astype(str)
    )
    var = pd.DataFrame(index=[f"marker_{i}" for i in range(len(shifts))])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def informative_adata():
    """600 cells x 10 markers where only marker_0/1/2 separate the three phases."""
    return _phase_adata(0, [3.0, 3.0, 3.0] + [0.0] * 7)


@pytest.fixture
def graded_adata():
    """600 cells x 10 markers; marker_0..4 carry decreasing phase signal, the rest are noise."""
    return _phase_adata(2, [2.0, 1.5, 1.0, 0.6, 0.3] + [0.0] * 5)


@dataclass
class CoreRun:
    selected: list  # most important first
    accuracy_curve: np.ndarray | None  # RF_min_max only
    var_column: pd.Series


@pytest.fixture
def run_core(monkeypatch):
    """Run the core function and recover what it only prints or plots.

    It returns just the AnnData, so the selection order is parsed from its
    verbose printout and the accuracy curve is read off the plot it shows.
    """
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(plt.gcf()))

    def run(adata, features, method, **kwargs):
        shown.clear()
        printout = io.StringIO()
        with contextlib.redirect_stdout(printout):
            random_forest_feature_selection(
                adata,
                features,
                "phase",
                method=method,
                feature_set_name="fs",
                plot=True,
                show=True,
                verbose=True,
                rf_params=FAST_RF,
                **kwargs,
            )
        final_set = printout.getvalue().split(
            "Optimal Feature Set sorted by RF feature importance"
        )[1]
        curve = None
        if shown:
            curve = np.asarray(shown[0].axes[0].get_lines()[0].get_ydata())
        plt.close("all")
        return CoreRun(re.findall(r"'([^']*)'", final_set), curve, adata.var["fs"])

    return run


def _minmax(**kwargs):
    return RFMinMaxSelector(rf_params=FAST_RF, verbose=False, **kwargs)


def _topn(n_features, **kwargs):
    return RFTopNSelector(
        n_features=n_features, rf_params=FAST_RF, verbose=False, **kwargs
    )


def assert_same_selection(selector, core):
    assert list(selector.results.selected_features) == core.selected
    if core.accuracy_curve is not None:
        np.testing.assert_array_equal(selector.accuracy_curve_, core.accuracy_curve)


# ---------------------------------------------------------------------------
# Equivalence with core.random_forest_feature_selection
# ---------------------------------------------------------------------------


def test_graded_adata_separates_the_cutoff_methods(graded_adata, run_core):
    """Precondition for test_minmax_matches_core_rf_min_max: the two cutoff
    methods disagree on this data, so matching core under both is only possible
    if the selector implements each one."""
    names = list(graded_adata.var_names)
    settings = dict(threshold=THRESHOLD, stable_counter=STABLE, **SHARED)

    increment = run_core(
        graded_adata.copy(), names, "RF_min_max", cutoff_method="increment", **settings
    )
    jump = run_core(
        graded_adata.copy(), names, "RF_min_max", cutoff_method="jump", **settings
    )

    assert increment.selected != jump.selected


@pytest.mark.parametrize("cutoff_method", ["increment", "jump"])
def test_minmax_matches_core_rf_min_max(graded_adata, run_core, cutoff_method):
    names = list(graded_adata.var_names)

    core = run_core(
        graded_adata.copy(),
        names,
        "RF_min_max",
        cutoff_method=cutoff_method,
        threshold=THRESHOLD,
        stable_counter=STABLE,
        **SHARED,
    )
    selector = _minmax(
        cutoff_method=cutoff_method,
        threshold=THRESHOLD,
        stable_iterations=STABLE,
        **SHARED,
    ).fit_adata(graded_adata.copy(), names, "phase")

    assert_same_selection(selector, core)


@pytest.mark.parametrize("n_features", [3, 5])
def test_topn_matches_core_rf_min_n(informative_adata, run_core, n_features):
    names = list(informative_adata.var_names)

    core = run_core(informative_adata.copy(), names, f"RF_min_{n_features}", **SHARED)
    selector = _topn(n_features, **SHARED).fit_adata(
        informative_adata.copy(), names, "phase"
    )

    assert_same_selection(selector, core)


@pytest.mark.parametrize(
    "make_selector",
    [lambda: _minmax(threshold=THRESHOLD, stable_iterations=STABLE), lambda: _topn(5)],
    ids=["minmax", "topn"],
)
def test_shuffled_feature_list_gives_same_selection(graded_adata, make_selector):
    names = list(graded_adata.var_names)
    shuffled = list(np.random.default_rng(1).permutation(names))
    assert shuffled != names

    in_order = make_selector().fit_adata(graded_adata.copy(), names, "phase")
    out_of_order = make_selector().fit_adata(graded_adata.copy(), shuffled, "phase")

    assert list(out_of_order.results.selected_features) == list(
        in_order.results.selected_features
    )
    np.testing.assert_array_equal(
        out_of_order.results.metadata["sorted_feature_names"],
        in_order.results.metadata["sorted_feature_names"],
    )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_unknown_feature_name_raises(informative_adata):
    with pytest.raises(ValueError, match="not_a_marker"):
        _topn(2).fit_adata(
            informative_adata, ["marker_0", "marker_1", "not_a_marker"], "phase"
        )


@pytest.mark.parametrize("label_dtype", ["category", "object"])
def test_nan_labels_are_dropped_like_core(informative_adata, run_core, label_dtype):
    labels = informative_adata.obs["phase"].astype(object).to_numpy()
    labels[:10] = np.nan
    labels[10:20] = "nan"
    informative_adata.obs["phase"] = pd.Series(
        labels, index=informative_adata.obs_names, dtype=label_dtype
    )
    names = list(informative_adata.var_names)

    prepared = prepare_feature_matrix(informative_adata, names, "phase", verbose=False)
    np.testing.assert_array_equal(prepared.dropped_indices, np.arange(20))

    core = run_core(informative_adata.copy(), names, "RF_min_max", **SHARED)
    selector = _minmax(**SHARED).fit_adata(informative_adata.copy(), names, "phase")
    assert_same_selection(selector, core)


def test_nan_and_inf_rows_are_dropped_like_core(informative_adata, run_core):
    informative_adata.X[0, 0] = np.nan
    informative_adata.X[1, 1] = np.inf
    informative_adata.X[2, 2] = -np.inf
    names = list(informative_adata.var_names)

    prepared = prepare_feature_matrix(informative_adata, names, "phase", verbose=False)
    np.testing.assert_array_equal(prepared.dropped_indices, [0, 1, 2])

    core = run_core(informative_adata.copy(), names, "RF_min_max", **SHARED)
    selector = _minmax(**SHARED).fit_adata(informative_adata.copy(), names, "phase")
    assert_same_selection(selector, core)


def test_boolean_var_key_selects_like_the_list_of_names(informative_adata, run_core):
    candidates = ["marker_0", "marker_1", "marker_2", "marker_5", "marker_7"]
    informative_adata.var["candidates"] = informative_adata.var_names.isin(candidates)

    core = run_core(informative_adata.copy(), candidates, "RF_min_max", **SHARED)
    selector = _minmax(**SHARED).fit_adata(
        informative_adata.copy(), "candidates", "phase"
    )

    assert_same_selection(selector, core)


def test_sparse_X_selects_like_dense_X(informative_adata):
    """core.random_forest_feature_selection cannot read a sparse X at all, so the
    reference here is the selector's own result on the dense matrix."""
    names = list(informative_adata.var_names)
    sparse_adata = informative_adata.copy()
    sparse_adata.X = sp.csr_matrix(sparse_adata.X)

    dense = _minmax(**SHARED).fit_adata(informative_adata, names, "phase")
    sparse = _minmax(**SHARED).fit_adata(sparse_adata, names, "phase")

    assert list(sparse.results.selected_features) == list(
        dense.results.selected_features
    )
    np.testing.assert_array_equal(sparse.accuracy_curve_, dense.accuracy_curve_)


# ---------------------------------------------------------------------------
# Writing the selection back to adata.var
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, make_selector",
    [
        ("RF_min_max", lambda: _minmax(**SHARED)),
        ("RF_min_5", lambda: _topn(5, **SHARED)),
    ],
    ids=["minmax", "topn"],
)
def test_transform_adata_writes_the_core_var_column(
    informative_adata, run_core, method, make_selector
):
    names = list(informative_adata.var_names)
    core = run_core(informative_adata.copy(), names, method, **SHARED)

    fs_adata = informative_adata.copy()
    returned = (
        make_selector()
        .fit_adata(fs_adata, names, "phase")
        .transform_adata(fs_adata, var_key="fs")
    )

    assert returned is fs_adata
    pd.testing.assert_series_equal(fs_adata.var["fs"], core.var_column)


# ---------------------------------------------------------------------------
# Accuracy plot
# ---------------------------------------------------------------------------


def test_plot_accuracy_curve_is_shown_by_default(informative_adata, monkeypatch):
    selector = _minmax().fit_adata(
        informative_adata, list(informative_adata.var_names), "phase"
    )
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(plt.gcf()))

    fig = selector.plot_accuracy_curve()

    assert isinstance(fig, plt.Figure)
    assert shown == [fig]
    np.testing.assert_array_equal(
        fig.axes[0].get_lines()[0].get_ydata(), selector.accuracy_curve_
    )


def test_plot_accuracy_curve_can_be_saved_without_showing(informative_adata, tmp_path):
    selector = _minmax().fit_adata(
        informative_adata, list(informative_adata.var_names), "phase"
    )
    plt.close("all")
    out = tmp_path / "accuracy.png"

    fig = selector.plot_accuracy_curve(save_path=out, show=False)

    assert isinstance(fig, plt.Figure)
    assert out.exists()
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------


def test_verbose_false_prints_nothing(informative_adata, capsys):
    _minmax().fit_adata(informative_adata, list(informative_adata.var_names), "phase")

    assert capsys.readouterr() == ("", "")


def test_verbose_true_reports_training_and_selection(informative_adata, capsys):
    selector = RFMinMaxSelector(rf_params=FAST_RF, verbose=True).fit_adata(
        informative_adata, list(informative_adata.var_names), "phase"
    )

    out = capsys.readouterr().out
    assert "Classification Report" in out
    assert f"Selected {selector.results.n_features_selected} features" in out


# ---------------------------------------------------------------------------
# Deliberate differences from core
# ---------------------------------------------------------------------------


def test_topn_rejects_more_features_than_available(informative_adata):
    """core's RF_min_20 on 10 features silently keeps all 10."""
    with pytest.raises(ValueError, match="cannot be greater"):
        _topn(20).fit_adata(
            informative_adata, list(informative_adata.var_names), "phase"
        )


def test_minmax_falls_back_to_best_prefix_when_no_step_clears_threshold(
    informative_adata,
):
    """core ends up training on 0 features here and raises a ValueError."""
    selector = _minmax(threshold=1.0).fit_adata(
        informative_adata, list(informative_adata.var_names), "phase"
    )

    assert selector.results.n_features_selected >= 1
    assert selector.results.n_features_selected == np.argmax(selector.accuracy_curve_)


# ---------------------------------------------------------------------------
# Persistence (skops)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_selector",
    [lambda: _minmax(**SHARED), lambda: _topn(5, **SHARED)],
    ids=["minmax", "topn"],
)
def test_save_load_round_trip(informative_adata, tmp_path, make_selector):
    pytest.importorskip("skops")
    selector = make_selector().fit_adata(
        informative_adata, list(informative_adata.var_names), "phase"
    )
    path = tmp_path / "selector.skops"

    selector.save(path)
    loaded = type(selector).load(path)

    assert type(loaded) is type(selector)
    assert loaded.get_params() == selector.get_params()
    np.testing.assert_array_equal(
        loaded.get_feature_names_out(), selector.get_feature_names_out()
    )
    np.testing.assert_array_equal(loaded.get_support(), selector.get_support())
    probe = np.random.default_rng(0).normal(size=(20, selector.model_.n_features_in_))
    np.testing.assert_array_equal(
        loaded.model_.predict(probe), selector.model_.predict(probe)
    )


def test_loaded_minmax_selector_plots_the_same_curve(informative_adata, tmp_path):
    pytest.importorskip("skops")
    selector = _minmax(**SHARED).fit_adata(
        informative_adata, list(informative_adata.var_names), "phase"
    )
    path = tmp_path / "selector.skops"
    selector.save(path)

    fig = RFMinMaxSelector.load(path).plot_accuracy_curve(show=False)

    np.testing.assert_array_equal(
        fig.axes[0].get_lines()[0].get_ydata(), selector.accuracy_curve_
    )


# ---------------------------------------------------------------------------
# core.random_forest_feature_selection, now a deprecated wrapper
# ---------------------------------------------------------------------------

ignore_core_deprecation = pytest.mark.filterwarnings(
    "ignore:random_forest_feature_selection is deprecated:DeprecationWarning"
)


@ignore_core_deprecation
@pytest.mark.parametrize(
    "method, settings",
    [
        ("RF_min_max", {"cutoff_method": "increment"}),
        ("RF_min_max", {"cutoff_method": "jump"}),
        ("RF_min_5", {}),
    ],
    ids=["increment", "jump", "top5"],
)
def test_core_wrapper_matches_the_legacy_implementation(
    graded_adata, run_core, monkeypatch, method, settings
):
    names = list(graded_adata.var_names)
    settings = dict(settings, threshold=THRESHOLD, stable_counter=STABLE, **SHARED)
    legacy = run_core(graded_adata.copy(), names, method, **settings)

    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(plt.gcf()))
    wrapped = graded_adata.copy()
    cc_core.random_forest_feature_selection(
        wrapped,
        names,
        "phase",
        method=method,
        feature_set_name="fs",
        verbose=False,
        rf_params=FAST_RF,
        **settings,
    )

    pd.testing.assert_series_equal(wrapped.var["fs"], legacy.var_column)
    if legacy.accuracy_curve is not None:
        np.testing.assert_array_equal(
            shown[0].axes[0].get_lines()[0].get_ydata(), legacy.accuracy_curve
        )


def test_core_wrapper_warns_that_it_is_deprecated(informative_adata):
    with pytest.warns(DeprecationWarning, match="RFTopNSelector"):
        cc_core.random_forest_feature_selection(
            informative_adata,
            list(informative_adata.var_names),
            "phase",
            method="RF_min_3",
            plot=False,
            verbose=False,
            rf_params=FAST_RF,
        )


@ignore_core_deprecation
def test_core_wrapper_keeps_its_default_var_column_name(informative_adata):
    cc_core.random_forest_feature_selection(
        informative_adata,
        list(informative_adata.var_names),
        "phase",
        method="RF_min_3",
        plot=False,
        verbose=False,
        rf_params=FAST_RF,
    )

    assert informative_adata.var["RF_min_3_feature_set"].sum() == 3


@ignore_core_deprecation
def test_core_wrapper_keeps_every_feature_when_n_exceeds_them(informative_adata):
    """RFTopNSelector raises here; the old function silently kept all features."""
    cc_core.random_forest_feature_selection(
        informative_adata,
        list(informative_adata.var_names),
        "phase",
        method="RF_min_20",
        feature_set_name="fs",
        plot=False,
        verbose=False,
        rf_params=FAST_RF,
    )

    assert informative_adata.var["fs"].all()


@ignore_core_deprecation
def test_core_wrapper_rejects_an_unknown_method(informative_adata):
    with pytest.raises(ValueError, match="RF_max"):
        cc_core.random_forest_feature_selection(
            informative_adata,
            list(informative_adata.var_names),
            "phase",
            method="RF_max",
            plot=False,
            verbose=False,
            rf_params=FAST_RF,
        )


def test_train_random_forest_model_warns_that_it_is_deprecated(informative_adata):
    with pytest.warns(DeprecationWarning, match="train_rf_model"):
        cc_core.train_random_forest_model(
            informative_adata.X,
            informative_adata.obs["phase"].to_numpy(),
            rf_params=FAST_RF,
            random_state=0,
            train_test_split_params={"test_size": 0.25},
            verbose=False,
        )
