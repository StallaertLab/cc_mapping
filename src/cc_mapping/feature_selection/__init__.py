"""
Feature selection methods for cell cycle mapping.

This module provides modular, scikit-learn-style feature selectors
for identifying informative features from high-dimensional cell data.

Classes
-------
FeatureSelector
    Abstract base class for all feature selectors.
RFTopNSelector
    Select top N features by Random Forest importance.
RFMinMaxSelector
    Find minimum features that maximize classification accuracy.

Functions
---------
prepare_feature_matrix
    Extract and validate feature matrix from AnnData.
validate_data
    Validate data is ready for feature selection.

Data Classes
------------
SelectionResult
    Container for feature selection results.
PreparedData
    Container for prepared feature matrix and metadata.

Notes
-----
``cc_mapping.core.random_forest_feature_selection`` is deprecated. The
selectors use the same defaults and, given the same arguments, choose the
same features in the same order. For example::

    # before
    core.random_forest_feature_selection(
        adata, adata.var_names, "phase", method="RF_min_max",
        threshold=0.05, stable_counter=5, feature_set_name="fs",
    )

    # after
    selector = RFMinMaxSelector(threshold=0.05, stable_iterations=5)
    selector.fit_adata(adata, adata.var_names, "phase")
    selector.transform_adata(adata, var_key="fs")
    fig = selector.plot_accuracy_curve()

.. list-table:: Arguments of ``random_forest_feature_selection`` and their replacements
   :header-rows: 1

   * - ``random_forest_feature_selection``
     - ``cc_mapping.feature_selection``
   * - ``method="RF_min_max"``
     - ``RFMinMaxSelector()``
   * - ``method="RF_min_<N>"``
     - ``RFTopNSelector(n_features=N)``, which raises if N exceeds the number
       of features (the old function kept them all)
   * - ``stable_counter``
     - ``stable_iterations``
   * - ``threshold``, ``cutoff_method``, ``random_state``, ``rf_params``,
       ``train_test_split_params``, ``verbose``
     - same names and defaults
   * - ``training_feature_set``, ``training_labels``
     - ``fit_adata(adata, feature_set_key, label_key)``
   * - ``feature_set_name``
     - ``transform_adata(adata, var_key=...)``; the default key is
       ``"selected_features"`` instead of ``f"{method}_feature_set"``
   * - ``plot``, ``save_path``, ``show``
     - ``plot_accuracy_curve(save_path=..., show=...)``, called explicitly

Examples
--------
Basic usage with numpy arrays:

>>> from cc_mapping.feature_selection import RFMinMaxSelector, validate_data
>>>
>>> validate_data(X, y, feature_names)  # Check for NaN/inf
>>> selector = RFMinMaxSelector(threshold=0.01)
>>> selector.fit(X, y, feature_names)
>>>
>>> print(f"Selected {selector.results.n_features_selected} features")
>>> mask = selector.get_support()

Usage with AnnData:

>>> from cc_mapping.feature_selection import prepare_feature_matrix, RFTopNSelector
>>>
>>> prepared = prepare_feature_matrix(
...     adata,
...     feature_set="intensity_features",
...     labels="cell_cycle_phase",
... )
>>>
>>> selector = RFTopNSelector(n_features=30)
>>> selector.fit(prepared.X, prepared.y, prepared.feature_names)
>>>
>>> # Add selection to adata
>>> adata = selector.transform_adata(adata, var_key="selected_features")

Saving and loading:

>>> selector.save("my_selector.skops")
>>> loaded = RFMinMaxSelector.load("my_selector.skops")
"""

from ._base import FeatureSelector, SelectionResult
from ._plotting import plot_accuracy_curve, plot_feature_importances
from ._preprocessing import (
    PreparedData,
    prepare_feature_matrix,
    validate_data,
)
from ._random_forest import RFMinMaxSelector, RFTopNSelector
from ._training import TrainingResult, train_rf_model

__all__ = [
    # Base classes
    "FeatureSelector",
    "SelectionResult",
    # Preprocessing
    "prepare_feature_matrix",
    "validate_data",
    "PreparedData",
    # Selectors
    "RFTopNSelector",
    "RFMinMaxSelector",
    # Plotting
    "plot_accuracy_curve",
    "plot_feature_importances",
    # Training utilities
    "train_rf_model",
    "TrainingResult",
]
