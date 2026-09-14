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
from ._preprocessing import (
    prepare_feature_matrix,
    validate_data,
    PreparedData,
)
from ._random_forest import RFTopNSelector, RFMinMaxSelector
from ._plotting import plot_accuracy_curve, plot_feature_importances
from ._training import train_rf_model, TrainingResult

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
