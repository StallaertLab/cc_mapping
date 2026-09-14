"""
Persistence utilities for saving/loading feature selectors using skops.

Skops provides secure serialization for scikit-learn models, which is safer
than pickle and supports model cards for documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import skops.io as sio
    SKOPS_AVAILABLE = True
except ImportError:
    SKOPS_AVAILABLE = False

if TYPE_CHECKING:
    from ._base import FeatureSelector


@dataclass
class SelectorState:
    """
    Complete state of a fitted FeatureSelector for serialization.
    
    Attributes
    ----------
    class_name : str
        Name of the selector class (e.g., "RFMinMaxSelector").
    module_name : str
        Module where the class is defined.
    hyperparameters : dict
        Selector hyperparameters from get_params().
    model : Any
        The fitted sklearn model.
    selected_features : np.ndarray
        Array of selected feature names.
    feature_importances : np.ndarray
        Array of feature importances.
    n_features_selected : int
        Number of selected features.
    metadata : dict
        Additional metadata from SelectionResult.
    feature_names : np.ndarray
        All feature names used during fitting.
    extra_state : dict
        Any additional state (e.g., accuracy_curve for RFMinMaxSelector).
    """
    class_name: str
    module_name: str
    hyperparameters: dict
    model: Any
    selected_features: np.ndarray
    feature_importances: np.ndarray
    n_features_selected: int
    metadata: dict
    feature_names: np.ndarray
    extra_state: dict


def _check_skops_available() -> None:
    """Raise ImportError if skops is not installed."""
    if not SKOPS_AVAILABLE:
        raise ImportError(
            "skops is required for saving/loading selectors. "
            "Install it with: pip install skops"
        )


def save_selector(selector: "FeatureSelector", path: str | Path) -> None:
    """
    Save a fitted FeatureSelector to disk using skops.
    
    Parameters
    ----------
    selector : FeatureSelector
        A fitted selector instance.
    path : str or Path
        File path to save to. Recommended extension: .skops
        
    Raises
    ------
    ImportError
        If skops is not installed.
    RuntimeError
        If selector has not been fitted.
        
    Examples
    --------
    >>> selector = RFMinMaxSelector()
    >>> selector.fit(X, y, feature_names)
    >>> save_selector(selector, "my_selector.skops")
    """
    _check_skops_available()
    
    if not selector.is_fitted_:
        raise RuntimeError(
            "Cannot save unfitted selector. Call fit() first."
        )
    
    # Collect extra state for specific selector types
    extra_state = {}
    
    # RFMinMaxSelector has additional state
    if hasattr(selector, "accuracy_curve_") and selector.accuracy_curve_ is not None:
        extra_state["accuracy_curve_"] = selector.accuracy_curve_
    if hasattr(selector, "_sorted_feature_names") and selector._sorted_feature_names is not None:
        extra_state["_sorted_feature_names"] = selector._sorted_feature_names
    
    state = SelectorState(
        class_name=selector.__class__.__name__,
        module_name=selector.__class__.__module__,
        hyperparameters=selector.get_params(),
        model=selector.model_,
        selected_features=selector.results_.selected_features,
        feature_importances=selector.results_.feature_importances,
        n_features_selected=selector.results_.n_features_selected,
        metadata=selector.results_.metadata,
        feature_names=selector.feature_names_,
        extra_state=extra_state,
    )
    
    path = Path(path)
    sio.dump(state, path)


def load_selector(path: str | Path) -> "FeatureSelector":
    """
    Load a fitted FeatureSelector from disk.
    
    Parameters
    ----------
    path : str or Path
        File path to load from.
        
    Returns
    -------
    FeatureSelector
        The loaded and fitted selector instance.
        
    Raises
    ------
    ImportError
        If skops is not installed.
        
    Examples
    --------
    >>> selector = load_selector("my_selector.skops")
    >>> print(selector.results.n_features_selected)
    >>> mask = selector.get_support()
    """
    _check_skops_available()
    
    path = Path(path)
    
    # Define trusted types for skops
    # These are the types we expect in our serialized state
    from sklearn.ensemble import RandomForestClassifier
    from ._base import SelectionResult
    
    trusted_types = [
        SelectorState,
        SelectionResult,
        RandomForestClassifier,
        np.ndarray,
    ]
    
    state: SelectorState = sio.load(path, trusted=trusted_types)
    
    # Dynamically get the selector class
    selector_class = _get_selector_class(state.class_name, state.module_name)
    
    # Create instance with saved hyperparameters
    selector = selector_class(**state.hyperparameters)
    
    # Restore fitted state
    from ._base import SelectionResult
    
    selector.model_ = state.model
    selector.feature_names_ = state.feature_names
    selector.results_ = SelectionResult(
        selected_features=state.selected_features,
        feature_importances=state.feature_importances,
        n_features_selected=state.n_features_selected,
        metadata=state.metadata,
    )
    selector.is_fitted_ = True
    
    # Restore extra state
    for key, value in state.extra_state.items():
        setattr(selector, key, value)
    
    return selector


def _get_selector_class(class_name: str, module_name: str):
    """
    Get selector class by name.
    
    Parameters
    ----------
    class_name : str
        Name of the selector class.
    module_name : str
        Module where the class is defined.
        
    Returns
    -------
    type
        The selector class.
    """
    # Import selector classes
    from ._random_forest import RFTopNSelector, RFMinMaxSelector
    
    class_map = {
        "RFTopNSelector": RFTopNSelector,
        "RFMinMaxSelector": RFMinMaxSelector,
    }
    
    if class_name not in class_map:
        raise ValueError(
            f"Unknown selector class: {class_name}. "
            f"Available: {list(class_map.keys())}"
        )
    
    return class_map[class_name]
