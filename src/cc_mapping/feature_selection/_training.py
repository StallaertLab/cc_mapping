"""
Model training utilities for feature selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Default parameters for Random Forest
DEFAULT_RF_PARAMS = {
    "min_samples_leaf": 50,
    "n_estimators": 150,
    "bootstrap": True,
    "oob_score": True,
    "n_jobs": -1,
}

DEFAULT_SPLIT_PARAMS = {
    "test_size": 0.25,
}


@dataclass
class TrainingResult:
    """
    Container for model training results.
    
    Attributes
    ----------
    model : RandomForestClassifier
        The trained model.
    accuracy : float
        Accuracy on the test set.
    predictions : np.ndarray
        Predictions on the test set.
    test_labels : np.ndarray
        True labels for the test set.
    classification_report : str
        Full classification report as string.
    """
    model: RandomForestClassifier
    accuracy: float
    predictions: np.ndarray
    test_labels: np.ndarray
    classification_report: str


def train_rf_model(
    X: np.ndarray,
    y: np.ndarray,
    rf_params: dict | None = None,
    train_test_split_params: dict | None = None,
    random_state: int = 42,
    verbose: bool = True,
    description: str = "",
) -> TrainingResult:
    """
    Train a Random Forest classifier and evaluate on held-out test set.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    rf_params : dict, optional
        Parameters for RandomForestClassifier. 
        Defaults to DEFAULT_RF_PARAMS.
    train_test_split_params : dict, optional
        Parameters for train_test_split.
        Defaults to DEFAULT_SPLIT_PARAMS.
    random_state : int, default=42
        Random state for reproducibility.
    verbose : bool, default=True
        Whether to print classification report.
    description : str, optional
        Description for verbose output.
        
    Returns
    -------
    TrainingResult
        Container with model, accuracy, and evaluation details.
        
    Examples
    --------
    >>> result = train_rf_model(X, y, verbose=True, description="initial")
    >>> print(f"Accuracy: {result.accuracy:.2%}")
    >>> model = result.model
    """
    if rf_params is None:
        rf_params = DEFAULT_RF_PARAMS.copy()
    if train_test_split_params is None:
        train_test_split_params = DEFAULT_SPLIT_PARAMS.copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, **train_test_split_params
    )
    
    # Train model
    model = RandomForestClassifier(random_state=random_state, **rf_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    # zero_division=0 gives the same 0.0 scores as the default, without warning
    # about classes the forest never predicts (the report is built even when quiet)
    report = metrics.classification_report(y_test, predictions, zero_division=0)
    
    if verbose:
        desc_str = f" ({description})" if description else ""
        print(f"Classification Report{desc_str}")
        print("=" * 60)
        print(report)
        print(f"Accuracy: {accuracy:.4f}")
        print()
    
    return TrainingResult(
        model=model,
        accuracy=accuracy,
        predictions=predictions,
        test_labels=y_test,
        classification_report=report,
    )


def get_sorted_feature_indices(
    model: RandomForestClassifier,
) -> np.ndarray:
    """
    Get feature indices sorted by importance (highest first).
    
    Parameters
    ----------
    model : RandomForestClassifier
        Fitted Random Forest model.
        
    Returns
    -------
    np.ndarray
        Indices that would sort features by decreasing importance.
    """
    return np.argsort(-model.feature_importances_)
