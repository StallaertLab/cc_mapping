"""
Base classes and data structures for feature selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass
class SelectionResult:
    """
    Container for feature selection results.

    Attributes
    ----------
    selected_features : np.ndarray
        Names of selected features, sorted by importance.
    feature_importances : np.ndarray
        Importance scores for selected features (same order as selected_features).
    n_features_selected : int
        Number of features selected.
    metadata : dict
        Method-specific metadata (e.g., accuracy curve for RF methods).
    """

    selected_features: np.ndarray
    feature_importances: np.ndarray
    n_features_selected: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SelectionResult(n_features={self.n_features_selected}, "
            f"top_features={list(self.selected_features[:3])}...)"
        )


class FeatureSelector(ABC):
    """
    Abstract base class for feature selection methods.

    All feature selectors should inherit from this class and implement
    the required abstract methods.

    Attributes
    ----------
    model_ : Any
        The fitted model (e.g., RandomForestClassifier). Available after fit().
    results_ : SelectionResult
        Detailed selection results. Available after fit().
    feature_names_ : np.ndarray
        Feature names used during fitting. Available after fit().
    is_fitted_ : bool
        Whether the selector has been fitted.
    """

    model_: Any = None
    results_: SelectionResult | None = None
    feature_names_: np.ndarray | None = None
    is_fitted_: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Self:
        """
        Fit the feature selector to the data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target labels of shape (n_samples,).
        feature_names : np.ndarray
            Names of features corresponding to columns in X.

        Returns
        -------
        Self
            The fitted selector instance (for method chaining).
        """
        ...

    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get a mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return indices instead of boolean mask.

        Returns
        -------
        np.ndarray
            Boolean mask of shape (n_features,) or integer indices.

        Raises
        ------
        RuntimeError
            If selector has not been fitted.
        """
        self._check_is_fitted()

        # Create boolean mask based on selected features
        mask = np.isin(self.feature_names_, self.results_.selected_features)

        if indices:
            return np.where(mask)[0]
        return mask

    def get_feature_names_out(self) -> np.ndarray:
        """
        Get names of selected features.

        Returns
        -------
        np.ndarray
            Array of selected feature names.

        Raises
        ------
        RuntimeError
            If selector has not been fitted.
        """
        self._check_is_fitted()
        return self.results_.selected_features.copy()

    @property
    def results(self) -> SelectionResult:
        """
        Get the selection results.

        Returns
        -------
        SelectionResult
            Detailed results from feature selection.

        Raises
        ------
        RuntimeError
            If selector has not been fitted.
        """
        self._check_is_fitted()
        return self.results_

    def fit_adata(
        self,
        adata: ad.AnnData,
        feature_set_key: str | list[str],
        label_key: str,
        drop_na: bool = True,
        drop_inf: bool = True,
    ) -> Self:
        """
        Convenience method to fit directly from AnnData object.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing the data.
        feature_set_key : str or list[str]
            Either a key in adata.var that marks features (boolean column),
            or a list of feature names. Order does not matter: the columns
            always follow adata.var_names.
        label_key : str
            Key in adata.obs containing target labels.
        drop_na : bool, default=True
            Whether to drop rows with NaN values.
        drop_inf : bool, default=True
            Whether to drop rows with infinite values.

        Returns
        -------
        Self
            The fitted selector instance.
        """
        # Import here to avoid circular imports
        from ._preprocessing import prepare_feature_matrix

        prepared = prepare_feature_matrix(
            adata=adata,
            feature_set=feature_set_key,
            labels=label_key,
            drop_na=drop_na,
            drop_inf=drop_inf,
            verbose=getattr(self, "verbose", True),
        )

        return self.fit(prepared.X, prepared.y, prepared.feature_names)

    def transform_adata(
        self,
        adata: ad.AnnData,
        var_key: str = "selected_features",
    ) -> ad.AnnData:
        """
        Add selection mask to adata.var.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to modify.
        var_key : str, default="selected_features"
            Key to use in adata.var for the selection mask.

        Returns
        -------
        ad.AnnData
            Modified AnnData object with selection mask in .var.

        Raises
        ------
        RuntimeError
            If selector has not been fitted.
        """
        self._check_is_fitted()

        # Create boolean mask for all features in adata
        mask = np.isin(adata.var_names.values, self.results_.selected_features)
        adata.var[var_key] = mask

        return adata

    def save(self, path: str | Path) -> None:
        """
        Save the fitted selector to disk using skops.

        Parameters
        ----------
        path : str or Path
            File path to save to.

        Raises
        ------
        RuntimeError
            If selector has not been fitted.
        """
        self._check_is_fitted()
        from ._persistence import save_selector

        save_selector(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load a fitted selector from disk.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        Self
            The loaded selector instance.
        """
        from ._persistence import load_selector

        return load_selector(path)

    @abstractmethod
    def get_params(self) -> dict:
        """
        Get hyperparameters of this selector.

        Returns
        -------
        dict
            Dictionary of hyperparameter names and values.
        """
        ...

    def _check_is_fitted(self) -> None:
        """Raise error if selector is not fitted."""
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit() or fit_adata() first."
            )
