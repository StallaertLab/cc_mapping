"""
Random Forest-based feature selection methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from typing_extensions import Self

from ._base import FeatureSelector, SelectionResult
from ._preprocessing import validate_data
from ._training import (
    DEFAULT_RF_PARAMS,
    DEFAULT_SPLIT_PARAMS,
    get_sorted_feature_indices,
    train_rf_model,
)


class RFTopNSelector(FeatureSelector):
    """
    Select top N features by Random Forest importance.

    This selector trains a Random Forest classifier on all features,
    then selects the N most important features based on feature importances.

    Parameters
    ----------
    n_features : int, default=30
        Number of top features to select.
    rf_params : dict, optional
        Parameters for RandomForestClassifier.
        Defaults to reasonable values for cell data.
    train_test_split_params : dict, optional
        Parameters for train_test_split.
    random_state : int, default=42
        Random state for reproducibility.
    verbose : bool, default=True
        Whether to print progress and results.

    Attributes
    ----------
    model_ : RandomForestClassifier
        The fitted Random Forest model (trained on all features).
    results_ : SelectionResult
        Detailed selection results.
    feature_names_ : np.ndarray
        All feature names used during fitting.

    Examples
    --------
    >>> selector = RFTopNSelector(n_features=30)
    >>> selector.fit(X, y, feature_names)
    >>> print(selector.results.selected_features)
    >>>
    >>> # Get boolean mask for selected features
    >>> mask = selector.get_support()
    >>> X_selected = X[:, mask]
    """

    def __init__(
        self,
        n_features: int = 30,
        rf_params: dict | None = None,
        train_test_split_params: dict | None = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_features = n_features
        self.rf_params = (
            rf_params if rf_params is not None else DEFAULT_RF_PARAMS.copy()
        )
        self.train_test_split_params = (
            train_test_split_params
            if train_test_split_params is not None
            else DEFAULT_SPLIT_PARAMS.copy()
        )
        self.random_state = random_state
        self.verbose = verbose

        # These are set after fitting
        self.model_ = None
        self.results_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: np.ndarray,
    ) -> Self:
        """
        Fit the selector by training RF and selecting top N features.

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
            The fitted selector instance.

        Raises
        ------
        ValueError
            If n_features is greater than the number of input features.
        ValueError
            If data contains NaN or infinite values.
        """
        feature_names = np.asarray(feature_names)
        validate_data(X, y, feature_names)

        n_input_features = X.shape[1]
        if self.n_features > n_input_features:
            raise ValueError(
                f"n_features ({self.n_features}) cannot be greater than "
                f"the number of input features ({n_input_features})."
            )

        self.feature_names_ = feature_names

        # Train RF on all features
        if self.verbose:
            print(f"Training Random Forest on {n_input_features} features...")

        result = train_rf_model(
            X=X,
            y=y,
            rf_params=self.rf_params,
            train_test_split_params=self.train_test_split_params,
            random_state=self.random_state,
            verbose=self.verbose,
            description="all features",
        )

        self.model_ = result.model

        # Get sorted feature indices and select top N
        sorted_indices = get_sorted_feature_indices(self.model_)
        selected_indices = sorted_indices[: self.n_features]

        selected_features = feature_names[selected_indices]
        selected_importances = self.model_.feature_importances_[selected_indices]

        # Train again on selected features for final metrics
        X_selected = X[:, selected_indices]

        if self.verbose:
            print(f"\nEvaluating with top {self.n_features} features...")

        final_result = train_rf_model(
            X=X_selected,
            y=y,
            rf_params=self.rf_params,
            train_test_split_params=self.train_test_split_params,
            random_state=self.random_state,
            verbose=self.verbose,
            description=f"top {self.n_features} features",
        )

        # Store results
        self.results_ = SelectionResult(
            selected_features=selected_features,
            feature_importances=selected_importances,
            n_features_selected=self.n_features,
            metadata={
                "initial_accuracy": result.accuracy,
                "final_accuracy": final_result.accuracy,
                "all_feature_importances": self.model_.feature_importances_,
                "sorted_feature_names": feature_names[sorted_indices],
            },
        )

        self.is_fitted_ = True

        if self.verbose:
            print(f"\nSelected {self.n_features} features:")
            print(
                f"  Accuracy: {result.accuracy:.4f} (all) -> {final_result.accuracy:.4f} (selected)"
            )

        return self

    def get_params(self) -> dict:
        """Get hyperparameters of this selector."""
        return {
            "n_features": self.n_features,
            "rf_params": self.rf_params,
            "train_test_split_params": self.train_test_split_params,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


class RFMinMaxSelector(FeatureSelector):
    """
    Find minimum features that maximize classification accuracy.

    This selector iteratively adds features (in order of RF importance)
    until adding more features no longer improves accuracy significantly.

    Parameters
    ----------
    threshold : float, default=0.01
        Minimum accuracy improvement to consider significant.
    stable_iterations : int, default=3
        Number of iterations without improvement before stopping.
    cutoff_method : {"increment", "jump"}, default="increment"
        Method for determining when to stop:
        - "increment": Checks if accuracy keeps improving within threshold
        - "jump": Resets counter when accuracy jumps above threshold
    rf_params : dict, optional
        Parameters for RandomForestClassifier.
    train_test_split_params : dict, optional
        Parameters for train_test_split.
    random_state : int, default=42
        Random state for reproducibility.
    verbose : bool, default=True
        Whether to print progress and results.

    Attributes
    ----------
    model_ : RandomForestClassifier
        The fitted Random Forest model (trained on selected features).
    results_ : SelectionResult
        Detailed selection results including accuracy curve.
    feature_names_ : np.ndarray
        All feature names used during fitting.
    accuracy_curve_ : np.ndarray
        Accuracy at each iteration (for plotting).

    Examples
    --------
    >>> selector = RFMinMaxSelector(threshold=0.01, stable_iterations=3)
    >>> selector.fit(X, y, feature_names)
    >>>
    >>> # See how many features were selected
    >>> print(f"Selected {selector.results.n_features_selected} features")
    >>>
    >>> # Plot the accuracy curve
    >>> fig = selector.plot_accuracy_curve()
    """

    def __init__(
        self,
        threshold: float = 0.01,
        stable_iterations: int = 3,
        cutoff_method: Literal["increment", "jump"] = "increment",
        rf_params: dict | None = None,
        train_test_split_params: dict | None = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.threshold = threshold
        self.stable_iterations = stable_iterations
        self.cutoff_method = cutoff_method
        self.rf_params = (
            rf_params if rf_params is not None else DEFAULT_RF_PARAMS.copy()
        )
        self.train_test_split_params = (
            train_test_split_params
            if train_test_split_params is not None
            else DEFAULT_SPLIT_PARAMS.copy()
        )
        self.random_state = random_state
        self.verbose = verbose

        # These are set after fitting
        self.model_ = None
        self.results_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        self.accuracy_curve_ = None
        self._sorted_feature_names = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: np.ndarray,
    ) -> Self:
        """
        Fit the selector by iteratively adding features.

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
            The fitted selector instance.
        """
        feature_names = np.asarray(feature_names)
        validate_data(X, y, feature_names)

        self.feature_names_ = feature_names
        n_features = X.shape[1]

        # Step 1: Train initial RF to get feature importances
        if self.verbose:
            print(f"Training initial RF on {n_features} features to get importances...")

        initial_result = train_rf_model(
            X=X,
            y=y,
            rf_params=self.rf_params,
            train_test_split_params=self.train_test_split_params,
            random_state=self.random_state,
            verbose=self.verbose,
            description="initial (all features)",
        )

        # Sort features by importance
        sorted_indices = get_sorted_feature_indices(initial_result.model)
        sorted_features = X[:, sorted_indices]
        self._sorted_feature_names = feature_names[sorted_indices]

        # Step 2: Iteratively add features
        if self.verbose:
            print(
                f"\nIteratively adding features (threshold={self.threshold}, "
                f"stable_iterations={self.stable_iterations})..."
            )

        optimal_n, accuracy_curve = self._find_optimal_n_features(sorted_features, y)

        self.accuracy_curve_ = accuracy_curve

        # Step 3: Get final results with optimal features
        selected_features = self._sorted_feature_names[:optimal_n]
        selected_importances = initial_result.model.feature_importances_[
            sorted_indices[:optimal_n]
        ]

        # Train final model on selected features
        X_selected = sorted_features[:, :optimal_n]

        if self.verbose:
            print(f"\nFinal evaluation with {optimal_n} features...")

        final_result = train_rf_model(
            X=X_selected,
            y=y,
            rf_params=self.rf_params,
            train_test_split_params=self.train_test_split_params,
            random_state=self.random_state,
            verbose=self.verbose,
            description=f"optimal ({optimal_n} features)",
        )

        self.model_ = final_result.model

        # Store results
        self.results_ = SelectionResult(
            selected_features=selected_features,
            feature_importances=selected_importances,
            n_features_selected=optimal_n,
            metadata={
                "initial_accuracy": initial_result.accuracy,
                "final_accuracy": final_result.accuracy,
                "accuracy_curve": accuracy_curve,
                "optimal_index": optimal_n,
                "sorted_feature_names": self._sorted_feature_names,
                "all_feature_importances": initial_result.model.feature_importances_[
                    sorted_indices
                ],
                "threshold": self.threshold,
                "stable_iterations": self.stable_iterations,
                "cutoff_method": self.cutoff_method,
            },
        )

        self.is_fitted_ = True

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Selected {optimal_n} features (out of {n_features})")
            print(
                f"Accuracy: {initial_result.accuracy:.4f} (all) -> {final_result.accuracy:.4f} (selected)"
            )
            print(f"\nTop features: {list(selected_features[:5])}...")

        return self

    def _find_optimal_n_features(
        self,
        sorted_features: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """
        Find optimal number of features using iterative training.

        Returns
        -------
        tuple[int, np.ndarray]
            Optimal number of features and accuracy curve.
        """
        n_features = sorted_features.shape[1]

        counter = 0
        max_acc_idx = 0
        acc_list = [0.0]  # Start with 0 for indexing convenience

        iterator = range(1, n_features + 1)
        if self.verbose:
            iterator = tqdm(
                iterator,
                desc="Adding features",
                total=n_features,
            )

        for n in iterator:
            if counter > self.stable_iterations:
                break

            # Train on first n features
            X_subset = sorted_features[:, :n]

            result = train_rf_model(
                X=X_subset,
                y=y,
                rf_params=self.rf_params,
                train_test_split_params=self.train_test_split_params,
                random_state=self.random_state,
                verbose=False,
            )

            acc_list.append(result.accuracy)

            # Check if we should continue
            acc_diff = abs(acc_list[max_acc_idx] - result.accuracy)

            if max_acc_idx == np.argmax(acc_list):
                # No new maximum
                pass
            elif acc_diff > self.threshold:
                # Significant improvement
                if self.cutoff_method == "jump":
                    max_acc_idx = np.argmax(acc_list)
                    counter = 0
                elif self.cutoff_method == "increment":
                    max_acc_idx, counter = self._increment_check(
                        acc_list, max_acc_idx, counter
                    )

            counter += 1

        accuracy_curve = np.array(acc_list)
        optimal_n = max_acc_idx

        # Handle edge case where optimal_n is 0
        if optimal_n == 0:
            optimal_n = np.argmax(acc_list)

        return optimal_n, accuracy_curve

    def _increment_check(
        self,
        acc_list: list[float],
        current_max_idx: int,
        counter: int,
    ) -> tuple[int, int]:
        """
        Check if accuracy keeps improving within stable_iterations.
        """
        temp_max_idx = current_max_idx
        temp_counter = counter

        for _ in range(self.stable_iterations):
            end_idx = min(temp_max_idx + self.stable_iterations + 1, len(acc_list))
            window = np.array(acc_list[temp_max_idx:end_idx])

            if len(window) == 0:
                break

            acc_diffs = window - acc_list[temp_max_idx]
            improvements = np.where(acc_diffs > self.threshold)[0]

            if len(improvements) == 0:
                break
            else:
                if temp_counter > 0:
                    temp_counter -= 1
                temp_max_idx += 1

        return temp_max_idx, temp_counter

    def plot_accuracy_curve(
        self,
        save_path: str | None = None,
        figsize: tuple[int, int] = (12, 6),
        show: bool = True,
    ):
        """
        Plot accuracy vs number of features.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        figsize : tuple, default=(12, 6)
            Figure size.
        show : bool, default=True
            Whether to display the figure with plt.show(). If False, the
            figure is closed after it is saved.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        self._check_is_fitted()

        # Import here to avoid import at module level
        from ._plotting import plot_accuracy_curve

        return plot_accuracy_curve(
            accuracy_curve=self.accuracy_curve_,
            optimal_n=self.results_.n_features_selected,
            feature_names=self._sorted_feature_names,
            threshold=self.threshold,
            stable_iterations=self.stable_iterations,
            cutoff_method=self.cutoff_method,
            save_path=save_path,
            figsize=figsize,
            show=show,
        )

    def get_params(self) -> dict:
        """Get hyperparameters of this selector."""
        return {
            "threshold": self.threshold,
            "stable_iterations": self.stable_iterations,
            "cutoff_method": self.cutoff_method,
            "rf_params": self.rf_params,
            "train_test_split_params": self.train_test_split_params,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }
