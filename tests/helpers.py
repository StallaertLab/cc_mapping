"""
Test Helper Utilities for the Project's Test Suite.

This module provides utility functions designed to simplify writing tests,
particularly focusing on testing components involving `anndata.AnnData` objects
and specific class behaviors (like `GMMThresholding`).

It includes functions for:

1.  **Test Data Generation/Manipulation:**
    - Creating variations of `anndata.AnnData` objects with specific
      properties or modifications needed to test edge cases and different
      input scenarios (e.g., `create_modified_adata`).

2.  **Custom Assertions:**
    - Providing reusable, higher-level assertion functions to check for
      complex states, multiple related conditions, or specific initialization
      patterns within the classes under test (e.g., `assert_initial_gmm_state`).

The goal of this module is to reduce boilerplate code in individual test files,
improve the readability and maintainability of the test suite, and centralize
common setup and verification logic.
"""

from typing import Any

import anndata as ad
import numpy as np

from cc_mapping.thresholding import (
    GMMThresholding,
    _GaussianMixtureModelInfo,
    _SingleThresholdingEventModel,
)


def create_modified_adata(
    base_adata: ad.AnnData,
    *,  # Force subsequent arguments to be keyword-only for clarity
    x_dtype: type[np.generic] | np.dtype | None = None,
    x_value_at: tuple[int, int, Any] | None = None,
    add_obs: dict[str, Any] | None = None,
    add_uns: dict[str, Any] | None = None,
    remove_obs: list[str] | tuple[str, ...] | None = None,
    remove_var: list[str] | tuple[str, ...] | None = None,
    remove_uns: list[str] | tuple[str, ...] | None = None,
) -> ad.AnnData:
    """
    Creates a modified copy of a base AnnData object for testing purposes.

    Leaves the original `base_adata` object unchanged.

    Args:
        base_adata: The original AnnData object to copy and modify.
        x_dtype: If provided, change the dtype of adata.X using .astype().
                 Example: `np.object_`, `np.float32`.
        x_value_at: If provided, a tuple (row_idx, col_idx, new_value)
                    to set a specific value in adata.X. Applied *after*
                    any potential dtype change.
        add_obs: If provided, a dictionary of {key: value} pairs to add or
                 overwrite columns in adata.obs. The 'value' should typically
                 be array-like with length matching the number of observations,
                 or a single value to be broadcast.
        add_uns: If provided, a dictionary of {key: value} pairs to add or
                 overwrite entries in adata.uns.
        remove_obs: If provided, a list or tuple of keys (column names) to
                    remove from adata.obs.
        remove_var: If provided, a list or tuple of keys (column names) to
                    remove from adata.var.
        remove_uns: If provided, a list or tuple of keys to remove from adata.uns.

    Returns:
        A new AnnData object with the specified modifications applied.

    Raises:
        KeyError: If trying to remove a key from .obs, .var, or .uns that
                  doesn't exist.
        IndexError: If `x_value_at` indices are out of bounds for adata.X.
        TypeError: If `x_dtype` conversion fails, or trying to assign an
                   incompatible type with `x_value_at`.
        AttributeError: If `base_adata.X` is None or doesn't support `.astype`.
        ValueError: If attempting operations incompatible with the data structure
                    (e.g., assigning incorrect length array to .obs).
    """
    if not isinstance(base_adata, ad.AnnData):
        raise TypeError("base_adata must be an AnnData object.")

    # Create a deep copy to avoid modifying the original object
    adata_copy = base_adata.copy()

    # --- Apply modifications to the copy ---

    # 1. Change X dtype (if requested)
    if x_dtype is not None:
        if adata_copy.X is None:
            raise ValueError("Cannot change dtype of X because base_adata.X is None.")
        try:
            # Use astype for numpy arrays, handle sparse potentially differently if needed
            # This assumes X is typically numpy or supports astype
            adata_copy.X = adata_copy.X.astype(x_dtype)
        except Exception as e:
            print(f"Error changing X dtype to {x_dtype}: {e}")
            raise  # Re-raise the exception

    # 2. Set specific X value (if requested)
    if x_value_at is not None:
        if adata_copy.X is None:
            raise ValueError("Cannot set value in X because base_adata.X is None.")
        try:
            row, col, val = x_value_at
            adata_copy.X[row, col] = val
        except IndexError:
            print(
                f"Error setting X value: Index ({row}, {col}) out of bounds for shape {adata_copy.X.shape}"
            )
            raise
        except (TypeError, ValueError) as e:
            print(
                f"Error setting X value: Type or value mismatch assigning '{val}' at ({row}, {col}). Error: {e}"
            )
            raise

    # 3. Add/Update .obs columns
    if add_obs:
        if not isinstance(add_obs, dict):
            raise TypeError("`add_obs` must be a dictionary of key-value pairs.")
        for key, value in add_obs.items():
            # Consider adding a length check if value is array-like
            if (
                hasattr(value, "__len__")
                and not isinstance(value, str)
                and len(value) != adata_copy.n_obs
            ):
                raise ValueError(
                    f"Length mismatch for obs key '{key}'. Expected {adata_copy.n_obs}, got {len(value)}."
                )
            adata_copy.obs[key] = value

    # 4. Add/Update .uns entries
    if add_uns:
        if not isinstance(add_uns, dict):
            raise TypeError("`add_uns` must be a dictionary of key-value pairs.")
        for key, value in add_uns.items():
            adata_copy.uns[key] = value

    # 5. Remove .obs columns
    if remove_obs:
        for key in remove_obs:
            try:
                del adata_copy.obs[key]
            except KeyError:
                print(f"Error removing obs key: Key '{key}' not found.")
                raise

    # 6. Remove .var columns
    if remove_var:
        for key in remove_var:
            try:
                del adata_copy.var[key]
            except KeyError:
                print(f"Error removing var key: Key '{key}' not found.")
                raise

    # 7. Remove .uns entries
    if remove_uns:
        for key in remove_uns:
            try:
                del adata_copy.uns[key]
            except KeyError:
                print(f"Error removing uns key: Key '{key}' not found.")
                raise

    return adata_copy


def assert_adata_copy_and_uns(gmm_obj: GMMThresholding, original_adata: ad.AnnData):
    """
    Asserts adata is copied and .uns['gmm_thresholding_events'] is initialized.

    Checks include:
    - `gmm_obj.adata` is not the same object as `original_adata`.
    - `gmm_obj.adata`'s `.X`, `.obs`, `.var` contents match `original_adata`.
    - `gmm_obj.adata.uns` contains the key 'gmm_thresholding_events'.
    - `gmm_obj.adata.uns['gmm_thresholding_events']` is a plain `dict`
      (anndata cannot write an `OrderedDict` to .h5ad).
    """
    assert gmm_obj.adata is not original_adata, (
        "Object's adata should be a copy, not the same object."
    )
    # Check crucial components are equal (adjust if using sparse matrices etc.)
    assert np.array_equal(gmm_obj.adata.X, original_adata.X), (
        "Copied adata.X data mismatch."
    )
    assert gmm_obj.adata.obs.equals(original_adata.obs), "Copied adata.obs mismatch."
    assert gmm_obj.adata.var.equals(original_adata.var), "Copied adata.var mismatch."
    # Check .uns key specifically
    assert "gmm_thresholding_events" in gmm_obj.adata.uns, (
        ".uns['gmm_thresholding_events'] key missing in object's adata."
    )
    assert type(gmm_obj.adata.uns["gmm_thresholding_events"]) is dict, (
        ".uns['gmm_thresholding_events'] has wrong type (should be a plain dict)."
    )


def assert_direct_attributes_initialized(
    gmm_obj: GMMThresholding,
    expected_feature: str,
    expected_label: str,
    expected_random_state: int | None = 42,
):
    """
    Asserts direct attribute assignments from __init__ parameters are correct.

    Checks include:
    - `gmm_obj.feature` matches `expected_feature`.
    - `gmm_obj.label_obs_save_str` matches `expected_label`.
    - `gmm_obj.random_state` matches `expected_random_state` (the init param).
    """
    if expected_random_state is None:
        expected_random_state = 42

    assert gmm_obj.feature == expected_feature, (
        f"Attribute 'feature' mismatch. Expected '{expected_feature}', got '{gmm_obj.feature}'."
    )
    assert gmm_obj.label_obs_save_str == expected_label, (
        f"Attribute 'label_obs_save_str' mismatch. Expected '{expected_label}', got '{gmm_obj.label_obs_save_str}'."
    )
    # Checks the random_state *parameter value* stored on the object
    assert gmm_obj.random_state == expected_random_state, (
        f"Attribute 'random_state' mismatch. Expected {expected_random_state}, got {gmm_obj.random_state}."
    )


def assert_default_internal_states(gmm_obj: GMMThresholding):
    """
    Asserts attributes initialized to default values within __init__.

    Checks include:
    - `gmm_obj.manual_decision_boundaries` is False.
    - `gmm_obj.decision_boundaries` is None.
    """
    assert gmm_obj._manual_decision_boundaries is False, (
        "Attribute 'manual_decision_boundaries' should default to False."
    )
    assert gmm_obj._decision_boundaries is None, (
        "Attribute 'decision_boundaries' should default to None."
    )


def assert_gmm_kwargs_processed(
    gmm_obj: GMMThresholding,
    input_gmm_kwargs: dict | None = None,  # Original kwargs passed to __init__
    input_random_state: int | None = None,  # Original random_state passed to __init__
):
    """
    Asserts the gmm_obj.gmm_kwargs attribute reflects the correct processing.

    Checks include:
    - Calculates the expected final `gmm_kwargs` dictionary based on whether
      `input_gmm_kwargs` was None or a dict, and whether 'random_state'
      was present in the input dict vs. needing to be added from `input_random_state`.
    - Asserts `gmm_obj.gmm_kwargs` deeply equals the calculated expected dictionary.
    """
    # Check if input_random_state is None and set to default
    if input_random_state is None:
        input_random_state = 42

    # Replicate the logic from __init__ to determine the expected kwargs
    if input_gmm_kwargs is None:
        expected_final_kwargs = {
            "init_params": "k-means++",
            "n_init": 10,
            "max_iter": 1000,
            "random_state": input_random_state,  # Uses init param directly
        }
    elif isinstance(input_gmm_kwargs, dict):
        expected_final_kwargs = input_gmm_kwargs.copy()  # Work on copy
        # If "random_state" *was* in input_gmm_kwargs, its value is kept.
        if "random_state" not in expected_final_kwargs:
            # Adds init param value if key is missing
            expected_final_kwargs["random_state"] = input_random_state

    assert gmm_obj.gmm_kwargs == expected_final_kwargs, (
        f"Attribute 'gmm_kwargs' mismatch. Expected {expected_final_kwargs}, got {gmm_obj.gmm_kwargs}."
    )


def assert_dependent_models_initialized(gmm_obj: GMMThresholding):
    """
    Asserts dependent models (gmm_info, internal_data) are initialized correctly.

    Checks include:
    - `gmm_obj.gmm_info` is an instance of `_GaussianMixtureModelInfo`.
    - `gmm_obj.gmm_info.gmm_kwargs` matches `gmm_obj.gmm_kwargs`.
    - `gmm_obj.internal_data` is an instance of `_SingleThresholdingEventModel`.
    - `gmm_obj.internal_data.feature_name` matches `gmm_obj.feature`.
    - `gmm_obj.internal_data.gmm_obs_label` matches `gmm_obj.label_obs_save_str`.
    - `gmm_obj.internal_data.gmm_info` is the *same object* as `gmm_obj.gmm_info`.
    """
    # Check gmm_info
    assert isinstance(gmm_obj._gmm_info, _GaussianMixtureModelInfo), (
        "Attribute 'gmm_info' has wrong type."
    )
    assert gmm_obj._gmm_info.gmm_kwargs == gmm_obj.gmm_kwargs, (
        "Processed 'gmm_kwargs' not correctly propagated to 'gmm_info'."
    )

    # Check internal_data
    assert isinstance(gmm_obj._internal_data, _SingleThresholdingEventModel), (
        "Attribute 'internal_data' has wrong type."
    )
    assert gmm_obj._internal_data.feature_name == gmm_obj.feature, (
        "Attribute 'feature' not correctly propagated to 'internal_data'."
    )
    assert gmm_obj._internal_data.gmm_obs_label == gmm_obj.label_obs_save_str, (
        "Attribute 'label_obs_save_str' not correctly propagated to 'internal_data'."
    )
    assert gmm_obj._internal_data.gmm_info is gmm_obj._gmm_info, (
        "'gmm_info' object identity not correctly propagated to 'internal_data'."
    )


def probs_from_winners(
    winners, n_components: int, winner_prob: float = 0.7
) -> np.ndarray:
    """
    Builds a probability matrix (samples x components) from each sample's most likely component.

    The winning component of each sample gets `winner_prob`; the other components
    share the rest equally.
    """
    winners = np.asarray(winners)
    other_prob = (1 - winner_prob) / (n_components - 1)
    probs = np.full((len(winners), n_components), other_prob)
    probs[np.arange(len(winners)), winners] = winner_prob
    return probs


class FakeGaussianMixture:
    """
    Stand-in for sklearn's GaussianMixture that returns fixed probabilities.

    `predict_proba` looks every sample up by its feature value, so it works for
    any subset or ordering of `feature_values` (which must be sorted and unique).
    Means must be ascending, so the thresholding classes' sort-by-mean step keeps
    the component order of `probs`.
    """

    def __init__(self, feature_values, probs, means=None, variance: float = 1.0):
        self._feature_values = np.asarray(feature_values, dtype=float)
        self._probs = np.asarray(probs, dtype=float)
        n_components = self._probs.shape[1]
        if means is None:
            means = np.arange(n_components, dtype=float)
        self.means_ = np.asarray(means, dtype=float).reshape(-1, 1)
        self.covariances_ = np.full((n_components, 1, 1), variance)
        self.weights_ = np.full(n_components, 1.0 / n_components)

    def fit(self, X):
        return self

    def predict_proba(self, X):
        values = np.asarray(X, dtype=float).ravel()
        rows = np.searchsorted(self._feature_values, values)
        return self._probs[rows]


def use_fake_gaussian_mixture(
    monkeypatch, feature_values, probs, means=None, variance: float = 1.0
):
    """
    Makes GMMThresholding and SequentialGMM fit a `FakeGaussianMixture` instead of sklearn's model.
    """
    from cc_mapping.thresholding import sequential, single

    def make_fake(n_components, **gmm_kwargs):
        fake = FakeGaussianMixture(
            feature_values, probs, means=means, variance=variance
        )
        assert n_components == fake.weights_.size, "n_components must match probs"
        return fake

    monkeypatch.setattr(single, "GaussianMixture", make_fake)
    monkeypatch.setattr(sequential, "GaussianMixture", make_fake)
