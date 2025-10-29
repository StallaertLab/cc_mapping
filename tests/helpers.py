"""
Test Helper Utilities for the Project's Test Suite.

This module provides utility functions designed to simplify writing tests,
particularly focusing on testing components involving `anndata.AnnData` objects
and specific class behaviors (like `GaussianMixtureModelThresholding`).

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
from typing import Any, Dict, Optional, Tuple, List, Union, Type
from collections import OrderedDict

import anndata as ad
import numpy as np

from src.cc_mapping.thresholding import (GaussianMixtureModelThresholding,
                                        _SingleThresholdingEventModel,
                                        _GaussianMixtureModelInfo)

def create_modified_adata(
    base_adata: ad.AnnData,
    *, # Force subsequent arguments to be keyword-only for clarity
    x_dtype: Optional[Union[Type[np.generic], np.dtype]] = None,
    x_value_at: Optional[Tuple[int, int, Any]] = None,
    add_obs: Optional[Dict[str, Any]] = None,
    add_uns: Optional[Dict[str, Any]] = None,
    remove_obs: Optional[Union[List[str], Tuple[str, ...]]] = None,
    remove_var: Optional[Union[List[str], Tuple[str, ...]]] = None,
    remove_uns: Optional[Union[List[str], Tuple[str, ...]]] = None
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
            raise # Re-raise the exception

    # 2. Set specific X value (if requested)
    if x_value_at is not None:
        if adata_copy.X is None:
             raise ValueError("Cannot set value in X because base_adata.X is None.")
        try:
            row, col, val = x_value_at
            adata_copy.X[row, col] = val
        except IndexError:
            print(f"Error setting X value: Index ({row}, {col}) out of bounds for shape {adata_copy.X.shape}")
            raise
        except (TypeError, ValueError) as e:
             print(f"Error setting X value: Type or value mismatch assigning '{val}' at ({row}, {col}). Error: {e}")
             raise

    # 3. Add/Update .obs columns
    if add_obs:
        if not isinstance(add_obs, dict):
            raise TypeError("`add_obs` must be a dictionary of key-value pairs.")
        for key, value in add_obs.items():
            # Consider adding a length check if value is array-like
            if hasattr(value, '__len__') and not isinstance(value, str) and len(value) != adata_copy.n_obs:
                raise ValueError(f"Length mismatch for obs key '{key}'. Expected {adata_copy.n_obs}, got {len(value)}.")
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
 
def assert_adata_copy_and_uns(
    gmm_obj: GaussianMixtureModelThresholding,
    original_adata: ad.AnnData
):
    """
    Asserts adata is copied and .uns['gmm_thresholding_events'] is initialized.

    Checks include:
    - `gmm_obj.adata` is not the same object as `original_adata`.
    - `gmm_obj.adata`'s `.X`, `.obs`, `.var` contents match `original_adata`.
    - `gmm_obj.adata.uns` contains the key 'gmm_thresholding_events'.
    - `gmm_obj.adata.uns['gmm_thresholding_events']` is an `OrderedDict`.
    """
    assert gmm_obj.adata is not original_adata, \
        "Object's adata should be a copy, not the same object."
    # Check crucial components are equal (adjust if using sparse matrices etc.)
    assert np.array_equal(gmm_obj.adata.X, original_adata.X), \
        "Copied adata.X data mismatch."
    assert gmm_obj.adata.obs.equals(original_adata.obs), \
        "Copied adata.obs mismatch."
    assert gmm_obj.adata.var.equals(original_adata.var), \
        "Copied adata.var mismatch."
    # Check .uns key specifically
    assert "gmm_thresholding_events" in gmm_obj.adata.uns, \
        ".uns['gmm_thresholding_events'] key missing in object's adata."
    assert isinstance(gmm_obj.adata.uns["gmm_thresholding_events"], OrderedDict), \
        ".uns['gmm_thresholding_events'] has wrong type (should be OrderedDict)."


def assert_direct_attributes_initialized(
    gmm_obj: GaussianMixtureModelThresholding,
    expected_feature: str,
    expected_label: str,
    expected_random_state: Optional[int] = 42
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

    assert gmm_obj.feature == expected_feature, \
        f"Attribute 'feature' mismatch. Expected '{expected_feature}', got '{gmm_obj.feature}'."
    assert gmm_obj.label_obs_save_str == expected_label, \
        f"Attribute 'label_obs_save_str' mismatch. Expected '{expected_label}', got '{gmm_obj.label_obs_save_str}'."
    # Checks the random_state *parameter value* stored on the object
    assert gmm_obj.random_state == expected_random_state, \
        f"Attribute 'random_state' mismatch. Expected {expected_random_state}, got {gmm_obj.random_state}."


def assert_default_internal_states(gmm_obj: GaussianMixtureModelThresholding):
    """
    Asserts attributes initialized to default values within __init__.

    Checks include:
    - `gmm_obj.manual_decision_boundaries` is False.
    - `gmm_obj.decision_boundaries` is None.
    """
    assert gmm_obj._manual_decision_boundaries is False, \
        "Attribute 'manual_decision_boundaries' should default to False."
    assert gmm_obj._decision_boundaries is None, \
        "Attribute 'decision_boundaries' should default to None."


def assert_gmm_kwargs_processed(
    gmm_obj: GaussianMixtureModelThresholding,
    input_gmm_kwargs: Optional[dict] = None, # Original kwargs passed to __init__
    input_random_state: Optional[int] = None          # Original random_state passed to __init__
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
            'init_params': 'k-means++', 'n_init': 10, 'max_iter': 1000,
            'random_state': input_random_state # Uses init param directly
        }
    elif isinstance(input_gmm_kwargs, dict):
        expected_final_kwargs = input_gmm_kwargs.copy() # Work on copy
        # If "random_state" *was* in input_gmm_kwargs, its value is kept.
        if "random_state" not in expected_final_kwargs:
            # Adds init param value if key is missing
            expected_final_kwargs["random_state"] = input_random_state

    assert gmm_obj.gmm_kwargs == expected_final_kwargs, \
        f"Attribute 'gmm_kwargs' mismatch. Expected {expected_final_kwargs}, got {gmm_obj.gmm_kwargs}."


def assert_dependent_models_initialized(gmm_obj: GaussianMixtureModelThresholding):
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
    assert isinstance(gmm_obj._gmm_info, _GaussianMixtureModelInfo), \
        "Attribute 'gmm_info' has wrong type."
    assert gmm_obj._gmm_info.gmm_kwargs == gmm_obj.gmm_kwargs, \
        "Processed 'gmm_kwargs' not correctly propagated to 'gmm_info'."

    # Check internal_data
    assert isinstance(gmm_obj._internal_data, _SingleThresholdingEventModel), \
        "Attribute 'internal_data' has wrong type."
    assert gmm_obj._internal_data.feature_name == gmm_obj.feature, \
        "Attribute 'feature' not correctly propagated to 'internal_data'."
    assert gmm_obj._internal_data.gmm_obs_label == gmm_obj.label_obs_save_str, \
        "Attribute 'label_obs_save_str' not correctly propagated to 'internal_data'."
    assert gmm_obj._internal_data.gmm_info is gmm_obj._gmm_info, \
        "'gmm_info' object identity not correctly propagated to 'internal_data'."