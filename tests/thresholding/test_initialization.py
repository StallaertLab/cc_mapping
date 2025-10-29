"""Tests for initialization and constructor validation of GaussianMixtureModelThresholding.

This module tests the constructor validation, parameter checking, and proper
initialization of the GaussianMixtureModelThresholding class.
"""

from src.cc_mapping.thresholding import GaussianMixtureModelThresholding
from tests.helpers import (
    assert_adata_copy_and_uns,
    assert_direct_attributes_initialized,
    assert_default_internal_states,
    assert_gmm_kwargs_processed,
    assert_dependent_models_initialized,
    create_modified_adata
)

from collections import OrderedDict

import numpy as np
import pytest
import matplotlib as mpl
mpl.use('Agg')  # Set the backend before importing pyplot

                

# --- Initialization Tests ---
def test_init_success_defaults(sample_adata):
    """Tests successful initialization with default kwargs using helper."""
    feature_name = 'gene1'
    label_name = 'my_labels'

    gmm_thresholding = GaussianMixtureModelThresholding(
        adata=sample_adata,
        feature=feature_name,
        label_obs_save_str=label_name,
    )

    assert_adata_copy_and_uns(gmm_thresholding, sample_adata)
    assert_direct_attributes_initialized(
        gmm_obj=gmm_thresholding,
        expected_feature=feature_name,
        expected_label=label_name,
    )
    assert_default_internal_states(gmm_thresholding)
    assert_gmm_kwargs_processed(
        gmm_obj=gmm_thresholding,
    )
    assert_dependent_models_initialized(gmm_thresholding)

    assert len(gmm_thresholding.adata.uns['gmm_thresholding_events']) == 0, \
        "The `gmm_thresholding_events` should be initialized as an empty OrderedDict" 


def test_init_success_custom_kwargs(sample_adata):
    """Tests successful initialization with custom gmm_kwargs using helper."""
    feature_name = 'gene2'
    label_name = 'custom_output'
    # Custom kwargs *without* random_state initially
    init_gmm_kwargs = {'n_components': 5, 'covariance_type': 'diag'}
    init_random_state = 123

    gmm_thresholding = GaussianMixtureModelThresholding(
        adata=sample_adata,
        feature=feature_name,
        label_obs_save_str=label_name,
        gmm_kwargs=init_gmm_kwargs,
        random_state=init_random_state
    )

    assert_adata_copy_and_uns(gmm_thresholding, sample_adata)
    assert_direct_attributes_initialized(
        gmm_obj=gmm_thresholding,
        expected_feature=feature_name,
        expected_label=label_name,
        expected_random_state=init_random_state # Ensure random state is set correctly
    )
    assert_default_internal_states(gmm_thresholding)
    assert_gmm_kwargs_processed(
        gmm_obj=gmm_thresholding,
        input_gmm_kwargs=init_gmm_kwargs, # Pass the original dict to check processing
        input_random_state=init_random_state # Ensure random state is passed correctly to GMM
    )
    assert_dependent_models_initialized(gmm_thresholding)

def test_init_success_custom_kwargs_with_random_state(sample_adata):
    """Tests initialization when random_state is already in gmm_kwargs using helper."""
    feature_name = 'gene1'
    label_name = 'custom_output_rs'
    # User provides random_state within the dict
    init_gmm_kwargs = {'n_components': 2, 'random_state': 999}
    init_random_state = 123 # This should be stored in self.random_state but ignored for gmm_kwargs dict

    gmm_thresholding = GaussianMixtureModelThresholding(
        adata=sample_adata,
        feature=feature_name,
        label_obs_save_str=label_name,
        gmm_kwargs=init_gmm_kwargs,
        random_state=init_random_state
    )

    assert_adata_copy_and_uns(gmm_thresholding, sample_adata)
    assert_direct_attributes_initialized(
        gmm_obj=gmm_thresholding,
        expected_feature=feature_name,
        expected_label=label_name,
        expected_random_state=init_random_state # Ensure random state is set correctly
    )
    assert_default_internal_states(gmm_thresholding)
    assert_gmm_kwargs_processed(
        gmm_obj=gmm_thresholding,
        input_gmm_kwargs=init_gmm_kwargs, # Pass the original dict to check processing
        input_random_state=init_random_state # Ensure random state is passed correctly to GMM
    )
    assert_dependent_models_initialized(gmm_thresholding)

def test_init_success_existing_uns_key(sample_adata):
    """Tests initialization when the uns key already exists correctly."""
    existing_uns_data = {'gmm_thresholding_events': OrderedDict({ 'previous_run': 'example_run_1'})} # Example data to simulate a previous run
    modified_adata = create_modified_adata(sample_adata, add_uns= existing_uns_data) # Ensure the original sample_adata is unchanged

    feature_name = 'gene1'
    label_name = 'new_labels'

    gmm_thresholding = GaussianMixtureModelThresholding(
        adata=modified_adata,
        feature=feature_name,
        label_obs_save_str=label_name
    )

    assert_adata_copy_and_uns(gmm_thresholding, modified_adata)
    assert_direct_attributes_initialized(
        gmm_obj=gmm_thresholding,
        expected_feature=feature_name,
        expected_label=label_name,
    )
    assert_default_internal_states(gmm_thresholding)
    assert_gmm_kwargs_processed(
        gmm_obj=gmm_thresholding,
    )
    assert_dependent_models_initialized(gmm_thresholding)

    assert gmm_thresholding.adata.uns['gmm_thresholding_events'] == existing_uns_data['gmm_thresholding_events'], \
        'The existing uns key `gmm_thresholding_events` should be preserved and match the original content.'
    assert 'previous_run' in gmm_thresholding.adata.uns['gmm_thresholding_events'], \
        'The existing uns key `gmm_thresholding_events` should still contain the previous run data.'

# --- Error Condition Tests ---
# These tests verify exceptions are raised correctly.

@pytest.mark.parametrize("invalid_adata", [
    None,
    123,
    "string",
    {},
    []
])
def test_init_invalid_adata_type(invalid_adata):
    """Tests TypeError when adata is not an AnnData object."""
    with pytest.raises(TypeError, match="adata must be an AnnData.AnnData object"):
        GaussianMixtureModelThresholding(
            adata=invalid_adata,
            feature='gene1',
            label_obs_save_str='labels'
        )

def test_init_non_numeric_x_adata(sample_adata):
    """Tests TypeError when adata.X is not a numeric type."""
    non_numeric_x_adata = create_modified_adata(sample_adata, x_dtype=np.object_) # Ensure the original sample_adata is unchanged
    # Match the updated error message in __init__
    with pytest.raises(TypeError, match="adata.X must be a numeric type"):
        GaussianMixtureModelThresholding(
            adata=non_numeric_x_adata,
            feature='gene1',
            label_obs_save_str='labels'
        )

def test_init_existing_uns_key_wrong_type(sample_adata):
    """Tests TypeError when the uns key exists but is not an OrderedDict."""
    wrong_uns_type_adata = create_modified_adata(sample_adata,add_uns={'gmm_thresholding_events': {'not_dict':'values'}}) # Ensure the original sample_adata is unchanged

    # Match the updated error message in __init__
    with pytest.raises(TypeError, match="The 'gmm_thresholding_events' key in the AnnData object's `.uns` attribute must be an OrderedDict."):
        GaussianMixtureModelThresholding(
            adata=wrong_uns_type_adata,
            feature='gene1',
            label_obs_save_str='labels'
        )

@pytest.mark.parametrize("invalid_feature, expected_exception, match_pattern", [
    (None, TypeError, "feature must be a string"),
    (123, TypeError, "feature must be a string"),
    ([], TypeError, "feature must be a string"),
    ("", ValueError, "feature cannot be an empty string"),
])
def test_init_invalid_feature_type_or_value(sample_adata, invalid_feature, expected_exception, match_pattern):
    """Tests errors for invalid feature types or empty string."""
    with pytest.raises(expected_exception, match=match_pattern):
        GaussianMixtureModelThresholding(
            adata=sample_adata,
            feature=invalid_feature,
            label_obs_save_str='labels'
        )

def test_init_feature_not_found(sample_adata):
    """Tests KeyError when the feature is not in adata.var_names."""
    not_present_feature = 'unknown_gene'
    with pytest.raises(KeyError, match=f"Feature '{not_present_feature}' not found in adata.var_names. Please check the feature name."):
        GaussianMixtureModelThresholding(
            adata=sample_adata,
            feature=not_present_feature,  # This feature does not exist in sample_adata.var_names
            label_obs_save_str='labels'
        )

@pytest.mark.parametrize("invalid_label, expected_exception,match_pattern", [
    (None, TypeError ,"label_obs_save_str must be a string"),
    (123,TypeError ,"label_obs_save_str must be a string"),
    ([], TypeError,"label_obs_save_str must be a string"),
    ("", ValueError ,"label_obs_save_str cannot be an empty string")
])
def test_init_invalid_label_type(sample_adata, invalid_label, expected_exception, match_pattern):
    """Tests TypeError for invalid label_obs_save_str types."""
    with pytest.raises(expected_exception, match=match_pattern):
        GaussianMixtureModelThresholding(
            adata=sample_adata,
            feature='gene1',
            label_obs_save_str=invalid_label
        )

def test_init_existing_obs_label(sample_adata):
    """Tests KeyError when the label_obs_save_str already exists in adata.obs."""
    adata_copy = sample_adata.copy()
    label_name = 'existing_labels'
    adata_copy.obs[label_name] = 'some_value' # Add the column
    existing_obs_label_adata = create_modified_adata(sample_adata, add_obs={label_name: np.repeat(0, len(sample_adata))}) # Ensure the original sample_adata is unchanged

    with pytest.raises(KeyError, match=f"obs key '{label_name}' already exists in the AnnData object. Please choose a different label."):
        GaussianMixtureModelThresholding(
            adata=existing_obs_label_adata,
            feature='gene1',
            label_obs_save_str=label_name
        )

@pytest.mark.parametrize("invalid_kwargs", [
    "not_a_dict",
    123,
    ["list", "is", "not", "dict"]
])
def test_init_invalid_gmm_kwargs_type(sample_adata, invalid_kwargs):
    """Tests TypeError when gmm_kwargs is provided but is not a dictionary."""
    # Match the updated error message in __init__
    with pytest.raises(TypeError, match="gmm_kwargs must be a dictionary or None"):
        GaussianMixtureModelThresholding(
            adata=sample_adata,
            feature='gene1',
            label_obs_save_str='labels',
            gmm_kwargs=invalid_kwargs
        )
