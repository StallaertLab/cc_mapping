"""Tests for the categorize_samples method of GMMThresholding.

This module tests the core categorization logic including default behavior,
label assignment, and basic validation. More specific tests for label collapsing
and manual thresholds are in their dedicated test modules.
"""

import pytest


def test_categorize_samples_default_success(sample_gmm_thresholding_instance):
    """Test if the categorize_samples method works correctly with default parameters."""
    n_components = 2
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=n_components)

    # Should warn about using default labels and require ordered_labels
    with pytest.warns(
        UserWarning, match="ordered_labels is not set. Using default labels"
    ):
        gmm.categorize_samples()

    assert (
        not gmm._manual_decision_boundaries
    ), "manual_decision_boundaries should be False by default."


def test_categorize_samples_with_ordered_labels(sample_gmm_thresholding_instance):
    """Test categorize_samples with explicit ordered labels."""
    n_components = 2
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=n_components)

    gmm.categorize_samples(ordered_labels=["Low", "High"])

    # Should not use manual thresholding
    assert (
        not gmm._manual_decision_boundaries
    ), "Should use automatic thresholding when no manual_thresholds provided"

    # Should have labels assigned
    assert "labels" in gmm.adata.obs.columns, "Labels should be added to adata.obs"

    # Should have 2 unique labels
    unique_labels = set(gmm.adata.obs["labels"].unique())
    assert unique_labels == {"Low", "High"}, "Should have exactly the ordered labels"


def test_categorize_samples_requires_fit(sample_gmm_thresholding_instance):
    """Test that categorize_samples requires fit to be called first."""
    gmm = sample_gmm_thresholding_instance

    # Attempting to categorize without fitting should fail
    with pytest.raises((AttributeError, ValueError)):
        gmm.categorize_samples(ordered_labels=["Low", "High"])
