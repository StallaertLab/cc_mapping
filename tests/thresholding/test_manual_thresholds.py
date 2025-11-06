"""Tests for manual threshold functionality in GMMThresholding.

This module tests the ability to override GMM-derived automatic thresholds with
user-specified manual thresholds. This is critical for ensuring consistent
thresholding across different datasets or experimental conditions.
"""

import pytest
import numpy as np
from src.cc_mapping.thresholding import GMMThresholding


### Basic Manual Threshold Tests ###

def test_manual_thresholds_override_gmm(sample_gmm_thresholding_instance):
    """Test that manual thresholds override GMM-derived automatic thresholds."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=2)
    
    manual_threshold = [1.5]
    gmm.categorize_samples(
        ordered_labels=['Low', 'High'],
        manual_thresholds=manual_threshold
    )
    
    # Should use manual thresholding
    assert gmm._manual_decision_boundaries == True, (
        "Should indicate manual thresholding is active"
    )
    
    # Should return exact threshold provided
    assert gmm.return_thresholds() == manual_threshold, (
        "Should return the exact manual threshold provided, not GMM-derived"
    )


def test_manual_thresholds_set_flag(sample_gmm_thresholding_instance):
    """Test that manual_decision_boundaries flag is set correctly."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=3)
    
    # Without manual thresholds
    gmm.categorize_samples(ordered_labels=['Low', 'Medium', 'High'])
    assert gmm._manual_decision_boundaries == False, (
        "Should be False when using automatic thresholding"
    )
    
    # With manual thresholds
    gmm.categorize_samples(
        ordered_labels=['Low', 'Medium', 'High'],
        manual_thresholds=[1.0, 2.0]
    )
    assert gmm._manual_decision_boundaries == True, (
        "Should be True when using manual thresholding"
    )


def test_manual_thresholds_with_collapsed_labels(sample_gmm_thresholding_instance):
    """Test manual thresholds work with duplicate label collapsing."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    manual_threshold = [1.8]
    gmm.categorize_samples(
        ordered_labels=['Low', 'Low', 'High', 'High'],
        manual_thresholds=manual_threshold,
        duplicate_labels=True
    )
    
    # Should use manual thresholding even with label collapsing
    assert gmm._manual_decision_boundaries == True, (
        "Manual thresholding should work with label collapsing"
    )
    
    # Should have the manual threshold
    assert gmm.return_thresholds() == manual_threshold, (
        "Should use manual threshold with collapsed labels"
    )
    
    # Verify labels are assigned
    unique_labels = set(gmm.adata.obs['labels'].unique())
    assert unique_labels == {'Low', 'High'}, (
        "Should have correct labels assigned with manual threshold"
    )


def test_manual_thresholds_multiple_thresholds(sample_gmm_thresholding_instance):
    """Test manual thresholding with multiple thresholds."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=3)
    
    manual_thresholds = [0.5, 1.5]
    gmm.categorize_samples(
        ordered_labels=['Low', 'Medium', 'High'],
        manual_thresholds=manual_thresholds
    )
    
    # Should use manual thresholding
    assert gmm._manual_decision_boundaries == True, (
        "Should use manual thresholding"
    )
    
    # Should return both thresholds
    assert gmm.return_thresholds() == manual_thresholds, (
        "Should return both manual thresholds in order"
    )
    
    # Verify all three labels are assigned
    unique_labels = set(gmm.adata.obs['labels'].unique())
    assert unique_labels == {'Low', 'Medium', 'High'}, (
        "Should have all three labels assigned"
    )


### Validation Tests ###

def test_manual_thresholds_count_validation(sample_gmm_thresholding_instance):
    """Test that manual threshold count must be n_categories - 1."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=2)
    
    # Need 1 threshold for 2 labels, providing 2
    with pytest.raises(
        ValueError,
        match="Number of thresholds.*must be one less than.*labels"
    ):
        gmm.categorize_samples(
            ordered_labels=['Low', 'High'],
            manual_thresholds=[1.0, 2.0]  # Too many!
        )


def test_manual_thresholds_count_validation_collapsed(sample_gmm_thresholding_instance):
    """Test threshold count validation with collapsed labels."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=6)
    
    # Collapsing to 3 unique labels, need 2 thresholds
    with pytest.raises(
        ValueError,
        match="Number of thresholds.*must be one less than.*unique labels"
    ):
        gmm.categorize_samples(
            ordered_labels=['Low', 'Low', 'Med', 'Med', 'High', 'High'],
            manual_thresholds=[1.0],  # Need 2 thresholds!
            duplicate_labels=True
        )


def test_manual_thresholds_type_validation(sample_gmm_thresholding_instance):
    """Test that manual thresholds must be numeric."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=2)
    
    # This should raise a TypeError or ValueError depending on validation
    with pytest.raises((TypeError, ValueError)):
        gmm.categorize_samples(
            ordered_labels=['Low', 'High'],
            manual_thresholds=['not_a_number']
        )


def test_manual_thresholds_empty_list(sample_gmm_thresholding_instance):
    """Test that empty manual threshold list is rejected."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=2)
    
    # Empty list should fail validation
    with pytest.raises(ValueError):
        gmm.categorize_samples(
            ordered_labels=['Low', 'High'],
            manual_thresholds=[]
        )


### Correct Assignment Tests ###

def test_manual_threshold_assigns_correctly(sample_gmm_thresholding_instance):
    """Test that samples are assigned correctly based on manual threshold."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=2)
    
    # Use a specific threshold
    threshold = 0.0  # Middle of data (mean is 0)
    gmm.categorize_samples(
        ordered_labels=['Low', 'High'],
        manual_thresholds=[threshold]
    )
    
    # Check that samples are split around the threshold
    low_samples = gmm.adata[gmm.adata.obs['labels'] == 'Low']
    high_samples = gmm.adata[gmm.adata.obs['labels'] == 'High']
    
    assert len(low_samples) > 0, "Should have some low samples"
    assert len(high_samples) > 0, "Should have some high samples"
    
    # Verify assignment logic - get feature values from X matrix
    feature = gmm.feature
    low_values = low_samples[:, feature].X.flatten()
    high_values = high_samples[:, feature].X.flatten()
    
    # With threshold at 0.0, there should be clear separation
    # All low values should be <= threshold, all high values should be > threshold
    # (or vice versa depending on sorting)
    assert np.max(low_values) <= threshold or np.min(high_values) > threshold, (
        "Manual threshold should create separation at the specified value"
    )


def test_manual_threshold_with_extreme_value(sample_gmm_thresholding_instance):
    """Test manual threshold with extreme value."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=2)
    
    # Use very high threshold - all samples should be 'Low'
    extreme_threshold = [1000.0]
    gmm.categorize_samples(
        ordered_labels=['Low', 'High'],
        manual_thresholds=extreme_threshold
    )
    
    # Most or all samples should be in first category
    labels = gmm.adata.obs['labels']
    low_count = (labels == 'Low').sum()
    high_count = (labels == 'High').sum()
    
    # With threshold way above data range, expect most samples in 'Low'
    assert low_count > high_count, (
        "Extreme high threshold should assign most samples to first category"
    )


def test_manual_threshold_ordering(sample_gmm_thresholding_instance):
    """Test that manual thresholds must be in ascending order."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=3)
    
    # Thresholds in wrong order
    with pytest.raises(ValueError, match="must be in ascending order"):
        gmm.categorize_samples(
            ordered_labels=['Low', 'Medium', 'High'],
            manual_thresholds=[2.0, 1.0]  # Wrong order!
        )


def test_manual_threshold_with_duplicates(sample_gmm_thresholding_instance):
    """Test that duplicate threshold values are rejected."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=3)
    
    with pytest.raises(ValueError, match="duplicate|unique"):
        gmm.categorize_samples(
            ordered_labels=['Low', 'Medium', 'High'],
            manual_thresholds=[1.0, 1.0]  # Duplicates!
        )


### Integration Tests ###

def test_manual_then_automatic_thresholding(sample_gmm_thresholding_instance):
    """Test switching from manual to automatic thresholding."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=2)
    
    # First use manual
    manual_threshold = [1.5]
    gmm.categorize_samples(
        ordered_labels=['Low', 'High'],
        manual_thresholds=manual_threshold
    )
    assert gmm._manual_decision_boundaries == True
    assert gmm.return_thresholds() == manual_threshold
    
    # Then use automatic thresholding
    gmm.categorize_samples(ordered_labels=['Low', 'High'])
    assert gmm._manual_decision_boundaries == False
    automatic_threshold = gmm.return_thresholds()
    
    # Automatic threshold should be different from manual
    assert automatic_threshold != manual_threshold, (
        "Automatic thresholding should produce different threshold than manual"
    )


def test_manual_threshold_consistency_across_calls(sample_gmm_thresholding_instance):
    """Test that manual threshold can be reused consistently."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=2)
    
    threshold = [1.0]
    
    # Apply twice
    gmm.categorize_samples(
        ordered_labels=['Low', 'High'],
        manual_thresholds=threshold
    )
    first_labels = gmm.adata.obs['labels'].copy()
    
    gmm.categorize_samples(
        ordered_labels=['Low', 'High'],
        manual_thresholds=threshold
    )
    second_labels = gmm.adata.obs['labels'].copy()
    
    # Should get identical results
    assert (first_labels == second_labels).all(), (
        "Manual threshold should produce consistent results across calls"
    )


def test_manual_threshold_with_single_category(sample_gmm_thresholding_instance):
    """Test that single category doesn't accept manual thresholds."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    # All same label = 1 category = 0 thresholds needed
    # Providing a threshold should fail
    with pytest.raises(ValueError):
        gmm.categorize_samples(
            ordered_labels=['Low', 'Low', 'Low', 'Low'],
            manual_thresholds=[1.0],  # Can't have threshold with 1 category!
            duplicate_labels=True
        )
