"""Tests for duplicate label collapsing functionality in GMMThresholding.

This module tests the ability to collapse multiple GMM components into fewer
categories by using duplicate labels. This is useful for cross-dataset robustness
where many components provide adaptive boundaries but fewer categories are desired.
"""

import pytest
import numpy as np
from src.cc_mapping.thresholding import GMMThresholding


### Basic Label Collapsing Tests ###

def test_collapse_to_binary_automatic_thresholding(sample_gmm_thresholding_instance):
    """Test collapsing 4 components to 2 categories with automatic thresholding."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    # Collapse 4 components into 2 categories
    gmm.categorize_samples(
        ordered_labels=['Low', 'Low', 'High', 'High'],
        duplicate_labels=True
    )
    
    # Should use automatic thresholding
    assert gmm._manual_decision_boundaries == False, (
        "Should use automatic thresholding when manual_thresholds not provided"
    )
    
    # Should have 1 threshold (2 categories)
    thresholds = gmm.return_thresholds()
    assert len(thresholds) == 1, (
        "Should have 1 threshold for 2 categories"
    )
    
    # Should have condensed labels
    assert gmm._internal_data.condensed_labels == ['Low', 'High'], (
        "Condensed labels should match unique labels in order"
    )
    
    # Verify only 2 unique labels in data
    unique_labels = set(gmm.adata.obs['labels'].unique())
    assert unique_labels == {'Low', 'High'}, (
        "Should only have 2 unique labels in assigned data"
    )


def test_collapse_to_binary_manual_thresholding(sample_gmm_thresholding_instance):
    """Test collapsing 4 components to 2 categories with manual thresholding."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    manual_threshold = [1.5]
    gmm.categorize_samples(
        ordered_labels=['Low', 'Low', 'High', 'High'],
        manual_thresholds=manual_threshold,
        duplicate_labels=True
    )
    
    # Should use manual thresholding
    assert gmm._manual_decision_boundaries == True, (
        "Should use manual thresholding when manual_thresholds provided"
    )
    
    # Should use the exact threshold provided
    assert gmm.return_thresholds() == manual_threshold, (
        "Should use the provided manual threshold"
    )
    
    # Should have condensed labels
    assert gmm._internal_data.condensed_labels == ['Low', 'High'], (
        "Condensed labels should match unique labels"
    )
    
    # Verify labels
    unique_labels = set(gmm.adata.obs['labels'].unique())
    assert unique_labels == {'Low', 'High'}, (
        "Should have 2 unique labels"
    )


def test_collapse_to_three_categories(sample_gmm_thresholding_instance):
    """Test collapsing 6 components to 3 categories with automatic thresholding."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=6)
    
    gmm.categorize_samples(
        ordered_labels=['Low', 'Low', 'Medium', 'Medium', 'High', 'High'],
        duplicate_labels=True
    )
    
    # Should have 2 thresholds (3 categories)
    thresholds = gmm.return_thresholds()
    assert len(thresholds) == 2, (
        "Should have 2 thresholds for 3 categories"
    )
    
    # Verify 3 unique labels
    unique_labels = set(gmm.adata.obs['labels'].unique())
    assert unique_labels == {'Low', 'Medium', 'High'}, (
        "Should have exactly 3 unique labels"
    )
    
    # Should use automatic thresholding
    assert gmm._manual_decision_boundaries == False, (
        "Should use automatic thresholding"
    )


def test_collapse_many_to_few(sample_gmm_thresholding_instance):
    """Test collapsing 8 components to 2 categories (cross-dataset use case)."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=8)
    
    # Collapse 8 components to 2 categories
    gmm.categorize_samples(
        ordered_labels=['Low'] * 3 + ['High'] * 5,
        duplicate_labels=True
    )
    
    # Verify automatic thresholding worked
    assert gmm._manual_decision_boundaries == False, (
        "Should use automatic thresholding"
    )
    assert len(gmm.return_thresholds()) == 1, (
        "Should have 1 threshold for 2 categories"
    )
    
    # Verify condensed probabilities were created
    assert gmm._internal_data.gmm_info.condensed_data_probs is not None, (
        "Condensed probabilities should be created"
    )
    assert gmm._internal_data.gmm_info.condensed_data_probs.shape[1] == 2, (
        "Condensed probabilities should have 2 columns for 2 categories"
    )
    
    # Verify labels
    unique_labels = set(gmm.adata.obs['labels'].unique())
    assert unique_labels == {'Low', 'High'}, (
        "Should have 2 unique labels"
    )


### Edge Cases ###

def test_collapse_requires_flag(sample_gmm_thresholding_instance):
    """Test that duplicate labels raise error when duplicate_labels=False."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    with pytest.raises(
        ValueError, 
        match="ordered GMM labels contain duplicate values"
    ):
        gmm.categorize_samples(
            ordered_labels=['Low', 'Low', 'High', 'High'],
            duplicate_labels=False  # or omit (default is False)
        )


def test_collapse_preserves_n_components(sample_gmm_thresholding_instance):
    """Test that n_components is preserved after label collapsing."""
    gmm = sample_gmm_thresholding_instance
    n_components = 8
    gmm.fit(n_components=n_components)
    
    gmm.categorize_samples(
        ordered_labels=['Low'] * 4 + ['High'] * 4,
        duplicate_labels=True
    )
    
    # n_components should still be 8, not changed to 2
    assert gmm._gmm_info.n_components == n_components, (
        f"n_components should remain {n_components} after collapsing, not be mutated"
    )


def test_collapse_creates_condensed_probabilities(sample_gmm_thresholding_instance):
    """Test that condensed probabilities are created during label collapsing."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    gmm.categorize_samples(
        ordered_labels=['Low', 'Low', 'High', 'High'],
        duplicate_labels=True
    )
    
    # Check condensed probabilities exist
    condensed_probs = gmm._internal_data.gmm_info.condensed_data_probs
    assert condensed_probs is not None, (
        "Condensed probabilities should be created"
    )
    
    # Check dimensions
    n_samples = len(gmm.adata)
    n_categories = 2
    assert condensed_probs.shape == (n_samples, n_categories), (
        f"Condensed probabilities should have shape ({n_samples}, {n_categories})"
    )
    
    # Check that probabilities sum to reasonable values
    # (max across categories for each sample)
    max_probs = np.max(condensed_probs, axis=1)
    assert np.all(max_probs <= 1.0), (
        "Probabilities should not exceed 1.0"
    )
    assert np.all(max_probs > 0.0), (
        "Probabilities should be positive"
    )


def test_collapse_with_sequential_groups(sample_gmm_thresholding_instance):
    """Test collapsing with sequential/contiguous duplicate pattern."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=6)
    
    # Sequential pattern: Low, Low, Low, High, High, High
    gmm.categorize_samples(
        ordered_labels=['Low', 'Low', 'Low', 'High', 'High', 'High'],
        duplicate_labels=True
    )
    
    # Should work and create 2 categories
    unique_labels = set(gmm.adata.obs['labels'].unique())
    assert unique_labels == {'Low', 'High'}, (
        "Should handle sequential duplicate patterns"
    )
    
    # Should have 1 threshold
    assert len(gmm.return_thresholds()) == 1, (
        "Should have 1 threshold for 2 unique labels"
    )


def test_collapse_rejects_non_contiguous_groups(sample_gmm_thresholding_instance):
    """Test that non-contiguous duplicate patterns are rejected."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=6)
    
    # Non-contiguous pattern: Low, High, Low, High, Low, High should be rejected
    with pytest.raises(
        ValueError, 
        match="Non-contiguous duplicate labels detected"
    ):
        gmm.categorize_samples(
            ordered_labels=['Low', 'High', 'Low', 'High', 'Low', 'High'],
            duplicate_labels=True
        )


### Validation Tests ###

def test_collapse_validates_threshold_count(sample_gmm_thresholding_instance):
    """Test that manual threshold count must match unique labels - 1."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    # Need 1 threshold for 2 unique labels, providing 2
    with pytest.raises(
        ValueError, 
        match="Number of thresholds.*must be one less than.*unique labels"
    ):
        gmm.categorize_samples(
            ordered_labels=['Low', 'Low', 'High', 'High'],
            manual_thresholds=[1.0, 2.0],  # Too many!
            duplicate_labels=True
        )


def test_collapse_label_count_must_match_components(sample_gmm_thresholding_instance):
    """Test that total label count must match n_components."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    # Providing only 3 labels for 4 components
    with pytest.raises(
        ValueError,
        match="Number of labels.*must equal number of.*fitted GMM components"
    ):
        gmm.categorize_samples(
            ordered_labels=['Low', 'Low', 'High'],  # Only 3 labels!
            duplicate_labels=True
        )


def test_collapse_with_all_same_label(sample_gmm_thresholding_instance):
    """Test edge case where all components get same label."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=4)
    
    gmm.categorize_samples(
        ordered_labels=['Low', 'Low', 'Low', 'Low'],
        duplicate_labels=True
    )
    
    # Should have 0 thresholds (only 1 category)
    thresholds = gmm.return_thresholds()
    assert len(thresholds) == 0, (
        "Should have no thresholds when all components map to one category"
    )
    
    # All samples should get same label
    unique_labels = set(gmm.adata.obs['labels'].unique())
    assert unique_labels == {'Low'}, (
        "All samples should have the same label"
    )


def test_collapse_preserves_order(sample_gmm_thresholding_instance):
    """Test that label order is preserved in condensed labels."""
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=6)
    
    # Order: High first, then Low, then Medium
    gmm.categorize_samples(
        ordered_labels=['High', 'High', 'Low', 'Low', 'Medium', 'Medium'],
        duplicate_labels=True
    )
    
    # Condensed labels should preserve first occurrence order
    assert gmm._internal_data.condensed_labels == ['High', 'Low', 'Medium'], (
        "Condensed labels should preserve the order of first occurrence"
    )
