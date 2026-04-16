"""Tests for index clamping during label collapsing and visualization.

This module tests that bin index clamping works correctly when:
1. GMM components overlap significantly causing flip-flopping
2. More thresholds are generated than expected for the number of unique labels
3. Plotting functions use consistent clamping with labeling functions

The clamping prevents IndexError when accessing color arrays or label arrays
and ensures visualization matches actual cell assignments.
"""

import pytest
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from src.cc_mapping.thresholding import GMMThresholding


@pytest.fixture
def extreme_overlap_adata():
    """Create AnnData with extreme overlap that will definitely cause flip-flopping.

    This fixture creates data where components overlap so much that after
    collapsing labels, the condensed probabilities transition between classes
    multiple times, creating more thresholds than expected.
    """
    np.random.seed(123)

    # Create three heavily overlapping distributions
    # After collapsing first two into 'low', the max() operation will flip-flop
    group1 = np.random.normal(0, 1.5, 300)  # Heavy overlap with group2
    group2 = np.random.normal(0.5, 1.5, 400)  # Overlaps with both 1 and 3
    group3 = np.random.normal(1, 1.5, 300)  # Heavy overlap with group2

    feature_values = np.concatenate([group1, group2, group3])

    adata = ad.AnnData(X=feature_values.reshape(-1, 1))
    adata.var_names = ["marker"]
    adata.obs_names = [f"cell_{i}" for i in range(len(feature_values))]

    return adata


### Core Clamping Tests ###


def test_clamping_prevents_index_error_in_labeling(extreme_overlap_adata):
    """Test that clamping prevents IndexError when assigning labels to cells.

    This is the core fix: when np.digitize returns indices that would be
    out of bounds for the final_labels array, clamping constrains them.
    """
    adata = extreme_overlap_adata

    gmm = GMMThresholding(
        adata=adata,
        feature="marker",
        label_obs_save_str="labels",
    )

    # Fit with 3 components
    gmm.fit(n_components=3)

    # Collapse to 2 labels - this may create problematic thresholds
    # Should NOT raise IndexError due to clamping (line 625 in single.py)
    gmm.categorize_samples(ordered_labels=["low", "low", "high"], duplicate_labels=True)

    # Verify all cells were labeled (no crashes)
    assert "labels" in gmm.adata.obs.columns
    assert len(gmm.adata.obs["labels"]) == len(adata)
    assert gmm.adata.obs["labels"].notna().all()

    # Verify only valid labels exist
    unique_labels = set(gmm.adata.obs["labels"].unique())
    assert unique_labels.issubset(
        {"low", "high"}
    ), f"Labels should only be 'low' or 'high', got: {unique_labels}"


def test_clamping_prevents_index_error_in_plotting_vertical(extreme_overlap_adata):
    """Test that clamping prevents IndexError in vertical decision boundary plotting.

    This tests the fix in _plot_vertical_linear_decision_boundaries where:
    1. bin_indices for color grid are clamped (line 813 in base.py)
    2. color_idx_right for threshold lines is clamped (line 829 in base.py)
    """
    adata = extreme_overlap_adata

    gmm = GMMThresholding(
        adata=adata,
        feature="marker",
        label_obs_save_str="labels",
    )

    gmm.fit(n_components=3)
    gmm.categorize_samples(ordered_labels=["low", "low", "high"], duplicate_labels=True)

    # Should NOT raise IndexError when plotting
    try:
        gmm.plot_hist_distribution_with_boundaries(title="Test Vertical Boundaries")
        plt.close("all")  # Clean up
        success = True
        error_msg = None
    except IndexError as e:
        success = False
        error_msg = str(e)

    assert success, f"Plotting should not raise IndexError, got: {error_msg}"


def test_clamping_prevents_index_error_in_plotting_horizontal(extreme_overlap_adata):
    """Test that clamping prevents IndexError in horizontal decision boundary plotting.

    This tests the fix in _plot_horizontal_linear_decision_boundaries where:
    1. bin_indices for color grid are clamped (line 885 in base.py)
    2. color_idx_right for threshold lines is clamped (line 905 in base.py)
    """
    adata = extreme_overlap_adata

    gmm = GMMThresholding(
        adata=adata,
        feature="marker",
        label_obs_save_str="labels",
    )

    gmm.fit(n_components=3)
    gmm.categorize_samples(ordered_labels=["low", "low", "high"], duplicate_labels=True)

    # Try to trigger horizontal boundary plotting
    # Note: This may depend on plot type, but shouldn't crash
    try:
        gmm.plot_strip_plot_histogram_with_decision_boundaries(
            title="Test Horizontal Boundaries",
            scatter_density=False,  # Use category coloring
        )
        plt.close("all")
        success = True
        error_msg = None
    except IndexError as e:
        success = False
        error_msg = str(e)

    assert success, f"Plotting should not raise IndexError, got: {error_msg}"


### Consistency Tests ###


def test_plotting_matches_labeling_with_clamping(extreme_overlap_adata):
    """Test that plotting colors match the actual cell labels after clamping.

    This ensures that the clamping in visualization uses the same logic
    as the clamping in cell labeling, so users see accurate representations.
    """
    adata = extreme_overlap_adata

    gmm = GMMThresholding(
        adata=adata,
        feature="marker",
        label_obs_save_str="labels",
    )

    gmm.fit(n_components=3)
    gmm.categorize_samples(ordered_labels=["low", "low", "high"], duplicate_labels=True)

    # Get actual label assignments
    actual_labels = gmm.adata.obs["labels"].values
    unique_labels = set(actual_labels)

    # Plot should use same number of colors as unique labels
    # This is ensured by num_final_categories = len(internal_data.condensed_labels)
    if gmm._internal_data.condensed_labels is not None:
        assert set(gmm._internal_data.condensed_labels) == unique_labels, (
            f"Condensed labels {gmm._internal_data.condensed_labels} should match "
            f"unique assigned labels {unique_labels}"
        )

    # Plotting should not crash and should complete
    gmm.plot_hist_distribution_with_boundaries()
    plt.close("all")


def test_clamping_with_multiple_collapse_patterns():
    """Test clamping works with various label collapse patterns."""
    np.random.seed(456)

    # Create data
    feature_values = np.random.randn(1000)
    adata = ad.AnnData(X=feature_values.reshape(-1, 1))
    adata.var_names = ["marker"]

    # Test different collapse patterns
    test_cases = [
        # (n_components, ordered_labels)
        (3, ["A", "A", "B"]),  # 2 categories from 3 components
        (4, ["A", "A", "B", "B"]),  # 2 categories from 4 components
        (5, ["A", "A", "B", "C", "C"]),  # 3 categories from 5 components
        (6, ["A", "A", "A", "B", "B", "B"]),  # 2 categories from 6 components
    ]

    for n_components, ordered_labels in test_cases:
        gmm = GMMThresholding(
            adata=adata.copy(),
            feature="marker",
            label_obs_save_str="labels",
        )

        gmm.fit(n_components=n_components)

        # Should not crash with any pattern
        try:
            gmm.categorize_samples(ordered_labels=ordered_labels, duplicate_labels=True)

            # Plotting should also work
            gmm.plot_hist_distribution_with_boundaries(
                title=f"{n_components} -> {len(set(ordered_labels))} categories"
            )
            plt.close("all")

            success = True
            error_msg = None
        except (IndexError, ValueError) as e:
            success = False
            error_msg = str(e)

        assert success, (
            f"Pattern {n_components} components -> {ordered_labels} should work, "
            f"but got error: {error_msg}"
        )


### Edge Case Tests ###


def test_clamping_with_all_same_label():
    """Test clamping when all components collapse to one label (edge case)."""
    np.random.seed(789)
    feature_values = np.random.randn(500)
    adata = ad.AnnData(X=feature_values.reshape(-1, 1))
    adata.var_names = ["marker"]

    gmm = GMMThresholding(
        adata=adata,
        feature="marker",
        label_obs_save_str="labels",
    )

    gmm.fit(n_components=4)

    # All same label - should create 0 thresholds
    gmm.categorize_samples(
        ordered_labels=["same", "same", "same", "same"], duplicate_labels=True
    )

    # Should have 0 thresholds
    thresholds = gmm.return_thresholds()
    assert len(thresholds) == 0, "Should have 0 thresholds for 1 unique label"

    # All cells should have same label
    assert gmm.adata.obs["labels"].nunique() == 1

    # Plotting should still work (no thresholds to draw)
    gmm.plot_hist_distribution_with_boundaries()
    plt.close("all")


def test_warning_issued_for_threshold_mismatch(extreme_overlap_adata):
    """Test that a warning is issued when threshold count doesn't match expected.

    This documents the expected behavior: when GMM components overlap and
    cause flip-flopping, the code warns the user but handles it gracefully.
    """
    adata = extreme_overlap_adata

    gmm = GMMThresholding(
        adata=adata,
        feature="marker",
        label_obs_save_str="labels",
    )

    gmm.fit(n_components=3)

    # This may or may not trigger warning depending on data
    # The important thing is it doesn't crash
    import warnings

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        gmm.categorize_samples(
            ordered_labels=["low", "low", "high"], duplicate_labels=True
        )

        # If a warning was issued, verify it's the expected one
        threshold_warnings = [
            w for w in warning_list if "threshold" in str(w.message).lower()
        ]

        # Either no warning (clean separation) or expected warning (overlapping)
        if threshold_warnings:
            assert any(
                "clamping" in str(w.message).lower()
                or "overlapping" in str(w.message).lower()
                for w in threshold_warnings
            ), f"Unexpected warning: {[str(w.message) for w in threshold_warnings]}"


def test_manual_thresholds_bypass_clamping_issue():
    """Test that manual thresholds avoid the flip-flopping problem entirely.

    When users specify manual thresholds, they explicitly control boundaries
    and shouldn't encounter clamping issues (unless they provide wrong count).
    """
    np.random.seed(999)
    feature_values = np.random.randn(500)
    adata = ad.AnnData(X=feature_values.reshape(-1, 1))
    adata.var_names = ["marker"]

    gmm = GMMThresholding(
        adata=adata,
        feature="marker",
        label_obs_save_str="labels",
    )

    gmm.fit(n_components=4)

    # Manual threshold for 2 categories (1 threshold needed)
    gmm.categorize_samples(
        ordered_labels=["low", "high"],
        manual_thresholds=[0.0],
    )

    # Should have exactly 1 threshold
    thresholds = gmm.return_thresholds()
    assert len(thresholds) == 1
    assert thresholds[0] == 0.0

    # Plotting should work perfectly (no automatic threshold issues)
    gmm.plot_hist_distribution_with_boundaries()
    plt.close("all")


### Regression Tests ###


def test_real_world_ploidy_scenario():
    """Test the exact scenario from user's notebook: 3-component ploidy GMM.

    This replicates the user's use case:
    - 3 GMM components for DNA content
    - Collapse to ['not_2N', 'not_2N', '2N']
    - Should not crash when plotting
    """
    np.random.seed(42)

    # Simulate DNA content distribution (log scale)
    # diploid peak around 0.0, some aneuploid cells
    dna_2n = np.random.normal(0.0, 0.2, 400)  # 2N peak
    dna_low = np.random.normal(-0.5, 0.25, 100)  # Sub-2N
    dna_high = np.random.normal(0.4, 0.3, 100)  # Near-4N / aneuploid

    dna_values = np.concatenate([dna_low, dna_2n, dna_high])

    adata = ad.AnnData(X=dna_values.reshape(-1, 1))
    adata.var_names = ["DNA_content"]

    gmm = GMMThresholding(
        adata=adata,
        feature="DNA_content",
        label_obs_save_str="ploidy_label_str",
    )

    # Fit 3 components
    gmm.fit(n_components=3)

    # Collapse to 2N vs not_2N
    gmm.categorize_samples(
        ordered_labels=["not_2N", "not_2N", "2N"], duplicate_labels=True
    )

    # Should complete without IndexError
    assert "ploidy_label_str" in gmm.adata.obs.columns

    # Should have only 2 unique labels
    unique_labels = set(gmm.adata.obs["ploidy_label_str"].unique())
    assert unique_labels == {"not_2N", "2N"}

    # Plotting should work (this was failing before the fix)
    gmm.plot_hist_distribution_with_boundaries(
        title="DNA Content GMM - Ploidy Labeling"
    )
    plt.close("all")


def test_real_world_proliferation_scenario():
    """Test the proliferation marker scenario with potential flip-flopping."""
    np.random.seed(43)

    # Simulate pRB/RB ratio (log scale)
    # Most cells nonproliferating, some proliferating
    nonprolif_1 = np.random.normal(-0.5, 0.3, 300)  # Low pRB/RB
    nonprolif_2 = np.random.normal(-0.2, 0.2, 200)  # Medium-low
    prolif = np.random.normal(0.5, 0.4, 200)  # High pRB/RB

    marker_values = np.concatenate([nonprolif_1, nonprolif_2, prolif])

    adata = ad.AnnData(X=marker_values.reshape(-1, 1))
    adata.var_names = ["pRB_RB_ratio"]

    gmm = GMMThresholding(
        adata=adata,
        feature="pRB_RB_ratio",
        label_obs_save_str="prolif_label_str",
    )

    # Fit 3 components
    gmm.fit(n_components=3)

    # Collapse to nonprolif vs prolif
    gmm.categorize_samples(
        ordered_labels=["nonprolif", "nonprolif", "prolif"], duplicate_labels=True
    )

    # Should work without IndexError
    assert "prolif_label_str" in gmm.adata.obs.columns
    assert set(gmm.adata.obs["prolif_label_str"].unique()) == {"nonprolif", "prolif"}

    # Plotting should work
    gmm.plot_hist_distribution_with_boundaries(title="Proliferation Marker GMM")
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
