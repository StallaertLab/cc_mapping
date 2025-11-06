"""Tests for the fit method of GMMThresholding.

This module tests GMM fitting functionality, including successful fits,
validation of n_components parameter, and proper storage of fit results.
"""

import pytest
from src.cc_mapping.thresholding import GMMThresholding


def test_fit_success(sample_adata, sample_gmm_thresholding_instance):
    """Test if the fit method works correctly."""
    n_components = 2
    gmm = sample_gmm_thresholding_instance
    gmm.fit(n_components=n_components)

    assert (
        gmm._gmm_info.data_probs is not None
    ), "GMM probabilities should not be None after fitting."
    assert len(gmm._gmm_info.data_probs) == len(sample_adata), (
        "Length of GMM probabilities should match the number of samples."
    )
    assert len(gmm._gmm_info.data_probs[0]) == n_components, (
        "Number of GMM probabilities should match the number of components."
    )
    assert (
        gmm._internal_data.gmm_info == gmm._gmm_info
    ), "Internal GMM info should match the fitted GMM info."


@pytest.mark.parametrize(
    "invalid_n_components, expected_exception, match_pattern",
    [
        (None, TypeError, "n_components must be a positive integer"),
        ("", TypeError, "n_components must be a positive integer"),
        ([], TypeError, "n_components must be a positive integer"),
        (-1, ValueError, "n_components must be a positive integer"),
    ],
)
def test_fit_invalid_n_components(
    sample_gmm_thresholding_instance,
    invalid_n_components,
    expected_exception,
    match_pattern,
):
    """Test if the GMMThresholding raises exceptions for invalid n_components type."""
    gmm = sample_gmm_thresholding_instance
    with pytest.raises(expected_exception, match=match_pattern):
        gmm.fit(n_components=invalid_n_components)
