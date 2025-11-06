"""
GMM-based thresholding package for single-cell analysis.

This package provides classes for performing Gaussian Mixture Model (GMM) based
thresholding on gene expression data within AnnData objects. It supports both
single-feature thresholding and sequential refinement operations.

Main Classes:
    GMMThresholding: Single-feature GMM thresholding
    SequentialGMM: Sequential refinement (Phase 2)

Pydantic Models:
    _GaussianMixtureModelInfo: GMM parameters and results storage
    _DecisionBoundariesModel: Decision boundary thresholds storage
    _SingleThresholdingEventModel: Complete thresholding event data

Base Classes:
    GaussianMixtureModelBase: Shared utilities for GMM operations

Usage::

    from cc_mapping.thresholding import GMMThresholding
    
    gmm = GMMThresholding(
        adata=adata,
        feature='gene1',
        label_obs_save_str='gene1_categories'
    )
    gmm.fit(n_components=2)
    gmm.categorize_samples(ordered_labels=['Low', 'High'])
    adata = gmm.return_adata()
"""

from .base import (
    _GaussianMixtureModelInfo,
    _DecisionBoundariesModel,
    _SingleThresholdingEventModel,
    GaussianMixtureModelBase,
)

from .single import GMMThresholding

from .sequential import SequentialGMM


__all__ = [
    # Main classes
    'GMMThresholding',
    'SequentialGMM',
    
    # Base class
    'GaussianMixtureModelBase',
    
    # Pydantic models (private but exposed for advanced usage)
    '_GaussianMixtureModelInfo',
    '_DecisionBoundariesModel',
    '_SingleThresholdingEventModel',
]

__version__ = '0.1.0'
