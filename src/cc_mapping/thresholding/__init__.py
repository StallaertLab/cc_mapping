"""
GMM-based thresholding package for single-cell analysis.

This package provides classes for performing Gaussian Mixture Model (GMM) based
thresholding on gene expression data within AnnData objects. It supports both
single-feature thresholding and sequential refinement operations.

Main Classes:
    GaussianMixtureModelThresholding: Single-feature GMM thresholding
    SequentialGaussianMixtureModelThresholding: Sequential refinement (Phase 2)
    
Pydantic Models:
    _GaussianMixtureModelInfo: GMM parameters and results storage
    _DecisionBoundariesModel: Decision boundary thresholds storage
    _SingleThresholdingEventModel: Complete thresholding event data
    
Base Classes:
    GaussianMixtureModelBase: Shared utilities for GMM operations

Usage:
    from cc_mapping.thresholding import GaussianMixtureModelThresholding
    
    gmm = GaussianMixtureModelThresholding(
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

from .single import GaussianMixtureModelThresholding

from .sequential import SequentialGaussianMixtureModelThresholding

from .utils import (
    create_boolean_label_combination,
    generate_thresholding_report,
)


__all__ = [
    # Main classes
    'GaussianMixtureModelThresholding',
    'SequentialGaussianMixtureModelThresholding',
    
    # Base class
    'GaussianMixtureModelBase',
    
    # Utility functions
    'create_boolean_label_combination',
    'generate_thresholding_report',
    
    # Pydantic models (private but exposed for advanced usage)
    '_GaussianMixtureModelInfo',
    '_DecisionBoundariesModel',
    '_SingleThresholdingEventModel',
]

__version__ = '2.0.0'  # Updated with Phase 1 refactoring
