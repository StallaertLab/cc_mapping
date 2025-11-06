"""
cc_mapping: GMM-based thresholding for single-cell analysis.

This package provides Gaussian Mixture Model (GMM) based thresholding
for gene expression data within AnnData objects. It supports both
single-feature thresholding and sequential refinement operations.

Main functionality:
- GMMThresholding: Single-feature GMM thresholding
- SequentialGMM: Sequential refinement across multiple features
- create_boolean_label_combination: Boolean operations on categorical labels
"""

from .thresholding import GMMThresholding, SequentialGMM
from .utils import create_boolean_label_combination

__all__ = [
    "GMMThresholding",
    "SequentialGMM",
    "create_boolean_label_combination",
]

__version__ = "0.1.0"