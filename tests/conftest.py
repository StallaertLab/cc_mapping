import sys
import os

import anndata as ad
import numpy as np
import pytest

from src.cc_mapping.thresholding import GMMThresholding


def pytest_configure():
    """
    Adds the project root directory to the Python path.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Adjust the path as needed
    sys.path.insert(0, project_root)

@pytest.fixture
def sample_adata():
    """Creates a sample AnnData object for testing."""
    np.random.seed(0)
    xx = np.random.normal(loc=1, scale=0.5, size=(1000, 2))
    adata = ad.AnnData(X = xx)
    adata.var_names = ['gene1', 'gene2']
    return adata

@pytest.fixture
def sample_gmm_thresholding_instance(sample_adata):
    """Creates an instance of GMMThresholding for testing."""
    gmm_thresholding_instance = GMMThresholding(adata=sample_adata,
                                                                 feature='gene1',
                                                                 label_obs_save_str='labels',
                                                                )
    return gmm_thresholding_instance