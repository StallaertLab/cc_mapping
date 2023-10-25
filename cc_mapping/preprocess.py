import numpy as np
import os
import scipy.stats as st
import pandas as pd
import anndata as ad

import sys
sys.path.append(os.getcwd())

from GLOBAL_VARIABLES.GLOBAL_VARIABLES import cc_mapping_package_dir
sys.path.append(cc_mapping_package_dir)

from cc_mapping.utils import get_str_idx

def row_data_partitioning(adata: ad.AnnData,
                            search_str: str,
                            search_obs: str,
                            reset_idx: bool  = True):

    search_idxs, _ = get_str_idx(search_str, adata.obs[search_obs])

    adata = adata[search_idxs,:].copy()

    if reset_idx == True:
        adata.obs.index =np.arange(adata.shape[0]).astype(str)

    return adata
