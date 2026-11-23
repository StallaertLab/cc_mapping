import re

import anndata as ad
import numpy as np


def row_data_partitioning(
    adata: ad.AnnData,
    search_str: str,
    search_obs: str,
    regex: bool = False,
    regex_flags: list[str] = None,
    reset_idx: bool = True,
):
    """
    Partition the rows of the input AnnData object based on a search string and observation column.

    Args:
        adata (ad.AnnData): The input AnnData object.
        search_str (str): The search string used to filter the rows.
        search_obs (str): The name of the observation column used for filtering.
        regex (bool, optional): Whether to use regular expressions for searching. Defaults to False.
        regex_flags (list[str], optional): List of regex flags to use if regex is True. Defaults to None.
        reset_idx (bool, optional): Whether to reset the index of the resulting AnnData object. Defaults to True.

    Returns:
        ad.AnnData: The partitioned AnnData object.
    """
    obs_values = adata.obs[search_obs]

    if regex:
        # Build regex flags
        flags = 0
        if regex_flags:
            for flag in regex_flags:
                flags |= getattr(re, flag)

        # Handle single pattern or multiple patterns
        if isinstance(search_str, str):
            pattern = search_str
        else:
            # Combine multiple patterns with OR
            pattern = "|".join(search_str)

        mask = obs_values.astype(str).str.contains(
            pattern, regex=True, flags=flags, na=False
        )
    else:
        # Use native pandas isin() for exact matching
        if isinstance(search_str, str):
            search_str = [search_str]
        mask = obs_values.isin(search_str)

    adata = adata[mask, :].copy()

    if reset_idx is True:
        adata.obs.index = np.arange(adata.shape[0]).astype(str)

    return adata
