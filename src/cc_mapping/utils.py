"""
Utility functions for the cc_mapping package.
"""

from __future__ import annotations

from typing import List

import anndata as ad
import numpy as np
import pandas as pd


def create_boolean_label_combination(
    adata: ad.AnnData,
    obs_key_1: str,
    match_values_1: List[str],
    obs_key_2: str,
    match_values_2: List[str],
    operator: str,
    output_obs_key: str,
    true_label: str,
    false_label: str,
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Combine two categorical observation columns using boolean operators.
    
    Creates a new binary label based on whether cells match specified values
    in both input observation columns, using the specified boolean operator.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with observations to combine.
    obs_key_1 : str
        First observation column name in adata.obs.
    match_values_1 : List[str]
        Values in obs_key_1 to match (considered "true" for boolean logic).
    obs_key_2 : str
        Second observation column name in adata.obs.
    match_values_2 : List[str]
        Values in obs_key_2 to match (considered "true" for boolean logic).
    operator : str
        Boolean operator - 'AND', 'OR', or 'XOR'.
    output_obs_key : str
        Name for new combined observation column.
    true_label : str
        Label for cells matching the boolean criteria.
    false_label : str
        Label for cells not matching the boolean criteria.
    overwrite : bool, default False
        If True, overwrites existing output_key. If False, raises error if 
        output_key exists.
        
    Returns
    -------
    ad.AnnData
        Modified AnnData object with new observation column.
        
    Raises
    ------
    KeyError
        If obs_key_1 or obs_key_2 don't exist in adata.obs.
    ValueError
        If operator is not 'AND', 'OR', or 'XOR'.
    KeyError
        If output_obs_key already exists in adata.obs and overwrite=False.
    TypeError
        If match_values_1 or match_values_2 are not lists.
    ValueError
        If any values in match_values_1 not found in obs_key_1.
    ValueError
        If any values in match_values_2 not found in obs_key_2.
        
    Examples
    --------
    AND: Both conditions must be true
    
    >>> adata = create_boolean_label_combination(
    ...     adata,
    ...     obs_key_1='treatment',
    ...     match_values_1=['control'],
    ...     obs_key_2='cell_cycle',
    ...     match_values_2=['G0'],
    ...     operator='AND',
    ...     output_obs_key='control_G0',
    ...     true_label='control_G0',
    ...     false_label='other'
    ... )
    
    OR: Either condition true
    
    >>> adata = create_boolean_label_combination(
    ...     adata,
    ...     obs_key_1='treatment',
    ...     match_values_1=['control', 'vehicle'],
    ...     obs_key_2='cell_cycle',
    ...     match_values_2=['G0', 'G1'],
    ...     operator='OR',
    ...     output_obs_key='quiescent_or_control',
    ...     true_label='positive',
    ...     false_label='other'
    ... )
    
    XOR: Exactly one condition true (exclusive or)
    
    >>> adata = create_boolean_label_combination(
    ...     adata,
    ...     obs_key_1='marker_A',
    ...     match_values_1=['positive'],
    ...     obs_key_2='marker_B',
    ...     match_values_2=['positive'],
    ...     operator='XOR',
    ...     output_obs_key='single_positive',
    ...     true_label='single_positive',
    ...     false_label='double_or_negative'
    ... )
    """
    # Validate observation columns exist
    if obs_key_1 not in adata.obs.columns:
        raise KeyError(
            f"obs_key_1 '{obs_key_1}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    if obs_key_2 not in adata.obs.columns:
        raise KeyError(
            f"obs_key_2 '{obs_key_2}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    # Validate operator
    valid_operators = ['AND', 'OR', 'XOR']
    operator = operator.upper()
    if operator not in valid_operators:
        raise ValueError(
            f"operator must be one of {valid_operators}, got '{operator}'"
        )
    
    # Validate output_obs_key doesn't already exist (unless overwrite=True)
    if output_obs_key in adata.obs.columns and not overwrite:
        raise KeyError(
            f"output_obs_key '{output_obs_key}' already exists in adata.obs. "
            "Set overwrite=True to replace it, or choose a different name."
        )
    
    # Validate match values are lists
    if not isinstance(match_values_1, list):
        raise TypeError(
            f"match_values_1 must be a list, got {type(match_values_1)}"
        )
    
    if not isinstance(match_values_2, list):
        raise TypeError(
            f"match_values_2 must be a list, got {type(match_values_2)}"
        )
    
    # Validate all values exist in their respective observation columns
    unique_obs_1 = set(adata.obs[obs_key_1].unique())
    for val in match_values_1:
        if val not in unique_obs_1:
            raise ValueError(
                f"Value '{val}' not found in obs_key_1 '{obs_key_1}'. "
                f"Available values: {sorted(unique_obs_1)}"
            )
    
    unique_obs_2 = set(adata.obs[obs_key_2].unique())
    for val in match_values_2:
        if val not in unique_obs_2:
            raise ValueError(
                f"Value '{val}' not found in obs_key_2 '{obs_key_2}'. "
                f"Available values: {sorted(unique_obs_2)}"
            )
    
    # Create boolean masks
    mask1 = adata.obs[obs_key_1].isin(match_values_1)
    mask2 = adata.obs[obs_key_2].isin(match_values_2)
    
    # Apply boolean operator
    if operator == 'AND':
        final_mask = mask1 & mask2
    elif operator == 'OR':
        final_mask = mask1 | mask2
    elif operator == 'XOR':
        final_mask = mask1 ^ mask2
    
    # Create new categorical column
    new_labels = np.where(final_mask, true_label, false_label)
    adata.obs[output_obs_key] = pd.Categorical(new_labels)
    
    return adata