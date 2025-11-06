from __future__ import annotations

import re
import anndata as ad
from collections import Counter
from math import floor
from typing import List

import numpy as np
import pandas as pd


def get_str_idx(
    strings_to_find: str | list[str] | np.ndarray[str],
    string_list: list[str] | np.ndarray[str],
    regex: bool = False,
    regex_flags: list[str] = None,
) -> list[np.ndarray]:
    """
    Takes in a string or list of strings and returns the indices and the names of the matching strings in the string list.
    Regex can be used to find strings that match a pattern.

    If the string list contains no duplicates, then a dictionary is used to speed up the search when searching for multiple strings.

    Args:
        strings_to_find (str | list[str] | np.ndarray[str]): A string or iterable of strings to search for.
        string_list (list[str] | np.ndarray[str]): An iterable of strings to search through.
        regex (bool, optional): Whether to use regular expressions for searching. Defaults to False.
        regex_flags (list[str], optional): List of regex flags to use if regex is True. Defaults to None.

    Raises:
        ValueError: If the search array contains duplicate strings.
        ValueError: If the string list contains non-string elements.

    Returns:
        list[np.ndarray]: Indices and names of the matching strings in the string list.
    """
    if regex:
        reFlags = []
        for flag in regex_flags:
            reFlags.append(getattr(re, flag))

        search_func = lambda regex, string_to_search: re.search(
            regex, string_to_search, *reFlags
        )
    else:
        search_func = (
            lambda matching_string, string_to_search: matching_string
            == string_to_search
        )

    if isinstance(strings_to_find, str):
        strings_to_find = [strings_to_find]

    if np.unique(strings_to_find).shape[0] != len(strings_to_find):
        raise ValueError("Search array of strings contains duplicate strings")

    if not np.all([isinstance(string, str) for string in string_list]):
        raise ValueError("String list must contain only strings")

    # if the string list contains no duplicates, then we can use a dictionary to speed up the search
    if np.unique(string_list).shape[0] == len(string_list) and not regex:
        string_list_dict = {string: idx for idx, string in enumerate(string_list)}
        feat_idx_names = np.array(
            [
                [string_list_dict[string], string]
                for string in strings_to_find
                if string in string_list_dict
            ]
        )
    else:
        match_list = []

        # creates a search function based on whether or not the user wants to use regex that returns true if there is a match
        for string in strings_to_find:
            feat_idx_names = [
                [idx, item]
                for idx, item in enumerate(string_list)
                if search_func(string, item)
            ]

            if feat_idx_names != []:
                match_list.append(feat_idx_names)

        if len(match_list) == 0:
            raise KeyError("No Matching Values")

        feat_idx_names = np.vstack(match_list)

    feature_idxs, feature_names = feat_idx_names[:, 0].astype(int), feat_idx_names[:, 1]
    return feature_idxs, feature_names


def equalize_conditions(adata: ad.AnnData, obs_str: str, ignore_min_list: list[str] = None) -> ad.AnnData:
    """
    Equalizes the conditions in the given AnnData object based on the specified observation string.

    Args:
        adata (ad.AnnData): The AnnData object containing the data.
        obs_str (str): The observation string specifying the condition to equalize.
        ignore_min_list (list[str], optional): A list of observation values to ignore when equalizing. Defaults to None.

    Returns:
        ad.AnnData: The modified AnnData object with equalized conditions.
    """
    obs_values = adata.obs[obs_str].copy()

    obs_counts = Counter(obs_values)

    if ignore_min_list is not None:
        for obs_val in ignore_min_list:
            obs_counts.pop(obs_val)

    min_obs_count = np.min(list(obs_counts.values()))

    idx_list = []
    for obs_val in obs_counts.keys():
        obs_idxs, _ = get_str_idx(obs_val, obs_values)

        np.random.seed(0)
        selected_obs_idxs = list(
            np.random.choice(obs_idxs, size=min_obs_count, replace=False)
        )
        idx_list.extend(selected_obs_idxs)

    adata = adata[idx_list, :].copy()

    return adata


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


def equalize_within_two_conditions(
    adata: ad.AnnData, first_obs_str: str, second_obs_str: str, ignore_min_list: list[str] = None
) -> ad.AnnData:
    """
    Equalizes the number of observations within two conditions in a single-cell dataset.

    Args:
        adata (ad.AnnData): Annotated data matrix.
        first_obs_str (str): Name of the first condition.
        second_obs_str (str): Name of the second condition.
        ignore_min_list (list[str], optional): List of observation values to ignore when calculating the minimum count. Defaults to None.

    Returns:
        ad.AnnData: Annotated data matrix with equalized observations.
    """
    f_obs_values = adata.obs[first_obs_str].copy()
    f_obs_counts = Counter(f_obs_values)
    unique_f_obs_values = sorted(f_obs_counts.keys())

    if ignore_min_list is not None:
        for obs_val in ignore_min_list:
            f_obs_counts.pop(obs_val)

    first_min_obs_count = np.min(f_obs_counts.values())

    s_obs_values = adata.obs[second_obs_str].copy()
    s_obs_counts = Counter(s_obs_values)
    num_unique_s_obs_values = len(s_obs_counts.keys())

    f_obs_per_s_obs = int(first_min_obs_count / num_unique_s_obs_values)

    fs_obs_array = np.array([f_obs_values, s_obs_values], dtype=str).T

    idx_list = []
    for f_obs in unique_f_obs_values:
        f_obs_idxs, _ = get_str_idx(f_obs, fs_obs_array[:, 0])
        single_f_obs_array = fs_obs_array[f_obs_idxs, :]

        s_obs_counts = Counter(single_f_obs_array[:, 1])

        s_obs_counts_keys = list(s_obs_counts.keys())
        s_obs_counts_values = np.array(list(s_obs_counts.values()))

        argsorted_s_obs_counts = np.argsort(s_obs_counts_values)

        sorted_s_obs_counts_values = s_obs_counts_values[argsorted_s_obs_counts]
        sorted_s_obs_counts_keys = s_obs_counts_keys[argsorted_s_obs_counts]

        for idx, (s_obs_key, s_obs_count) in enumerate(
            zip(sorted_s_obs_counts_keys, sorted_s_obs_counts_values, strict=False)
        ):
            s_obs_idxs, _ = get_str_idx(s_obs_key, single_f_obs_array[:, 1])

            if s_obs_count > f_obs_per_s_obs:
                num_idxs_to_add = f_obs_per_s_obs

            else:
                num_idxs_to_add = s_obs_count

                difference = f_obs_per_s_obs - s_obs_count

                # if a s_obs_key does not have enough observations, then we need to add the difference to the total needed from the other s_obs_keys
                # TODO: this can lead to a situation where the last s_obs_key may not have enough observations to equalize the conditions
                f_obs_per_s_obs += floor(difference / num_unique_s_obs_values - idx)

            np.random.seed(0)
            selected_obs_idxs = list(
                np.random.choice(s_obs_idxs, size=num_idxs_to_add, replace=False)
            )
            idx_list.extend(selected_obs_idxs)

    adata = adata[idx_list, :].copy()

    return adata