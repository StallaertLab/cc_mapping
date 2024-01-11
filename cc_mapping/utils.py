import numpy as np
from collections import Counter
from math import floor
import re
from typing import Union, List

def get_str_idx(strings_to_find: Union[ str, List[str], np.ndarray[str] ],
                string_list: Union[ List[str], np.ndarray[str] ],
                regex: bool =False) -> (np.ndarray, np.ndarray):

    """ Takes in a string or list of strings and returns the indices and 
        the names of the matching strings in the string list
        Regex can be used to find strings that match a pattern

        If the string list contains no duplicates, then a dictionary is used to speed up the search 
        when searching for multiple strings

    Args:
        str_to_find (str, list, np.ndarray): a str or iterable of strings to search for
        string_list (list, np.ndarray): a iterable of strings to search through
        regex (bool, optional): bool to control the use of regular expressions during the search. Defaults to False.

    Raises:
        ValueError: if the search array contains duplicate strings

    Returns:
        (np.ndarray, np.ndarray): indices and names of the matching strings in the string list
    """
    
    if regex:
        search_func = lambda regex, string_to_search: re.search(regex, string_to_search)
    else: 
        search_func = lambda matching_string, string_to_search: matching_string == string_to_search

    if type(strings_to_find) == str:
        strings_to_find = [strings_to_find]

    if np.unique(strings_to_find).shape[0] != len(strings_to_find):
        raise ValueError('Search array of strings contains duplicate strings')
    
    #if the string list contains no duplicates, then we can use a dictionary to speed up the search
    if np.unique(string_list).shape[0] == len(string_list) and not regex:

        string_list_dict = {string:idx for idx, string in enumerate(string_list)}
        feat_idx_names = np.array([[string_list_dict[string],string] for string in strings_to_find if string in string_list_dict.keys()])   
    else:
        match_list = []

        #creates a search function based on whether or not the user wants to use regex that returns true if there is a match
        for string in strings_to_find:

            feat_idx_names = [[idx,string_list[idx]] for idx, item in enumerate(string_list) if search_func(string, item)]

            if feat_idx_names != []:
                match_list.append(feat_idx_names)

        feat_idx_names = np.vstack(match_list)
        
    return feat_idx_names[:,0].astype(int), feat_idx_names[:,1]

def equalize_conditions(adata, obs_str, ignore_min_list = None):

    obs_values = adata.obs[obs_str].copy()

    obs_counts = Counter(obs_values)

    if ignore_min_list is not None:
        for obs_val in ignore_min_list:
            obs_counts.pop(obs_val)

    min_obs_count = np.min(list(obs_counts.values()))

    idx_list = []
    for obs_val in obs_counts.keys():

        obs_idxs, _ = utils.get_str_idx(obs_val, obs_values)

        np.random.seed(0)   
        selected_obs_idxs = list(np.random.choice(obs_idxs, size = min_obs_count, replace = False))
        idx_list.extend(selected_obs_idxs)

        
    adata = adata[idx_list,:].copy()

    return adata

def equalize_within_two_conditions(adata, first_obs_str, second_obs_str, ignore_min_list = None):

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

    f_obs_per_s_obs = int(first_min_obs_count/num_unique_s_obs_values)

    fs_obs_array = np.array([f_obs_values, s_obs_values],dtype=str).T

    idx_list = []
    for f_obs in unique_f_obs_values:

        f_obs_idxs, _ = utils.get_str_idx(f_obs, fs_obs_array[:,0])
        single_f_obs_array = fs_obs_array[f_obs_idxs,:]

        s_obs_counts = Counter(single_f_obs_array[:,1])

        s_obs_counts_keys = list(s_obs_counts.keys())
        s_obs_counts_values = np.array(list(s_obs_counts.values()))

        argsorted_s_obs_counts = np.argsort(s_obs_counts_values)

        sorted_s_obs_counts_values = s_obs_counts_values[argsorted_s_obs_counts]
        sorted_s_obs_counts_keys = s_obs_counts_keys[argsorted_s_obs_counts]

        for idx, (s_obs_key, s_obs_count) in enumerate(zip(sorted_s_obs_counts_keys, sorted_s_obs_counts_values)):

            s_obs_idxs, _ = utils.get_str_idx(s_obs_key, single_f_obs_array[:,1])

            if s_obs_count > f_obs_per_s_obs:

                num_idxs_to_add = f_obs_per_s_obs

            else:

                num_idxs_to_add = s_obs_count

                difference = f_obs_per_s_obs - s_obs_count

                #if a s_obs_key does not have enought observations, then we need to add the differece to the total needed from the other s_obs_keys
                #TODO: this can lead to a situation where the last s_obs_key may not have enough observations to equalize the conditions
                f_obs_per_s_obs += floor(difference/num_unique_s_obs_values-idx)

            np.random.seed(0)   
            selected_obs_idxs = list(np.random.choice(s_obs_idxs, size = num_idxs_to_add, replace = False))
            idx_list.extend(selected_obs_idxs)
                    
    adata = adata[idx_list,:].copy()

    return adata