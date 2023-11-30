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
    if np.unique(string_list).shape[0] == len(string_list):

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


def equalize_cell_lines(adata, equalize_term, exclude_list = ['PANC1'], ignore_min_list = []):

    #ensures the PANC1 cells are not included in the equalization
    cell_counts  = Counter(adata.obs['cell_line'])

    for item in exclude_list:
        cell_counts.pop(item)

    temp_cell_counts = cell_counts.copy()

    for item in ignore_min_list:
        temp_cell_counts.pop(item)

    #finds the smallest cell line population to equalize to
    min_cell_line_count_idx = np.argmin(list(temp_cell_counts.values()))

    #gets the smallest population
    min_cell_line_num = list(temp_cell_counts.values())[min_cell_line_count_idx]

    cell_line_obs = adata.obs['cell_line'].values
    unique_cell_lines = np.unique(cell_line_obs)

    if equalize_term == 'cell_lines_and_conditions' or equalize_term == 'CLC':

        condition_obs = adata.obs['condition'].values
        unique_conditions = np.unique(condition_obs)

        cells_per_condition = int(min_cell_line_num/len(unique_conditions))

        all_cc_condition_array = np.array([cell_line_obs, condition_obs],dtype=str).T

        new_cell_idxs = []
        for cell_line in unique_cell_lines:

            #ensures the PANC1 cells are not included in the equalization
            if cell_line in exclude_list:
                continue    

            cell_line_idxs = np.argwhere(all_cc_condition_array[:,0] == cell_line)[:,0]
            cell_line_condition_array = all_cc_condition_array[cell_line_idxs,:]

            num_conditions = [len(cell_line_condition_array[cell_line_condition_array[:,1] == condition]) for condition in unique_conditions]
            sorted_num_conditions_idxs = np.argsort(num_conditions)
            sorted_conditions = unique_conditions[sorted_num_conditions_idxs]
            sorted_num_conditions = np.array(num_conditions)[sorted_num_conditions_idxs]

            for idx, (condition, num_condition) in enumerate(zip(sorted_conditions, sorted_num_conditions), start=1):

                cond_cell_idxs = np.argwhere((all_cc_condition_array[:,1] == condition) & (all_cc_condition_array[:,0] == cell_line) )[:,0]

                if num_condition < cells_per_condition:
                    difference = cells_per_condition - num_condition
                    number_cells_to_grab = num_condition

                    if ignore_min_list == []:
                        cells_per_condition += floor(difference/(len(unique_conditions)-idx))
                else:
                    number_cells_to_grab = cells_per_condition
                
                np.random.seed(0)   
                selected_cell_idxs = list(np.random.choice(cond_cell_idxs, size = number_cells_to_grab, replace = False))
                new_cell_idxs.extend(selected_cell_idxs)

    false_array = np.repeat(False, adata.shape[0])
    false_array[new_cell_idxs] = True
    adata.obs[f'equalize_{equalize_term}'] = false_array

    adata = adata[adata.obs[f'equalize_{equalize_term}'],:].copy()

    return adata
