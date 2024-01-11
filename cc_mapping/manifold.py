import anndata as ad
import scipy.stats as st
import matplotlib as mpl
import os
import numpy as np
np.seterr(all="ignore")
import anndata as ad
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import phate

from typing import Union

import sys
sys.path.append(os.getcwd())

from GLOBAL_VARIABLES.GLOBAL_VARIABLES import cc_mapping_package_dir
sys.path.append(cc_mapping_package_dir)

from cc_mapping.plot import general_plotting_function, get_legend, combine_Lof_plots

def run_phate(adata: ad.AnnData,
              feature_set:str,
              layer: str,
              phate_param_dict: dict = {},
              obsm_save_key: str = 'X_phate',
              hyperparam: bool = False):
    
    feature_set_bool = adata.var[feature_set].values
    data = adata.layers[layer][:,feature_set_bool].copy()

    phate_operator = phate.PHATE(**phate_param_dict)
    phate_coords = phate_operator.fit_transform(data) # Reduce the dimensions of the data using the PHATE algorithm

    if hyperparam:
        return phate_coords

    adata.obsm[obsm_save_key] = phate_coords

    return adata


def plot_phate_coords( colors: Union[np.ndarray,list],
                      phate_coords: np.ndarray= None,
                      kwargs: dict = {},
                      adata: ad.AnnData = None,
                      axe: mpl.axes = None,
                      hyperparam: bool = False,
                      obsm_embedding: str = 'X_phate'
                      ):

    if not hyperparam:
        fig, axe = plt.subplots(1,1, figsize=(10,10))
        phate_coords = adata.obsm[obsm_embedding]

    axe.scatter(phate_coords[:,0], phate_coords[:,1],
        c=colors,
        **kwargs)

    axe.axis('off')
    axe.set_yticks([])
    axe.set_xticks([])

    return axe

def phate_hyperparameter_search_plotting_function(axe, idx_dict, plotting_dict):

    col_idx = idx_dict['col_idx']-1
    row_idx = idx_dict['row_idx']-1

    adata = plotting_dict['adata']
    feature_set = plotting_dict['feature_set']
    color_name = plotting_dict['color_name']
    layer = plotting_dict['layer']
    unit_size = plotting_dict['unit_size']
    kwargs = plotting_dict['kwargs']

    hyperparam_dict = plotting_dict['hyperparam_dict'].copy()
    hyperparam_info_dict = plotting_dict['hyperparam_info_dict'].copy()

    row_param_name = hyperparam_info_dict['row_label']
    col_param_name = hyperparam_info_dict['col_label']

    row_param_list = hyperparam_dict[row_param_name]
    col_param_list = hyperparam_dict[col_param_name]

    hyperparam_dict[row_param_name] = row_param_list[row_idx]
    hyperparam_dict[col_param_name] = col_param_list[col_idx]

    phate_coords = run_phate(adata, feature_set, layer,hyperparam_dict, hyperparam=True)

    colors = adata.obs[color_name].values
    
    axe = plot_phate_coords(phate_coords= phate_coords, colors=colors, kwargs=kwargs, axe=axe, hyperparam=True)
    
    return axe

def perform_phate_hyperparameter_search(adata: ad.AnnData,
                                  feature_set: str,
                                  layer: str,
                                  hyperparam_dict: dict,
                                  hyperparam_info_dict: dict,
                                  color_name: list = None,
                                  save_path: str = None,
                                  unit_size: int = 10,
                                  kwargs: dict = {}):

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            raise ValueError(f'{save_dir} does not exist')

    plotting_function = phate_hyperparameter_search_plotting_function
                                  
    constant_param_name = hyperparam_info_dict['constant_label']

    backend = mpl.get_backend()
    mpl.use('agg')

    final_figure_dims = hyperparam_info_dict['final_figure_dims']
    number_param_plots = len(hyperparam_dict[constant_param_name])
    final_num_rows = final_figure_dims[0]
    final_num_cols = final_figure_dims[1]

    total_num_plots = final_num_rows * final_num_cols

    element_list = []   
    figure_list = []
    for idx in tqdm(range(total_num_plots), total=number_param_plots, desc = 'Generating hyperparameter search plots'):

        plotting_dict = {'adata':adata,
                            'feature_set':feature_set,
                            'color_name':color_name,
                            'layer':layer,
                            'unit_size':unit_size,
                            'kwargs':kwargs,
                            }

        # -1 is needed for correct indexing 
        if idx <= number_param_plots-1:
            temp_hyperparam_dict = hyperparam_dict.copy()
            temp_hyperparam_info_dict = hyperparam_info_dict.copy()
            temp_hyperparam_dict[constant_param_name] = hyperparam_dict[constant_param_name][idx]
            temp_hyperparam_info_dict['param_dict'] = temp_hyperparam_dict

            plotting_dict['hyperparam_dict'] = temp_hyperparam_dict
            plotting_dict['hyperparam_info_dict'] = temp_hyperparam_info_dict

            fig = general_plotting_function(plotting_function, temp_hyperparam_info_dict, plotting_dict, hyperparam_search = True, unit_size=unit_size)
            
        else:
            #generate a blank plot to fill in the empty space
            fig = general_plotting_function(plotting_function , temp_hyperparam_info_dict, plotting_dict , blank = True, hyperparam_search = True)
    
        figure_list.append(fig)

    combined_fig = combine_Lof_plots(figure_list, final_figure_dims)

    patches,colors = get_legend(adata, color_name)

    combined_fig.legend(handles=patches, fontsize = 2*unit_size)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    mpl.use(backend)

    return combined_fig            
