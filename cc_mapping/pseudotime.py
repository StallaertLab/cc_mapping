import palantir
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import contextlib
import numpy as np 
import anndata as ad
from collections import Counter

import sys
sys.path.append(os.getcwd())

from GLOBAL_VARIABLES.GLOBAL_VARIABLES import cc_mapping_package_dir
sys.path.append(cc_mapping_package_dir)

from cc_mapping.plot import general_plotting_function



def run_palantir_pseudotime(adata: ad.AnnData,
                            root_cell: str,
                            data_key: str,
                            n_components: int,
                            num_waypoints:int,
                            knn: int,
                            seed:int = 0,
                            plot:bool = True,
                            kwargs: dict = {}):

    try:
        with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    palantir.utils.run_diffusion_maps(adata, n_components=n_components, pca_key=data_key ,seed=seed)
                    palantir.utils.determine_multiscale_space(adata)

                    palantir.core.run_palantir(adata, root_cell, num_waypoints=num_waypoints, knn=knn,seed=seed)
    except Exception as e:
        print(e)
        exit()

    if plot:
        fig = palantir.plot.plot_palantir_results(adata, embedding_basis = 'X_phate',**kwargs)
        return fig, adata
    else:
        return adata

def palantir_pseudotime_hyperparam_plotting_function(axe, idx_dict, plotting_dict,unit_size=10,s=10):

    col_idx = idx_dict['col_idx']-1
    row_idx = idx_dict['row_idx']-1

    hyperparam_dict = plotting_dict['hyperparam_dict'].copy()
    random_seed = hyperparam_dict['random_seed']
    hyperparam_dict.pop('random_seed')

    hyperparam_info_dict = plotting_dict['hyperparam_info_dict'].copy()
    kwargs = plotting_dict['kwargs'].copy()

    row_param_name = hyperparam_info_dict['row_label']
    col_param_name = hyperparam_info_dict['col_label']

    row_param_list = hyperparam_dict[row_param_name]
    col_param_list = hyperparam_dict[col_param_name]

    if row_param_name == 'num_waypoints':
        num_waypoints = row_param_list[row_idx]
        knn = col_param_list[col_idx]
    elif row_param_name == 'knns':
        knn = row_param_list[row_idx]
        num_waypoints = col_param_list[col_idx]

    n_components = hyperparam_dict['n_components']

    adata = plotting_dict['adata']
    root_cell = plotting_dict['root_cell']
    data_key = plotting_dict['data_key']

    fig, _ = run_palantir_pseudotime(adata, root_cell, data_key, n_components, num_waypoints, knn, seed=random_seed, kwargs=kwargs)
    
    canvas = fig.canvas
    canvas.draw()

    #this is a numpy array of the plot
    element = np.array(canvas.buffer_rgba())
    axe.imshow(element)
    def get_plot_limits(element, type):
        if type == 'right':
            element = np.flip(element, axis=1)

        truth_bool_array = np.repeat(True, element.shape[1])

        white_cols = np.where(np.all(np.isclose(element, 255), axis=0))[0]
        cols = list(Counter(white_cols).keys())
        counts = list(Counter(white_cols).values())
        white_cols = np.where(np.array(counts) == 4)[0]
        white_col_idxs = [cols[idx] for idx in white_cols]
        truth_bool_array[white_col_idxs] = False

        return truth_bool_array

    left_truth_bool_array = get_plot_limits(element, 'left')
    right_truth_bool_array = get_plot_limits(element, 'right')

    fig = plt.figure(figsize=(10,10))
    xll = np.argwhere(left_truth_bool_array==True)[0]
    xul = element.shape[1]-np.argwhere(right_truth_bool_array==True)[0]
    axe.set_xlim(xll,xul)
    axe.axis('off')
    return axe,element


def perform_palantir_hyperparameter_search(adata: ad.AnnData,
                                  data_key: str,
                                  root_cell: str,
                                  hyperparam_dict: dict,
                                  hyperparam_info_dict: dict,
                                  save_path: str = None,
                                  unit_size: int = 10,
                                  kwargs: dict = {}):

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            raise ValueError(f'{save_dir} does not exist')

    plotting_function = palantir_pseudotime_hyperparam_plotting_function
                                  
    constant_param_name = hyperparam_info_dict['constant_label']

    backend = mpl.get_backend()
    mpl.use('agg')

    final_figure_dims = hyperparam_info_dict['final_figure_dims']
    number_param_plots = len(hyperparam_dict[constant_param_name])
    final_num_rows = final_figure_dims[0]
    final_num_cols = final_figure_dims[1]

    total_num_plots = final_num_rows * final_num_cols

    element_list = []   
    for idx in tqdm(range(total_num_plots), total=number_param_plots, desc = 'Generating hyperparameter search plots'):

        temp_hyperparam_dict = hyperparam_dict.copy()
        temp_hyperparam_info_dict = hyperparam_info_dict.copy()
        temp_hyperparam_dict[constant_param_name] = hyperparam_dict[constant_param_name][idx]
        temp_hyperparam_info_dict['param_dict'] = temp_hyperparam_dict

        plotting_dict = {'adata':adata.copy(),
                            'data_key':data_key,
                            'unit_size':unit_size,
                            'root_cell':root_cell,
                            'hyperparam_dict': temp_hyperparam_dict,
                            'hyperparam_info_dict':temp_hyperparam_info_dict,
                            'kwargs':kwargs,
                            }

        # -1 is needed for correct indexing 
        if idx <= number_param_plots-1:
            fig = general_plotting_function(plotting_function, temp_hyperparam_info_dict, plotting_dict, hyperparam_search = True, unit_size=unit_size)
            
        else:
            #generate a blank plot to fill in the empty space
            fig = general_plotting_function(plotting_function, temp_hyperparam_info_dict, plotting_dict , blank = True, hyperparam_search = True)
    
        canvas = fig.canvas
        canvas.draw()

        #this is a numpy array of the plot
        element = np.array(canvas.buffer_rgba())

        element_list.append(element) 

    #shapes the figures generated above into the final figure dimensions
    counter = 0
    fig_rows =[]
    for plot in range(final_num_rows):
        fig_row = element_list[counter:counter+final_num_cols]
        fig_row = np.hstack(fig_row)
        fig_rows.append(fig_row)
        counter+=final_num_cols

    fig_rows = tuple(fig_rows)

    plot_array = np.vstack(fig_rows)

    mpl.use(backend)

    #plots the final figure
    fig,ax = plt.subplots(figsize=(unit_size*final_num_cols, unit_size*final_num_rows), constrained_layout=True)

    ax.set_axis_off()

    ax.matshow(plot_array)
    #plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax            