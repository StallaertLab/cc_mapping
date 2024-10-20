import palantir
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import contextlib
import numpy as np 
import anndata as ad
from collections import Counter

from .plot import general_plotting_function, combine_Lof_plots


def run_palantir_pseudotime(adata: ad.AnnData,
                            root_cell: str,
                            data_key: str,
                            n_components: int,
                            num_waypoints:int,
                            knn: int,
                            obsm_embedding: str = 'X_phate',
                            seed:int = 0,
                            plot:bool = True,
                            kwargs: dict = {}):
    """
    Runs the Palantir pseudotime analysis on the provided AnnData object.

    Parameters:
        adata (ad.AnnData): The AnnData object containing the data.
        root_cell (str): The name of the root cell for pseudotime analysis.
        data_key (str): The key in `adata.obsm` where the data is stored.
        n_components (int): The number of diffusion components to compute.
        num_waypoints (int): The number of waypoints to compute.
        knn (int): The number of nearest neighbors to use for constructing the kNN graph.
        obsm_embedding (str, optional): The key in `adata.obsm` where the embedding is stored. Defaults to 'X_phate'.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        kwargs (dict, optional): Additional keyword arguments to pass to `palantir.plot.plot_palantir_results`.

    Returns:
        matplotlib.figure.Figure or None: The generated plot figure, or None if an error occurred.
    """
    try:
        with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    
                    palantir.utils.run_diffusion_maps(adata, n_components=n_components, pca_key=data_key ,seed=seed)
                    palantir.utils.determine_multiscale_space(adata)

                    palantir.core.run_palantir(adata, root_cell, num_waypoints=num_waypoints, knn=knn,seed=seed)
    except Exception as e:
        print(f'n_components: {n_components}, num_waypoints: {num_waypoints}, knn: {knn}')
        print(e)

        if plot:
            fig = plt.figure(figsize=(10,10))
            plt.tight_layout()
            return fig
            

    if plot:
        fig = palantir.plot.plot_palantir_results(adata, embedding_basis = obsm_embedding, **kwargs)
        plt.tight_layout()
        return fig

def palantir_pseudotime_hyperparam_plotting_function(axe, idx_dict, plotting_dict,unit_size=10,s=10):

    """
    Plotting function for palantir pseudotime hyperparameters.

    Args:
        axe (matplotlib.axes.Axes): The axes on which to plot the pseudotime hyperparameters.
        idx_dict (dict): A dictionary containing the row and column indices.
        plotting_dict (dict): A dictionary containing the plotting information.
        unit_size (int, optional): The size of the units in the plot. Defaults to 10.
        s (int, optional): The size of the markers in the plot. Defaults to 10.

    Returns:
        matplotlib.axes.Axes: The modified axes object.
        numpy.ndarray: The plot as a numpy array.
    """

    col_idx = idx_dict['col_idx']-1
    row_idx = idx_dict['row_idx']-1

    hyperparam_dict = plotting_dict['hyperparam_dict'].copy()
    random_seed = hyperparam_dict['random_seed']
    hyperparam_dict.pop('random_seed')

    hyperparam_info_dict = plotting_dict['hyperparam_info_dict'].copy()
    kwargs = plotting_dict['kwargs'].copy()

    obsm_embedding = plotting_dict['obsm_embedding']

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

    fig = run_palantir_pseudotime(adata, root_cell, data_key, n_components, num_waypoints, knn, obsm_embedding= obsm_embedding, seed=random_seed, kwargs=kwargs)
    
    canvas = fig.canvas
    canvas.draw()

    #this is a numpy array of the plot
    element = np.array(canvas.buffer_rgba())
    axe.imshow(element)
    
    if not np.all(element == 255):
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
                                  additional_plotting_dict_params: dict,
                                  plotting_function: callable = palantir_pseudotime_hyperparam_plotting_function,
                                  obsm_embedding: str = 'X_phate',
                                  save_path: str = None,
                                  unit_size: int = 10,
                                  kwargs: dict = {}):

    """
    Perform hyperparameter search for Palantir pseudotime analysis.

    Parameters:
    adata (ad.AnnData): Annotated data object.
    data_key (str): Key for accessing the data in `adata`.
    root_cell (str): Name of the root cell for pseudotime analysis.
    hyperparam_dict (dict): Dictionary containing hyperparameters to be searched.
    hyperparam_info_dict (dict): Dictionary containing information about hyperparameters.
    obsm_embedding (str, optional): Key for accessing the embedding in `adata.obsm`. Defaults to 'X_phate'.
    save_path (str, optional): Path to save the final figure. Defaults to None.
    unit_size (int, optional): Size of each subplot in the final figure. Defaults to 10.
    kwargs (dict, optional): Additional keyword arguments for plotting functions. Defaults to {}.

    Returns:
    ax: The matplotlib axis object containing the final figure.
    """
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            raise ValueError(f'{save_dir} does not exist')

    constant_param_name = hyperparam_info_dict['constant_label']

    backend = mpl.get_backend()
    mpl.use('agg')

    final_figure_dims = hyperparam_info_dict['final_figure_dims']
    number_param_plots = len(hyperparam_dict[constant_param_name])
    final_num_rows = final_figure_dims[0]
    final_num_cols = final_figure_dims[1]

    total_num_plots = final_num_rows * final_num_cols

    if total_num_plots < number_param_plots:
        raise ValueError(f'Number of plots ({number_param_plots}) exceeds the number of subplots ({total_num_plots})')

    fig_list = []
    for idx in tqdm(range(total_num_plots), total=number_param_plots, desc = 'Generating hyperparameter search plots'):


        plotting_dict = {'adata':adata.copy(),
                            'data_key':data_key,
                            'unit_size':unit_size,
                            'root_cell':root_cell,
                            'obsm_embedding':obsm_embedding,
                            'kwargs':kwargs,
                            }

        if additional_plotting_dict_params:
            plotting_dict.update(additional_plotting_dict_params)

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
            fig = general_plotting_function(plotting_function, temp_hyperparam_info_dict, plotting_dict , blank = True, hyperparam_search = True)
    
        fig_list.append(fig)

    combined_fig = combine_Lof_plots(fig_list, final_figure_dims, save_path=save_path)

    return combined_fig
