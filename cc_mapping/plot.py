import matplotlib as mpl
import pandas as pd
import matplotlib.patches as mpatches
import os 
import matplotlib._pylab_helpers
import anndata as ad
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Union,List
import itertools
import anndata as ad
from math import floor, ceil

import sys
sys.path.append(os.getcwd())

from GLOBAL_VARIABLES.GLOBAL_VARIABLES import cc_mapping_package_dir
sys.path.append(cc_mapping_package_dir)

from cc_mapping.utils import get_str_idx

def plot_row_partitions(adata: ad.AnnData,
                        obs_search_term: str,
                        colors: Union[list, np.ndarray] = None,
                        column_labels: Union[list, np.ndarray] = None,
                        kwargs: dict = {},
                        plot_all: bool = True,
                        unit_size: int  = 20,
                        save_path: str = None):
    """
    Plot row partitions of the given AnnData object.

    Parameters:
    adata (ad.AnnData): The AnnData object containing the data.
    obs_search_term (str): The search term for selecting the observations.
    color_info_dict: The dictionary containing color information.
    kwargs (dict, optional): Additional keyword arguments for the plotting function. Defaults to {}.
    unit_size (int, optional): The size of each unit in the plot. Defaults to 20.
    save_path (str, optional): The path to save the plot. Defaults to None.

    Raises:
    ValueError: If the save directory does not exist.

    Returns:
    None
    """
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            raise ValueError(f'{save_dir} does not exist')

    plotting_function = row_partition_plotting_function
    
    plotting_dict = {'adata': adata,
                    'Lof_colors': colors,
                    'obs_search_term': obs_search_term,
                     'kwargs': {}
                     }
    if kwargs != {}:
        plotting_dict['kwargs'] = kwargs

    if column_labels is not None:
        plotting_dict['column_labels'] = column_labels

    fig = general_plotting_function(plotting_function, {} , plotting_dict, unit_size=unit_size, plot_all=plot_all)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

def row_partition_plotting_function(ax, idx_dict, plotting_dict):
    """
    Plotting function for row partition.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot.
    - idx_dict (dict): A dictionary containing row and column indices.
    - plotting_dict (dict): A dictionary containing plotting parameters.

    Returns:
    - ax (matplotlib.axes.Axes): The modified axes object.
    """
    Lof_colors = plotting_dict['Lof_colors']
    column_labels = plotting_dict['column_labels']

    kwargs = plotting_dict['kwargs'].copy()

    adata = plotting_dict['adata'].copy()
    phate_df = adata.obsm['X_phate']

    obs_search_term = plotting_dict['obs_search_term']

    row_idx = idx_dict['row_idx']
    col_idx = idx_dict['col_idx']

    color_name = Lof_colors[row_idx-1]

    ax.scatter(phate_df[:,0], phate_df[:,1],c='lightgrey', **kwargs)

    colors = adata.obs_vector(color_name)

    if colors.dtype != 'object' and not isinstance(colors, pd.Categorical):
        vmin = np.percentile(colors, 1)
        vmax = np.percentile(colors, 99)
        kwargs.update( { 'vmin' : vmin,
                         'vmax' : vmax,
                         'cmap' : 'rainbow',})

    plotting_df = phate_df

    Lof_label_idxs = [get_str_idx(label, adata.obs[obs_search_term])[0] for label in column_labels if label != 'ALL']

    current_plotting_label = column_labels[col_idx-1]

    if current_plotting_label == 'ALL':
        condition_df = plotting_df 
    else:
        label_idxs = Lof_label_idxs[col_idx-1]
        condition_df =plotting_df[label_idxs,:]
        colors = colors[label_idxs]

    ax.scatter(condition_df[:,0], condition_df[:,1], c=colors, **kwargs)
        
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis('off')
    ax.axis('tight')

    return ax

def general_plotting_function(plotting_function,
                            param_info_dict = None,
                            plotting_dict = None,
                            hyperparam_search = False,
                            plot_all = False,
                            blank = False,
                            fontsize = 35,
                            unit_size=10,
                            param_plot_proportion = 0.20):
    """
    A general plotting function that creates a grid of subplots for visualization.

    Parameters:
    - plotting_function: The function used to plot on each subplot.
    - param_info_dict: A dictionary containing information about the parameters for hyperparameter search.
    - plotting_dict: A dictionary containing information for plotting.
    - hyperparam_search: A boolean indicating whether hyperparameter search is enabled.
    - blank: A boolean indicating whether to return a blank figure.
    - fontsize: The fontsize for the annotations.
    - unit_size: The size of each subplot in units.
    - param_plot_proportion: The proportion of the plot dedicated to parameter labels.

    Returns:
    - fig: The created figure object.
    """

    if  hyperparam_search == True:
        param_dict = param_info_dict['param_dict']

        row_param_name = param_info_dict['row_label']
        col_param_name = param_info_dict['col_label']
        constant_param_name = param_info_dict['constant_label']

        row_param_list = param_dict[row_param_name]
        col_param_list = param_dict[col_param_name]

        num_rows = len(row_param_list)
        num_cols = len(col_param_list)

    else:
        adata = plotting_dict['adata'].copy()
        search_obs_term = plotting_dict['obs_search_term']
        color_names = plotting_dict['Lof_colors']

        row_labels = color_names

        if not plotting_dict.get('column_labels'):
            col_labels = np.unique(adata.obs[search_obs_term])

            if plot_all:
                col_labels = np.append(col_labels, 'ALL')

            plotting_dict['column_labels'] = col_labels 

        else:
            col_labels = plotting_dict['column_labels']


        num_cols = len(col_labels)
        num_rows = len(row_labels)

    row_cmap = plt.cm.get_cmap('tab20')
    col_cmap = plt.cm.get_cmap('Dark2')

    width_ratios = [param_plot_proportion] + np.repeat(1,num_cols).tolist()
    height_ratios = [param_plot_proportion] + np.repeat(1,num_rows).tolist()

    anno_opts = dict(xycoords='axes fraction',
                    va='center', ha='center',fontsize = 5*unit_size)

    col_param_limits = np.linspace(0,1,num_cols+1)
    row_param_limits = np.linspace(0,1,num_rows+1)

    fig = plt.figure(figsize=(unit_size*num_cols,unit_size*num_rows),constrained_layout=True)

    if blank:
        return fig

    gs = fig.add_gridspec(num_rows+1, num_cols+1, width_ratios=width_ratios, height_ratios=height_ratios)

    plot_idx = 0
    first = True
    for row_idx, col_idx in itertools.product(range(num_rows+1), range(num_cols+1)):
            #this creates the labels on the first row and first column 
            if row_idx == 0 or col_idx == 0:
                if first:
                    #this makes one long plot that spans the first row
                    init_row = fig.add_subplot(gs[0, 1:])
                    #this splits the first row into multiple labels based on the number of columns and annotates each one
                    for col_num in range(num_cols):

                        left_limit = col_param_limits[col_num]
                        right_limit = col_param_limits[col_num+1]

                        annotate_xy = (((left_limit + right_limit)/2) , 0.5)
                        anno_opts['xy'] = annotate_xy

                        init_row.axvspan(left_limit, right_limit, facecolor=row_cmap(col_num), alpha=0.5)

                        if hyperparam_search == True:
                            init_row.annotate(f'{col_param_name} = {col_param_list[col_num]}', **anno_opts)
                        else:
                            init_row.annotate(f'{col_labels[col_num]}', **anno_opts)

                        init_row.set_yticks([])
                        init_row.set_xticks([])
                        init_row.set_xlim(0,1)

                    #this makes one long plot that spans the first column
                    init_col = fig.add_subplot(gs[1:,0])

                    #this splits the first column into multiple labels based on the number of rows and annotates each one
                    for row_num in range(num_rows):
                        upper_limit = row_param_limits[row_num]
                        lower_limit = row_param_limits[row_num+1]

                        annotate_xy = (0.5,((lower_limit + upper_limit)/2) )
                        anno_opts['xy'] = annotate_xy

                        init_col.axhspan(lower_limit, upper_limit, facecolor=col_cmap(row_num), alpha=0.5)

                        #Indexing is -(row_num+1) to make the plots go from top to bottom
                        if hyperparam_search == True:
                            init_col.annotate(f'{row_param_name} = {row_param_list[-(row_num+1)]}',rotation = 90, **anno_opts)
                        else:
                            init_col.annotate(f'{row_labels[-(row_num+1)]}',rotation = 90, **anno_opts)

                        init_col.set_yticks([])
                        init_col.set_xticks([])
                        init_col.set_ylim(0,1)
                        
                    #add a small box in the upper right corner to indicate what the constant parameter is for the hyperparameter search
                    if hyperparam_search == True:

                        constant_var_name = param_info_dict['constant_label']
                        constant_param = fig.add_subplot(gs[0,0])
                        anno_opts['xy'] = (0.5,0.5)
                        anno_opts['fontsize'] = 25
                        constant_param.annotate(f'{constant_var_name}={param_dict[constant_param_name]}', **anno_opts)
                        constant_param.set_xticks([])
                        constant_param.set_yticks([])

                    first == False
                    continue

            ax = fig.add_subplot(gs[row_idx,col_idx])

            idx_dict = {'row_idx':row_idx,
                        'col_idx':col_idx,
                        'plot_idx':plot_idx}
                        
            ax = plotting_function(ax, idx_dict, plotting_dict)
            plot_idx += 1 
    return fig

def get_legend(adata: ad.AnnData,
               color_name: str):
    """ get patches from adata.obs[color_name] to be used for creating a legend and returns the list of colors as well

    Args:
        adata (ad.AnnData): AnnData object
        color (str): name of the anndata obs column to use for coloring (i.e.'cell_line_colors')
            The labels for the legend will gotten by removing the last underscore and the last word from the color name (i.e. 'cell_line')
            This will be used to find the obs columns that contain the labels for the legend

    Returns:
        List[mpatches.Patch], List :patches that will be used to create the legend using matplotlib.pyplot.legend() 
                                    and the list of colors corresponding to the color name obs column 
    """
    colors = adata.obs_vector(color_name)

    label_name = color_name.removesuffix('_colors')
    labels = adata.obs[label_name].values

    col_lab_array = np.array([colors, labels],dtype=str).T
    uni_col_lab_matches = np.unique(col_lab_array, axis=0)

    patch_list = []
    for color, label in uni_col_lab_matches:
        patch = mpatches.Patch(color=color, label=label)
        patch_list.append(patch)

    return patch_list, colors

def combine_Lof_plots(Lof_plots: List[mpl.figure.Figure] = None,
                      fig_dims: tuple = None,
                      default_padding: tuple = (0,0),
                      default_padding_color: tuple = 255,
                      save_path: str = None,
                      title_kwargs: dict = None,
                      title: str = None,
                      inline: bool = False):

    """
    Combines a list of matplotlib figures into a single figure with specified dimensions.

    Parameters:
    - Lof_plots (List[mpl.figure.Figure]): List of matplotlib figures to be combined. If `inline` is True, this parameter is ignored.
    - fig_dims (tuple): Dimensions of the final combined figure in terms of number of rows and columns.
    - default_padding (tuple): Padding to be applied to each figure in terms of number of rows and columns.
    - default_padding_color (tuple): Color value (RGB) to be used for the default padding.
    - save_path (str): Path to save the combined figure. If not provided, the figure will be displayed.
    - title_kwargs (dict): Keyword arguments for customizing the title of the combined figure.
    - title (str): Title of the combined figure.
    - inline (bool): If True, the function will automatically retrieve all open figures and combine them. The `Lof_plots` parameter will be ignored.

    Returns:
     - None
    """

    if inline:
        Lof_plots=[manager.canvas.figure
                for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

    #converts the figures into numpy arrays
    for fig_idx, fig in enumerate(Lof_plots):
        canvas = fig.canvas
        canvas.draw()

        element = np.array(canvas.buffer_rgba())

        Lof_plots[fig_idx] = element

    if inline:
        for fig in plt.get_fignums():
            plt.close(fig)

    max_figure_dims = np.max([(fig.shape[0],fig.shape[1]) for fig in Lof_plots],axis=0)

    for fig_idx, fig in enumerate(Lof_plots):

        row_diff = max_figure_dims[0] - fig.shape[0]
        col_diff = max_figure_dims[1] - fig.shape[1]

        if row_diff > 0:
            fig = np.pad(fig, ((floor(row_diff/2),ceil(row_diff/2)),(0,0),(0,0)), 'constant', constant_values=255)
        if col_diff > 0:
            fig = np.pad(fig, ((0,0),(floor(col_diff/2),ceil(col_diff/2)), (0,0)), 'constant', constant_values=255)

        default_row_padding = default_padding[0]
        default_col_padding = default_padding[1]

        fig = np.pad(fig, ((default_row_padding,default_row_padding),(default_col_padding,default_col_padding), (0,0)), 'constant', constant_values=default_padding_color)

        Lof_plots[fig_idx] = fig

    final_num_rows = fig_dims[0]
    final_num_cols = fig_dims[1]

    if len(Lof_plots) < final_num_rows*final_num_cols:
        fig_num_diff = final_num_rows*final_num_cols - len(Lof_plots)
        for _ in range(fig_num_diff):
            Lof_plots.append(np.ones_like(fig)*default_padding_color)

    #shapes the figures generated above into the final figure dimensions
    counter = 0
    fig_rows =[]
    for plot in range(final_num_rows):
        fig_row = Lof_plots[counter:counter+final_num_cols]
        fig_row = np.hstack(fig_row)
        fig_rows.append(fig_row)
        counter+=final_num_cols

    fig_rows = tuple(fig_rows)

    plot_array = np.vstack(fig_rows)

    f_unit_size = 15
    #plots the final figure
    fig,ax = plt.subplots(figsize=(f_unit_size*final_num_cols, f_unit_size*final_num_rows), constrained_layout=True, dpi =600)

    ax.matshow(plot_array)

    if title:
        ax.set_title(title, **title_kwargs)

    ax.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    elif inline:
        plt.show()
        plt.close()
    else:
        return fig