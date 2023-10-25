import palantir


def plot_palantir_pseudotime(adata, root_cell, data_key, n_components, num_waypoints, knn):

    palantir.utils.run_diffusion_maps(adata, n_components=20, pca_key=data_key ,seed=0)
    palantir.utils.determine_multiscale_space(adata)

    palantir.core.run_palantir(adata, root_cell, num_waypoints=1000, knn=30,seed=0)

    fig = palantir.plot.plot_palantir_results(adata, embedding_basis = 'X_phate')

    return fig, adata


    

def palantir_pseudotime_hyperparam_plotting_function(axe, idx_dict, plotting_dict,unit_size=10,s=10):

    plot_idx = idx_dict['plot_idx']
    optimal_parameters = plotting_dict['optimal_parameters']

    adata = optimal_parameters['adata']
    root_cell = optimal_parameters['root_cell']
    data_key = optimal_parameters['data_key']
    n_components = optimal_parameters['n_components']
    num_waypoints = optimal_parameters['num_waypoints']
    knn = optimal_parameters['knn']

    fig = plot_palantir_pseudotime(adata, data_key, n_components, num_waypoints, knn)

    
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

