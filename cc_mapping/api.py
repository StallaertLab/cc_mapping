import requests
from urllib.parse import urlencode
from typing import Union
import numpy as np 
import pandas as pd

from .manifold import plot_phate_coords
from .utils import get_str_idx
from collections import Counter
import matplotlib.pyplot as plt
import palantir

def RunG0MO3(G0MOName: str,
                DataFile: str,
             SampleName: str,
             GMMKwargs: dict,
             G0MOFeatures: list,
             G0MO_Instructions: dict,
             local_host: str = 'http://127.0.0.1:8080/') -> None:


     RulesList = ['DataFile', 'G0MOName', 'SampleName']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + 'G0MO/RunG0MO?' + url_params
     data = {
               'GMMKwargs': GMMKwargs,
               'G0M0Features': G0MOFeatures,
               'G0MO_Instructions': G0MO_Instructions
          }

     try:
          response = requests.post(url, json=data)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')

def get_G0MO3_results(G0MOName: str,
                      SampleName: str,
                      local_host: str = 'http://127.0.0.1:8000/') -> dict:

     RulesList = ['G0MOName', 'SampleName']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     RulesList = ['G0MOName', 'SampleName']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + 'G0MO/ReturnG0MOParameters?' + url_params
     response = requests.get(url)
     return response.json()

def RunRandomForestFS( RandomForestFSName: str,
                        DataFile: str,
                        G0MOName: str,
                        SampleName: str,
                        RandomForestFeatureSelectionArgs: dict[str, Union[int, str, float, bool]],
                        RandomForestClassifierArgs: dict[str, Union[int, str, float, bool]],
                        TrainTestSplitArgs: dict[str, Union[int, str, float, bool]],
                        FeaturesToRemove: list[str],
                        local_host: str = 'http://127.0.0.1:8080/') -> None:

    ROUTE_ROOT = 'RandomForestFS'
    ROUTE_NAME = f'Run{ROUTE_ROOT}'

    RulesList = [
                'DataFile',
                 'RandomForestFSName',
                 'G0MOName',
                 'SampleName'
                 ]

    params_dict = {}
    for key in RulesList:
        params_dict[key] = locals()[key]

    url_params = urlencode(params_dict)

    url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params

    data = {'RandomForestClassifierArgs': RandomForestClassifierArgs,
            'TrainTestSplitArgs': TrainTestSplitArgs,
            'RandomForestFeatureSelectionArgs': RandomForestFeatureSelectionArgs,
            'FeaturesToRemove': FeaturesToRemove
            }

    response = requests.post(url, json=data)

def GetRandomForestFSResults( RandomForestFSName: str,
                              SampleName: str,
                              local_host: str = 'http://127.0.0.1:8000/') -> dict:
    
    ROUTE_ROOT = 'RandomForestFS'
    ROUTE_NAME = f'Return{ROUTE_ROOT}Parameters'
    RulesList = ['RandomForestFSName', 'SampleName']

    params_dict = {}
    for key in RulesList:
        params_dict[key] = locals()[key]

    url_params = urlencode(params_dict)

    url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params
    response = requests.get(url)
    return response.json()

def GetPhateArgs(PhateArgs: dict[str, Union[int, str, float, bool]],
                local_host: str = 'http://127.0.0.1:8000/') -> dict:

    ROUTE_ROOT = 'Phate'
    ROUTE_NAME = f'Get{ROUTE_ROOT}ArgsID'

    url = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?'
    data = {'InputPhateArgs': PhateArgs}
    response = requests.get(url, json=data)
    return response.json()

def RunPhate(SampleName: str,
                DataFile: str,
             PhateArgs: dict,
             RandomForestFSName: str,
             local_host: str = 'http://127.0.0.1:8080/') -> None:

     ROUTE_ROOT = 'Phate'
     ROUTE_NAME = f'Run{ROUTE_ROOT}'


     RulesList = ['DataFile', 'SampleName', 'RandomForestFSName']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params
     data = PhateArgs

     try:
          response = requests.post(url, json=data)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')

def ReturnPhateResults( SampleName: str,
                         RandomForestFSName: str,
                         PhateArgs: dict,
                         local_host: str = 'http://127.0.0.1:8000/') -> None:

     ROUTE_ROOT = 'Phate'
     ROUTE_NAME = f'Return{ROUTE_ROOT}Parameters'

     RulesList = ['SampleName', 'RandomForestFSName']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params
     data = PhateArgs

     try:
          response = requests.get(url, json=data)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')
     
     return response.json()

def AddOptimalPhateRun( SampleName: str,
                         RandomForestFSName: str,
                         PhateArgs: dict,
                         local_host: str = 'http://127.0.0.1:8000/') -> None:

     ROUTE_ROOT = 'Phate'
     ROUTE_NAME = f'AddOptimal{ROUTE_ROOT}Run'

     RulesList = ['SampleName', 'RandomForestFSName']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params
     data = PhateArgs

     try:
          response = requests.post(url, json=data)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')
     
     return response.json()

def ReturnOptimalPhateRun( SampleName: str,
                         RandomForestFSName: str,
                         local_host: str = 'http://127.0.0.1:8000/') -> None:

     ROUTE_ROOT = 'Phate'
     ROUTE_NAME = f'ReturnOptimal{ROUTE_ROOT}Run'
     
     RulesList = ['SampleName', 'RandomForestFSName']
     
     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params

     try:
          response = requests.get(url)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')
     
     return response.json()
     
def RunPalantir(SampleName: str,
                DataFile: str,
                RandomForestFSName: str,
                DiffusionMapArgs: dict,
                MultiscaleSpaceArgs: dict,
                PalantirRunArgs: dict,
                RootCellIdx: int,
                local_host: str = 'http://127.0.0.1:8080/') -> None:
    
     ROUTE_ROOT = 'Palantir'
     ROUTE_NAME = f'Run{ROUTE_ROOT}'


     RulesList = ['DataFile', 'SampleName', 'RandomForestFSName', 'RootCellIdx']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params

     data = {'DiffusionMapArgs': DiffusionMapArgs,
                'MultiscaleSpaceArgs': MultiscaleSpaceArgs,
                'PalantirRunArgs': PalantirRunArgs}
     try:
          response = requests.post(url, json=data)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')

def ReturnPalantirResults( SampleName: str,
                         RandomForestFSName: str,
                         DiffusionMapArgs: dict,
                         MultiscaleSpaceArgs: dict,
                         PalantirRunArgs: dict,
                         RootCellIdx: int,
                         local_host: str = 'http://127.0.0.1:8000/') -> None:

     ROUTE_ROOT = 'Palantir'
     ROUTE_NAME = f'Return{ROUTE_ROOT}Parameters'

     RulesList = ['SampleName', 'RandomForestFSName', 'RootCellIdx']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params
     data = {'DiffusionMapArgs': DiffusionMapArgs,
                'MultiscaleSpaceArgs': MultiscaleSpaceArgs,
                'PalantirRunArgs': PalantirRunArgs}

     try:
          response = requests.get(url, json=data)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')
     
     return response.json()

def AddOptimalPalantirRun( SampleName: str,
                         RandomForestFSName: str,
                         DiffusionMapArgs: dict,
                         MultiscaleSpaceArgs: dict,
                         PalantirRunArgs: dict,
                         RootCellIdx: int,
                         local_host: str = 'http://127.0.0.1:8000/') -> None:

     ROUTE_ROOT = 'Palantir'
     ROUTE_NAME = f'AddOptimal{ROUTE_ROOT}Run'

     RulesList = ['SampleName', 'RandomForestFSName', 'RootCellIdx']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params
     data = {'DiffusionMapArgs': DiffusionMapArgs,
                'MultiscaleSpaceArgs': MultiscaleSpaceArgs,
                'PalantirRunArgs': PalantirRunArgs}

     try:
          response = requests.post(url, json=data)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')
     
     return response.json()

def PalantirHelp( ROUTENAME: str,
                         local_host: str = 'http://127.0.0.1:8000/') -> None:

     ROUTE_ROOT = 'Palantir'
     ROUTE_NAME = f'{ROUTE_ROOT}Help'

     RulesList = ['ROUTENAME']

     params_dict = {}
     for key in RulesList:
          params_dict[key] = locals()[key]

     url_params = urlencode(params_dict)

     url  = local_host + f'{ROUTE_ROOT}/{ROUTE_NAME}?' + url_params
     try:
          response = requests.get(url)
     except:
          raise Exception('Could not connect to the server. Please check if the server is running.')
     
     return response.json()

def api_phate_hyperparameter_search_plotting_function(axe, idx_dict, plotting_dict):
    """
    Plot PHATE coordinates for hyperparameter search.

    Args:
        axe (matplotlib.axes.Axes): The axes object to plot on.
        idx_dict (dict): Dictionary containing the row and column indices.
        plotting_dict (dict): Dictionary containing the plotting parameters.

    Returns:
        matplotlib.axes.Axes: The updated axes object.

    """
    col_idx = idx_dict['col_idx']-1
    row_idx = idx_dict['row_idx']-1

    adata = plotting_dict['adata']
    feature_set = plotting_dict['feature_set']
    color_name = plotting_dict['color_name']
    unit_size = plotting_dict['unit_size']
    kwargs = plotting_dict['kwargs']

    sample_name = plotting_dict['sample_name']

    if color_name is None:
        G0MOName = plotting_dict['G0MOName']

    hyperparam_dict = plotting_dict['hyperparam_dict'].copy()
    hyperparam_info_dict = plotting_dict['hyperparam_info_dict'].copy()

    row_param_name = hyperparam_info_dict['row_label']
    col_param_name = hyperparam_info_dict['col_label']

    row_param_list = hyperparam_dict[row_param_name]
    col_param_list = hyperparam_dict[col_param_name]

    hyperparam_dict[row_param_name] = row_param_list[row_idx]
    hyperparam_dict[col_param_name] = col_param_list[col_idx]

    try:    
        phate_coords_file = ReturnPhateResults(SampleName= sample_name,
                        RandomForestFSName= feature_set,
                        PhateArgs= hyperparam_dict)['PhateCoordsSavePath']

        print(f'phate_coords_file: {phate_coords_file}')
    except Exception as e:
        axe = plot_phate_coords(blank=True)
        print(e)
        return axe

    if color_name is None:

        try:
            PhaseLabelPath = get_G0MO3_results(G0MOName= G0MOName, SampleName= sample_name)['PhaseLabelSavePath']
        except:
            raise ValueError('G0MONAme not found in database.') 

        phase_labels = pd.read_csv(PhaseLabelPath, index_col=0).iloc[:, 0]
        phase_colors = phase_labels.map(adata.uns['cc_phase_color_mappings'])

        color_name = 'PhaseLabels'
        adata.obs[color_name] = phase_colors.values

    phate_coords = pd.read_csv(phate_coords_file, index_col=0)

    if np.all(phate_coords['CellID'].values.astype('<U5') != adata.obs['CellID'].values.astype('<U5')):
        raise ValueError('CellID mismatch between adata and phate_coords file.')
    
    phate_coords = phate_coords[['PhateX', 'PhateY']].values

    axe = plot_phate_coords(adata=adata, phate_coords= phate_coords, colors=color_name, kwargs=kwargs, axe=axe, hyperparam=True, unit_size=unit_size)
    
    return axe

def api_palantir_pseudotime_hyperparam_plotting_function(axe, idx_dict, plotting_dict,unit_size=10,s=10):

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

    sample_name = plotting_dict['sample_name']
    random_forest_fs_name = plotting_dict['random_forest_fs_name']
    PhateArgs = plotting_dict['PhateArgs']
    G0MOName = plotting_dict['G0MOName']

    hyperparam_info_dict = plotting_dict['hyperparam_info_dict'].copy()

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

    param_set = {
        'DiffusionMapArgs': {'n_components': n_components, 'seed':random_seed},
        'MultiscaleSpaceArgs': {},
        'PalantirRunArgs': {'num_waypoints': num_waypoints, 'knn': knn, 'seed': random_seed},
    }

    root_cell = plotting_dict['root_cell']

    try:    
        phate_coords_file = ReturnPhateResults(SampleName= sample_name,
                        RandomForestFSName= random_forest_fs_name,
                        PhateArgs= PhateArgs)['PhateCoordsSavePath']
        
        phate_coords = pd.read_csv(phate_coords_file, index_col=0)

    except Exception as e:
        raise ValueError(f'Error in retrieving phate_coords_file: {e}')

    try:
        PhaseLabelPath = get_G0MO3_results(G0MOName= G0MOName, SampleName= sample_name)['PhaseLabelSavePath']
    except:
        raise ValueError('G0MONAme not found in database.') 

    try:    
        print(sample_name, random_forest_fs_name, root_cell, param_set)
        PalantirResults = ReturnPalantirResults(SampleName= sample_name,
                    RandomForestFSName= random_forest_fs_name,
                    RootCellIdx= root_cell,
                    **param_set)
        print(PalantirResults)
        print()

    except Exception as e:
        print(f'Error in retrieving PalantirResults: {e}')
        fig, axe = plt.subplots(1,1, figsize=(unit_size,unit_size))
        axe.axis('off')
        return axe

    PhaseLabels = pd.read_csv(PhaseLabelPath, index_col=0)
    phase_set_list = ['G0', 'G1', 'S', 'G2']

    phase_idxs, _ = get_str_idx(phase_set_list, PhaseLabels.iloc[:,0])

    ps_phate_coords = phate_coords.iloc[phase_idxs,1:]
    ps_phate_coords.index = np.arange(ps_phate_coords.shape[0])

    BranchProbs = pd.read_csv(PalantirResults['BranchProbsSavePath'], index_col=0)
    Pseudotime = pd.read_csv(PalantirResults['PseudotimeSavePath'], index_col=0)
    Entropy = pd.read_csv(PalantirResults['EntropySavePath'], index_col=0)
    Waypoints = pd.read_csv(PalantirResults['WaypointsSavePath'], index_col=0)
    Waypoints.index = Waypoints.squeeze().values

    palantir_results = palantir.presults.PResults(pseudotime = Pseudotime.values, entropy= Entropy.values, branch_probs= BranchProbs, waypoints= Waypoints.index)
    fig = palantir.plot.plot_palantir_results(data=ps_phate_coords, pr_res = palantir_results)
    
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