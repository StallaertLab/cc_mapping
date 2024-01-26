
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
np.seterr(all="ignore")
import anndata as ad
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture 

from typing import Union

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import sys
from GLOBAL_VARIABLES.GLOBAL_VARIABLES import cc_mapping_package_dir
sys.path.append(cc_mapping_package_dir)

from cc_mapping import utils

def train_random_forest_model(features,
                              labels,
                              rf_params: dict ,
                              random_state: bool,
                              train_test_split_params: dict,
                              verbose: bool = True,
                              ):

    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                labels,
                                                                                random_state = random_state,
                                                                                **train_test_split_params)
    rf_classifier = RandomForestClassifier(random_state = random_state, **rf_params)

    rf_classifier.fit(train_features, train_labels)

    rf_pred_labels=rf_classifier.predict(test_features)

    accuracy = metrics.accuracy_score(test_labels, rf_pred_labels)

    if verbose:
        print()
        print(f'Classification Report for RF model trained with given feature set')
        print('##################################################################')
        print()
        print(metrics.classification_report(test_labels, rf_pred_labels))

    return rf_classifier, accuracy

def random_forest_feature_selection(adata: ad.AnnData,
                                    training_feature_set: str,
                                    training_labels: str,
                                    feature_set_name: str = None,
                                    method: str = 'RF_min_max',
                                    random_state: int = 42,
                                    threshold: float = 0.01,
                                    stable_counter: int = 10,
                                    plot: bool = True,
                                    train_test_split_params: dict = {'test_size':0.25},
                                    rf_params: dict = {'min_samples_leaf':50,
                                                        'n_estimators':150,
                                                        'bootstrap':True,
                                                        'oob_score':True,
                                                        'n_jobs':-1, },
                                    ) -> ad.AnnData:
                                    
    """ Trains a random forest classifier on the training feature set and labels using one of two methods:
        RF_min_30: Selects the top 30 features based on the random forest feature importance
        RF_min_max: Selects the minimum number of features that maximizes the accuracy of the random forest classifier
                    This is done by iteratively adding features to the feature set until the accuracy of the classifier
                    does not improve for x number of iterations
    

    Args:
        adata (ad.AnnData): _description_
        training_feature_set (str): _description_
        training_labels (str): _description_
        feature_set_name (str, optional): _description_. Defaults to None.
        method (str, optional): _description_. Defaults to 'RF_min_max'.
        stable_counter (int, optional): _description_. Defaults to 10.

    Returns:
        ad.AnnData: returns the adata object with the feature set added to the .var attribute
    """

    feature_set_idxs, _ = utils.get_str_idx(training_feature_set, adata.var_names.values)

    try:
        phase_nan_idx, _ = utils.get_str_idx('nan', adata.obs[training_labels])
    except:
        phase_nan_idx = []

    trunc_cell_data = adata.X[:,feature_set_idxs].copy()

    trunc_cell_data = np.delete(trunc_cell_data, phase_nan_idx, axis=0)
    labels = np.delete(adata.obs[training_labels].values, phase_nan_idx, axis=0)

    trunc_cell_data[trunc_cell_data == np.inf] = np.nan
    trunc_cell_data[trunc_cell_data == -np.inf] = np.nan

    nan_data_idx = np.isnan(trunc_cell_data).any(axis=1)

    trunc_cell_data = trunc_cell_data[~nan_data_idx]
    labels = labels[~nan_data_idx]

    # Convert to numpy array
    features = trunc_cell_data

    rf_classifier, _ = train_random_forest_model(features = features,
                                                 labels = labels,
                                                 rf_params = rf_params,
                                                 train_test_split_params=train_test_split_params,
                                                 random_state=random_state)
        
    #negative to have it sort from highest to lowest
    sorted_idxs = np.argsort(-rf_classifier.feature_importances_)
    sorted_feature_set = np.array(training_feature_set)[sorted_idxs]
    sorted_features = trunc_cell_data[:,sorted_idxs]

    if method == 'RF_min_30':
        optim_RF_feature_set = sorted_feature_set[:30]
    elif method == 'RF_min_max':

        counter = 0
        max_acc_arg = 0
        acc_list = [0]
        for num_feats in tqdm(range(1, sorted_features.shape[1]+1),total= sorted_features.shape[1], desc = f'Training RF model iteratively using most important RF features until {stable_counter} stable iterations'):

            trunc_feature_set = sorted_features[:,:num_feats]

            _ , accuracy = train_random_forest_model(features = trunc_feature_set,
                                                        labels = labels,
                                                        rf_params = rf_params,
                                                        train_test_split_params=train_test_split_params,
                                                        random_state=random_state,
                                                        verbose=False)
            acc_list.append(accuracy)

            if max_acc_arg == np.argmax(acc_list):
                counter += 1
            else:
                acc_difference = np.abs(acc_list[max_acc_arg] - accuracy)
                if acc_difference > threshold:
                    max_acc_arg = np.argmax(acc_list)
                    counter = 0
                    continue
                
                counter += 1

            if counter == stable_counter:
                break


        acc_list = np.array(acc_list)
        optim_feat_num = max_acc_arg + 1
        optim_RF_feature_set = sorted_feature_set[:optim_feat_num]

    optim_feature_set = sorted_features[:,:optim_feat_num]

    _ = train_random_forest_model(features = optim_feature_set,
                                    labels = labels,
                                    rf_params = rf_params,
                                    train_test_split_params=train_test_split_params,
                                    random_state=random_state,)

    print()
    print('##################################################################################')
    print()

    print('Optimal Feature Set sorted by RF feature importance')
    print('###################################################')
    print(optim_RF_feature_set)

    feat_idxs, _ = utils.get_str_idx(optim_RF_feature_set, adata.var_names.values)

    fs_bool = np.repeat(False, adata.shape[1])
    fs_bool[feat_idxs] = True

    if feature_set_name is None:
        feature_set_name = f'{method}_feature_set'

    adata.var[feature_set_name] = fs_bool

    if plot:
        fig = plt.figure(figsize=(10,5))

        x_axis = np.arange( len(acc_list), dtype=int)
        plt.plot(x_axis, acc_list)
        plt.axvline(optim_feat_num, color='r', linestyle='--', label=f'Optimal Feature Set Size: {optim_feat_num}')
        plt.title(f'Stable Counter {stable_counter} - Stable Threshold {threshold*100}%')
        plt.xticks(x_axis)

        percentages = np.arange(0,110,10,dtype=int).tolist()
        rounded_percentages =  [ f'{elem} %' for elem in percentages ]
        plt.grid(visible = True, alpha = 0.5, linestyle = '--', which = 'both')
        plt.yticks(np.arange(0,1.1,0.1), rounded_percentages)
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.xlim(0, len(acc_list)-1)
        plt.ylim(0,1.05)
        plt.legend(loc='lower right')

    return adata

def fit_GMM(x: Union[np.ndarray, list],
            n_components: int = None, 
            kwargs: dict = None):

    if kwargs is None:
        kwargs = {}

    GM = GaussianMixture(n_components=n_components, random_state=0, **kwargs)

    means = GM.fit(x).means_.squeeze()
    covs = GM.fit(x).covariances_.squeeze()
    weights = GM.fit(x).weights_
    data_probs = GM.fit(x).predict_proba(x)

    mean_argsort_idx = np.argsort(means)

    GMM_dict = {'means': means[mean_argsort_idx],
                'covs': covs[mean_argsort_idx],
                'weights': weights[mean_argsort_idx],
                'data_probs': data_probs[:,mean_argsort_idx],
                'n_components': n_components}

    return GMM_dict

    
def generate_GMM_labels(adata: ad.AnnData,
                        GMM_adata: ad.AnnData,
                        GMM_data_probs: np.ndarray,
                        argsort_mapping_dict: dict,
                        color_mapping_dict: dict,
                        obs_label: str ='GMM_phase'):

    argsort_data_probs = np.argmax(GMM_data_probs, axis=1)

    argsort_phases = [argsort_mapping_dict[arg_idx] for arg_idx in argsort_data_probs]

    GMM_adata_cell_IDs = GMM_adata.obs['CellID'].copy()

    GMM_adata_CI_idxs, _ = get_str_idx(GMM_adata_cell_IDs, adata.obs['CellID'])

    temp_GMM_phases = adata.obs[obs_label].copy()

    for idx, CI_idx in enumerate(GMM_adata_CI_idxs):
        temp_GMM_phases[CI_idx] = argsort_phases[idx]

    adata.obs[obs_label] =  temp_GMM_phases
    adata.obs[f"{obs_label}_colors"] = [color_mapping_dict[phase] for phase in adata.obs[obs_label]]

    return adata
