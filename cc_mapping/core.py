
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

from cc_mapping.utils import get_str_idx

def random_forest_feature_selection(adata: ad.AnnData,
                                    training_feature_set: str,
                                    training_labels: str,
                                    feature_set_name: str = None,
                                    method: str = 'RF_min_max',
                                    stable_counter: int = 10,
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

    feature_set_idxs, _ = get_str_idx(training_feature_set, adata.var_names.values)

    try:
        phase_nan_idx, _ = get_str_idx('nan', adata.obs[training_labels])
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

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    # Instantiate model 
    rf_classifier = RandomForestClassifier(
                        min_samples_leaf=50,
                        n_estimators=150,
                        bootstrap=True,
                        oob_score=True,
                        n_jobs=-1,
                        random_state=42)

    rf_classifier.fit(train_features, train_labels)

    rf_pred_labels=rf_classifier.predict(test_features)
    accuracy = metrics.accuracy_score(test_labels, rf_pred_labels)

    print()
    print(f'Classification Report for RF model trained with given feature set')
    print('##################################################################')
    print()
    print(metrics.classification_report(test_labels, rf_pred_labels))
        
    #negative to have it sort from highest to lowest
    sorted_idxs = np.argsort(-rf_classifier.feature_importances_)
    sorted_feature_set = np.array(training_feature_set)[sorted_idxs]
    sorted_features = trunc_cell_data[:,sorted_idxs]

    if method == 'RF_min_30':
        optim_RF_feature_set = sorted_feature_set[:30]
    elif method == 'RF_min_max':

        counter = 0
        max_acc_arg = 0
        acc_list = []
        first=True
        for num_cols in tqdm(range(1, sorted_features.shape[1]+1),total= sorted_features.shape[1], desc = f'Training RF model iteratively using most important RF features until {stable_counter} stable iterations'):

            trunc_feature_set = sorted_features[:,:num_cols]

            trunc_train_features, trunc_test_features, trunc_train_labels, trunc_test_labels = train_test_split(trunc_feature_set, labels, test_size = 0.25, random_state = 42)

            trunc_rf_classifier = RandomForestClassifier(
                                min_samples_leaf=50,
                                n_estimators=150,
                                bootstrap=True,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=42)

            trunc_rf_classifier.fit(trunc_train_features, trunc_train_labels)

            trunc_pred_labels=trunc_rf_classifier.predict(trunc_test_features)

            accuracy = metrics.accuracy_score(trunc_test_labels, trunc_pred_labels)
            acc_list.append([num_cols,accuracy])

            if max_acc_arg == np.argmax(np.array(acc_list)[:,1]):
                best_model = True
                counter += 1
            else:
                best_model = False
                max_acc_arg = np.argmax(np.array(acc_list)[:,1])
                counter = 0

            if first ==True or best_model == False:
                best_rf_classifier = trunc_rf_classifier
                best_train_features = trunc_train_features
                best_train_labels = trunc_train_labels
                best_test_features = trunc_test_features
                best_test_labels = trunc_test_labels
                first = False

            if counter == stable_counter:
                break


        acc_list = np.array(acc_list)
        max_accuracy = acc_list[max_acc_arg,1]
        optim_feat_num = max_acc_arg + 1
        optim_RF_feature_set = sorted_feature_set[:optim_feat_num]

    print()
    print(f'Classification Report for Optimal RF model trained with {optim_feat_num} features')
    print('##################################################################################')
    print()

    trunc_rf_classifier = RandomForestClassifier(
                        min_samples_leaf=50,
                        n_estimators=150,
                        bootstrap=True,
                        oob_score=True,
                        n_jobs=-1,
                        random_state=42)

    trunc_rf_classifier.fit(best_train_features, best_train_labels)

    trunc_pred_labels=best_rf_classifier.predict(best_test_features)

    print(metrics.classification_report(best_test_labels, trunc_pred_labels))

    print()
    print('##################################################################################')
    print()

    print('Optimal Feature Set sorted by RF feature importance')
    print('###################################################')
    print(optim_RF_feature_set)

    feat_idxs, _ = get_str_idx(optim_RF_feature_set, adata.var_names.values)

    fs_bool = np.repeat(False, adata.shape[1])
    fs_bool[feat_idxs] = True

    if feature_set_name is None:
        feature_set_name = f'{method}_feature_set'

    adata.var[feature_set_name] = fs_bool

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
