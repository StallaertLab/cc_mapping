
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
np.seterr(all="ignore")
import anndata as ad
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append(r'C:\Users\dap182\Documents\Stallaert_lab\PDAC_pipeline')

from cc_mapping.utils import get_str_idx

def random_forest_feature_selection(adata: ad.AnnData,
                                    training_feature_set: str,
                                    training_labels: str,
                                    feature_set_name: str = None,
                                    method: str = 'RF_min_max',
                                    ):

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
    train_features, _, train_labels, _ = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    # Instantiate model 
    rf_classifier = RandomForestClassifier(
                        min_samples_leaf=50,
                        n_estimators=150,
                        bootstrap=True,
                        oob_score=True,
                        n_jobs=-1,
                        random_state=42)

    rf_classifier.fit(train_features, train_labels)
        
    #negative to have it sort from highest to lowest
    sorted_idxs = np.argsort(-rf_classifier.feature_importances_)
    sorted_feature_set = np.array(training_feature_set)[sorted_idxs]
    sorted_features = trunc_cell_data[:,sorted_idxs]

    if method == 'RF_min_30':
        optim_RF_feature_set = sorted_feature_set[:30]
    elif method == 'RF_min_max':

        stable_counter = 0
        max_acc_arg = 0
        acc_list = []
        for num_cols in range(1, sorted_features.shape[1]+1):

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
                stable_counter += 1
            else:
                max_acc_arg = np.argmax(np.array(acc_list)[:,1])
                stable_counter = 0

            if stable_counter == 10:
                break


        acc_list = np.array(acc_list)
        max_accuracy = acc_list[max_acc_arg,1]
        optim_feat_num = max_acc_arg + 1
        optim_RF_feature_set = sorted_feature_set[:optim_feat_num]


    feat_idxs, _ = get_str_idx(optim_RF_feature_set, adata.var_names.values)

    fs_bool = np.repeat(False, adata.shape[1])
    fs_bool[feat_idxs] = True

    if feature_set_name is None:
        feature_set_name = f'{method}_feature_set'

    adata.var[feature_set_name] = fs_bool

    return adata

