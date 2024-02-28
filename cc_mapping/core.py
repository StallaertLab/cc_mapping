
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import re
from sklearn.preprocessing import LabelEncoder
import numpy as np
np.seterr(all="ignore")
import anndata as ad
import matplotlib.patches as mpatches
from scipy import stats as st
from tqdm import tqdm
from sklearn import svm
from collections import OrderedDict
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture 

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import sys
from GLOBAL_VARIABLES.GLOBAL_VARIABLES import cc_mapping_package_dir
sys.path.append(cc_mapping_package_dir)

from cc_mapping import utils, preprocess

def train_random_forest_model(features,
                              labels,
                              rf_params: dict ,
                              random_state: bool,
                              train_test_split_params: dict,
                              verbose: bool = True,
                              ):
    """
    Trains a random forest model using the given features and labels.

    Parameters:
    - features: The input features for training the model.
    - labels: The target labels for training the model.
    - rf_params: A dictionary of parameters for the random forest classifier.
    - random_state: A boolean value indicating whether to use a random state for reproducibility.
    - train_test_split_params: A dictionary of parameters for the train-test split.
    - verbose: A boolean value indicating whether to print the classification report.

    Returns:
    - rf_classifier: The trained random forest classifier.
    - accuracy: The accuracy of the model on the test set.
    """
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

def RF_increment_counter(acc_list, optim_feat_num, counter, stable_counter, threshold):
    """
    Increment the counter based on the accuracy difference between the current feature and the previous feature.

    Parameters:
    acc_list (list): List of accuracy values.
    optim_feat_num (int): Index of the current feature.
    counter (int): Counter value.
    stable_counter (int): Number of stable features.
    threshold (float): Threshold value for accuracy difference.

    Returns:
    tuple: A tuple containing the updated optim_feat_num, counter, and continue_bool values.
    """
    trun_acc_list = np.array(acc_list[optim_feat_num:optim_feat_num+stable_counter+1])
    acc_diff_list = trun_acc_list - acc_list[optim_feat_num]

    diff_surpass_threshold = np.where(acc_diff_list > threshold)[0]

    if len(diff_surpass_threshold)==0:
        continue_bool = False
    else:
        if counter != 0:
            counter -= 1

        optim_feat_num += 1
        continue_bool = True

    return optim_feat_num, counter , continue_bool

def random_forest_feature_selection(adata: ad.AnnData,
                                    training_feature_set: str,
                                    training_labels: str,
                                    feature_set_name: str = None,
                                    method: str = 'RF_min_max',
                                    random_state: int = 42,
                                    threshold: float = 0.01,
                                    stable_counter: int = 3,
                                    plot: bool = True,
                                    cutoff_method: str = 'increment',
                                    train_test_split_params: dict = {'test_size':0.25},
                                    rf_params: dict = {'min_samples_leaf':50,
                                                        'n_estimators':150,
                                                        'bootstrap':True,
                                                        'oob_score':True,
                                                        'n_jobs':-1, },
                                    ) -> ad.AnnData:
                                    
    """
    Trains a random forest classifier on the training feature set and labels using one of two methods:
    RF_min_30: Selects the top 30 features based on the random forest feature importance
    RF_min_max: Selects the minimum number of features that maximizes the accuracy of the random forest classifier
    This is done by iteratively adding features to the feature set until the accuracy of the classifier
    does not improve for x number of iterations

    Args:
        adata (ad.AnnData): The AnnData object containing the data.
        training_feature_set (str): The name of the feature set to be used for training.
        training_labels (str): The name of the labels to be used for training.
        feature_set_name (str, optional): The name of the feature set to be added to the .var attribute of the adata object. Defaults to None.
        method (str, optional): The method to be used for feature selection. Defaults to 'RF_min_max'.
        random_state (int, optional): The random state for reproducibility. Defaults to 42.
        threshold (float, optional): The threshold for determining when to stop adding features. Defaults to 0.01.
        stable_counter (int, optional): The number of stable iterations before stopping. Defaults to 3.
        plot (bool, optional): Whether to plot the accuracy vs. number of features graph. Defaults to True.
        cutoff_method (str, optional): The method for determining when to stop adding features. Defaults to 'increment'.
        train_test_split_params (dict, optional): The parameters for train test split. Defaults to {'test_size':0.25}.
        rf_params (dict, optional): The parameters for the random forest classifier. Defaults to {'min_samples_leaf':50, 'n_estimators':150, 'bootstrap':True, 'oob_score':True, 'n_jobs':-1}.

    Returns:
        ad.AnnData: The adata object with the feature set added to the .var attribute.
    """

    if feature_set_name is None:
        feature_set_name = f'{method}_feature_set'

    # Get the indices of the features in the training feature set
    feature_set_idxs, _ = utils.get_str_idx(training_feature_set, adata.var_names.values)

    #remove the nan values from the feature set
    #TODO: I need to make this more generalizable because there are other forms of nan in the data
    try:
        phase_nan_idx, _ = utils.get_str_idx('nan', adata.obs[training_labels])
    except:
        phase_nan_idx = []

    #isolates the feature set from the adata object
    feature_set = adata.X[:,feature_set_idxs].copy()

    #removes the nan values from the feature set and labels
    feature_set = np.delete(feature_set, phase_nan_idx, axis=0)
    labels = np.delete(adata.obs[training_labels].values, phase_nan_idx, axis=0)

    #remove the inf values from the feature set
    feature_set[feature_set == np.inf] = np.nan
    feature_set[feature_set == -np.inf] = np.nan
    nan_data_idx = np.isnan(feature_set).any(axis=1)

    feature_set = feature_set[~nan_data_idx]
    labels = labels[~nan_data_idx]

    #train the random forest model on all the features to get the feature importances
    rf_classifier, _ = train_random_forest_model(features = feature_set,
                                                 labels = labels,
                                                 rf_params = rf_params,
                                                 train_test_split_params=train_test_split_params,
                                                 random_state=random_state)
        
    #negative to have it sort from highest to lowest
    sorted_idxs = np.argsort(-rf_classifier.feature_importances_)
    sorted_feature_set = np.array(training_feature_set)[sorted_idxs]
    sorted_features = feature_set[:,sorted_idxs]

    #if the method is RF_min_30, then the optimal feature set is the top 30 features
    #this works with any number of features
    if re.search('(?<=RF_min_)[0-9]+',method):
        optim_feat_num = int(re.search('(?<=RF_min_)[0-9]+',method).group(0))
        optim_RF_feature_set = sorted_feature_set[:optim_feat_num]
        
    elif method == 'RF_min_max':

        counter = 0
        max_acc_arg = 0
        acc_list = [0]
        for num_feats in tqdm(range(1, sorted_features.shape[1]+1),total= sorted_features.shape[1], desc = f'Training RF model iteratively using most important RF features until {stable_counter} stable iterations'):

            if counter > stable_counter:
                break

            #truncates the feature set to the current number of features
            trunc_feature_set = sorted_features[:,:num_feats]

            #trains the random forest model on the truncated feature set
            _ , accuracy = train_random_forest_model(features = trunc_feature_set,
                                                        labels = labels,
                                                        rf_params = rf_params,
                                                        train_test_split_params=train_test_split_params,
                                                        random_state=random_state,
                                                        verbose=False)
            acc_list.append(accuracy)

            #calulate the difference between the current accuracy and the maximum accuracy determined before this iteration
            acc_difference = np.abs(acc_list[max_acc_arg] - accuracy)

            #if the maximum accuracy is the same as the current accuracy, pass on
            if max_acc_arg == np.argmax(acc_list):
                pass

            # if the difference between the current accuracy and the maximum accuracy is greater than the threshold
            # and the number of features is greater than 1 (to avoid the 0th index in the accuracy list)
            #elif acc_difference > threshold and num_feats > 1:
            elif acc_difference > threshold:

                #if the cuttoff method is jump, then if the difference is greater than the threshold,
                #then set the maximum accuracy argument to the current number of features and reset counter
                if cutoff_method == 'jump':
                    max_acc_arg = np.argmax(acc_list)
                    counter = 0

                #if the cuttoff method is increment, then if the difference is greater than the threshold,
                #then the optimal number of features is increased by 1 and the check occurs again until the condition is not met
                #when the acc_diccerece is less than the threshold, the counter also decreased by 1 
                elif cutoff_method == 'increment':

                    temp_max_acc_arg, temp_counter = max_acc_arg, counter
                    for _ in range(stable_counter):
                        temp_max_acc_arg, temp_counter, continue_bool = RF_increment_counter(acc_list, temp_max_acc_arg, temp_counter, stable_counter, threshold)

                        if not continue_bool:
                            break

                    if (temp_max_acc_arg, temp_counter) != (max_acc_arg, counter):
                        max_acc_arg = temp_max_acc_arg
                        counter = temp_counter

            counter += 1

        acc_list = np.array(acc_list)
        optim_feat_num = max_acc_arg 

    optim_RF_feature_set = sorted_feature_set[:optim_feat_num]

    optim_feature_set = sorted_features[:,:optim_feat_num]

    #train the random forest model on the optimal feature set for output performance metrics
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

    #converts the optimal feature set to a boolean array
    feat_idxs, _ = utils.get_str_idx(optim_RF_feature_set, adata.var_names.values)
    fs_bool = np.repeat(False, adata.shape[1])
    fs_bool[feat_idxs] = True

    adata.var[feature_set_name] = fs_bool

    if plot and method == 'RF_min_max':
        fig = plt.figure(figsize=(10,5))

        x_axis = np.arange( len(acc_list), dtype=int)
        plt.plot(x_axis, acc_list)
        plt.axvline(optim_feat_num, color='r', linestyle='--', label=f'Optimal Feature Set Size: {optim_feat_num}')
        plt.title(f'Stable Counter {stable_counter} - Stable Threshold {threshold*100}% - Cutoff Method: {cutoff_method}')
        plt.xticks(x_axis)

        percentages = np.arange(0,110,10,dtype=int).tolist()
        rounded_percentages =  [ f'{elem} %' for elem in percentages ]
        plt.grid(visible = True, alpha = 0.5, linestyle = '--', which = 'both')
        plt.yticks(np.arange(0,1.1,0.1), rounded_percentages)
        
        xtick_labels = np.insert(sorted_feature_set[:len(acc_list)-1], 0, '')
        plt.xticks(np.arange(0, len(acc_list)),xtick_labels, rotation = 45, ha = 'right', fontsize = 8)
        plt.ylabel('Accuracy')
        plt.xlim(0, len(acc_list)-1)
        plt.ylim(0,1.05)
        plt.legend(loc='lower right')
        plt.tight_layout()

    return adata

class GMM_CC_phase_prediction():
    """
    Class for performing GMM-based phase prediction in single-cell RNA-seq data.

    Parameters:
    - adata: Anndata object containing the single-cell RNA-seq data.
    - GMM_obs_label: String, the name of the observation column to store the GMM phase labels.
    - GMM_kwargs: Dictionary, additional keyword arguments for the GaussianMixture model.

    Methods:
    - row_data_partitioning: Partition the data based on a search term and phase observation label.
    - define_gene_adata: Define the Anndata object for a specific gene, with or without row data partitioning.
    - fit_GMM: Fit a GaussianMixture model to a specific gene and assign phase labels.
    - compare_GMM_labels: Compare phase labels between two GMM sets and assign new labels.
    - define_GMM_compare_parameters: Define parameters for comparing GMM sets.
    - plot_GMM: Plot the GaussianMixture model and the linear decision boundaries.
    - plot_linear_decision_boundaries: Plot the linear decision boundaries for a specific gene.
    """
    def __init__(self,
                 adata,
                 GMM_obs_label: str = 'GMM_phase_labels',
                 GMM_kwargs: dict = None,):

        self.adata = adata

        self.GMM_obs_label = GMM_obs_label

        if self.GMM_obs_label in adata.obs.columns:
            del self.adata.obs[self.GMM_obs_label]

        if GMM_kwargs is None:
            self.GMM_kwargs = {}
        else:
            self.GMM_kwargs = GMM_kwargs

        self.cc_phase_info_dict = {}

    def row_data_partitioning(self,
                                  GMM_set_name: str,
                                  obs_search_term: str,
                                  GMM_set_name_to_partition: str = None,
                                  phase_obs_label: str = None):
            """
            Partition the row data based on the given parameters.

            Parameters:
            - GMM_set_name (str): The name of the GMM set.
            - obs_search_term (str): The search term for selecting observations.
            - GMM_set_name_to_partition (str, optional): The name of the GMM set to partition. Defaults to None.
            - phase_obs_label (str, optional): The label for phase observations. Defaults to None.

            Returns:
            None
            """
            if phase_obs_label is None:
                phase_obs_label = self.GMM_obs_label

            if GMM_set_name_to_partition is None:
                trunc_adata = preprocess.row_data_partitioning(self.adata, obs_search_term, phase_obs_label)
            else:
                gene_adata = self.cc_phase_info_dict[GMM_set_name_to_partition]['gene_adata']

                gene_adata = preprocess.row_data_partitioning(gene_adata, obs_search_term, phase_obs_label)
                cell_ids = gene_adata.obs['CellID']

                trunc_adata = self.adata[self.adata.obs['CellID'].isin(cell_ids)].copy()

            self.cc_phase_info_dict[GMM_set_name] = {}
            self.cc_phase_info_dict[GMM_set_name]['trunc_adata'] = trunc_adata

    def define_gene_adata(self,
                              gene: str,
                              row_data_partitioning: bool = False,
                              GMM_set_name: bool = None):
            """
            Returns a subset of the AnnData object containing the expression data for a specific gene.

            Parameters:
                gene (str): The name of the gene.
                row_data_partitioning (bool, optional): If True, returns a subset of the AnnData object based on row data partitioning. Defaults to False.
                GMM_set_name (bool, optional): The name of the GMM set. Required if row_data_partitioning is True. Defaults to None.

            Returns:
                AnnData: A subset of the AnnData object containing the expression data for the specified gene.
            """

            if not row_data_partitioning:
                return self.adata[:, gene].copy()
            else:
                trunc_adata = self.cc_phase_info_dict[GMM_set_name]['trunc_adata']
                return trunc_adata[:, gene].copy()

    def fit_GMM(self,
                    gene: str,
                    ordered_GMM_labels: list,
                    random_state: int = 0,
                    n_components: int = None,
                    GMM_set_name: str = None,
                    GMM_kwargs: dict = None,
                    row_data_partitioning: bool = False,):
            """
            Fits a Gaussian Mixture Model (GMM) to the gene expression data for a specific gene.

            Args:
                gene (str): The name of the gene.
                ordered_GMM_labels (list): The ordered labels for the GMM components.
                random_state (int, optional): The random seed for reproducibility. Defaults to 0.
                n_components (int, optional): The number of components in the GMM. Defaults to None.
                GMM_set_name (str, optional): The name of the GMM set. Defaults to None.
                row_data_partitioning (bool, optional): Whether to perform row data partitioning. Defaults to False.

            Returns:
                None
            """
            gene_adata = self.define_gene_adata(gene, row_data_partitioning, GMM_set_name)

            x = gene_adata.X.copy()

            if GMM_kwargs is None:
                GMM_kwargs = self.GMM_kwargs

            GM = GaussianMixture(n_components=n_components, random_state=random_state, **GMM_kwargs)

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

            gene_adata.uns[f'{gene}_GMM_dict'] = GMM_dict

            data_probs = gene_adata.uns[f'{gene}_GMM_dict']['data_probs'].copy()
            argmax_data_probs = np.argmax(data_probs, axis=1)
            phase_labels = [ordered_GMM_labels[i] for i in argmax_data_probs]

            gene_adata.obs[self.GMM_obs_label] = phase_labels

            if GMM_set_name is None:
                GMM_set_name = str(len(self.cc_phase_info_dict.keys()))

            self.cc_phase_info_dict[GMM_set_name] = {'gene':gene,
                                                     'gene_adata':gene_adata,
                                                     'n_components':n_components,
                                                     'ordered_GMM_labels':ordered_GMM_labels,
                                                     'row_data_partitioning':row_data_partitioning,}

    #labels for the GMM set names lave two labels, one with ~ and one without (ie. G0/~G0)
    def compare_GMM_labels(self, 
                                GMM_set_name_list: list,
                                GMM_set_save_name: str):
            """
            Compare the GMM labels for two sets and update the gene_adata object with the results.

            Parameters:
            - GMM_set_name_list (list): A list of two GMM set names to compare.
            - GMM_set_save_name (str): The name to save the GMM set comparison results.

            Returns:
            None
            """
            c1_label = GMM_set_name_list[0]
            c2_label = GMM_set_name_list[1]

            genes = [self.cc_phase_info_dict[key]['gene'] for key in GMM_set_name_list]

            compare_dict = self.define_GMM_compare_parameters(GMM_set_name_list)
        
            labels = [compare_dict[key]['c_positive_label'] for key in GMM_set_name_list]

            c1_pidxs = compare_dict[c1_label]['c_pidxs']
            c2_pidxs = compare_dict[c2_label]['c_pidxs']

            c1c2_idxs = np.intersect1d(c1_pidxs, c2_pidxs)

            c1_selected_data_probs = compare_dict[c1_label]['c_data_probs'][c1c2_idxs]
            c2_selected_data_probs = compare_dict[c2_label]['c_data_probs'][c1c2_idxs]

            c1c2_data_probs = np.concatenate([c1_selected_data_probs[np.newaxis,:],
                                                c2_selected_data_probs[np.newaxis,:]],axis=0)
            
            max_c1c2_data_probs = np.max(c1c2_data_probs, axis=2)

            argmax_c1c2_data_probs = np.argmax(max_c1c2_data_probs, axis=0)

            c1c2_labels = [labels[arg_idx] for arg_idx in argmax_c1c2_data_probs]

            gene_adata = self.define_gene_adata(genes)
            gene_adata.obs[self.GMM_obs_label] = np.repeat('NA',gene_adata.shape[0])
            GMM_labels = gene_adata.obs[self.GMM_obs_label].copy()

            GMM_labels[c1_pidxs] = compare_dict[c1_label]['c_positive_label']
            GMM_labels[c2_pidxs] = compare_dict[c2_label]['c_positive_label'] 
            GMM_labels[c1c2_idxs] = c1c2_labels

            gene_adata.obs[self.GMM_obs_label] = GMM_labels

            self.cc_phase_info_dict[GMM_set_save_name] = {'GMM_set_name_list':GMM_set_name_list,
                                                          'gene_adata':gene_adata.copy()}

    def define_GMM_compare_parameters(self, GMM_set_name_list: list):
        """
        Define GMM compare parameters.

        Args:
            GMM_set_name_list (list): List of GMM set names.

        Returns:
            dict: Dictionary containing compare parameters for each GMM set name.
                The dictionary has the following structure:
                {
                    'GMM_set_name': {
                        'c_data_probs': c_data_probs,
                        'c_positive_label': c_positive_label,
                        'c_pidxs': c_pidxs
                    },
                    ...
                }
        """
        compare_dict = OrderedDict()

        for key in GMM_set_name_list:
            c_phase_dict = self.cc_phase_info_dict[key]
            c_gene = c_phase_dict['gene']
            c_adata = c_phase_dict[f'gene_adata']
            c_data_probs = c_adata.uns[f'{c_gene}_GMM_dict']['data_probs']
            c_phase_labels = c_adata.obs[self.GMM_obs_label]
            c_positive_label = np.unique(c_phase_labels)[1].replace('~', '')
            c_pidxs, _ = utils.get_str_idx(c_positive_label, c_phase_labels)

            compare_dict[key] = {
                'c_data_probs': c_data_probs,
                'c_positive_label': c_positive_label,
                'c_pidxs': c_pidxs,
            }

        return compare_dict

    def plot_GMM(self,
                    GMM_set_name: str,
                    num_std: int =3,
                    hist_kwargs: dict = None,
                    cmap: plt.cm = plt.cm.rainbow,
                    unit_size: int = 5,
                    ratio: tuple = (1,1),
                    resolution: int = 1000,):
            """
            Plots the Gaussian Mixture Model (GMM) for a given GMM set.

            Parameters:
            - GMM_set_name (str): The name of the GMM set.
            - num_std (int): The number of standard deviations to include in the plot.
            - hist_kwargs (dict): Optional keyword arguments for customizing the histogram.
            - cmap (plt.cm): The colormap to use for the GMM plot.
            - unit_size (int): The size of the plot in inches.
            - ratio (tuple): The ratio of the plot width to height.
            - resolution (int): The resolution of the plot.

            Returns:
            None
            """
            gene_adata = self.cc_phase_info_dict[GMM_set_name]['gene_adata']
            labels = self.cc_phase_info_dict[GMM_set_name]['ordered_GMM_labels']

            gene = gene_adata.var_names[0]
            gene_x = gene_adata.X.copy()
            GMM_dict = gene_adata.uns[f'{gene}_GMM_dict']

            n_components = GMM_dict['n_components']
            means = GMM_dict['means']
            covs = GMM_dict['covs']
            weights = GMM_dict['weights']

            colors = cmap(np.linspace(0, 1, n_components))

            row_ratio, col_ratio = ratio
            fig = plt.subplots(1,1,figsize=(col_ratio*unit_size,  row_ratio*unit_size))

            for guas_idx in range(n_components):
                g_mean = means[guas_idx]
                g_cov = covs[guas_idx]
                g_weight = weights[guas_idx]

                std = np.sqrt(g_cov)

                x_min = g_mean - num_std*std
                x_max = g_mean + num_std*std

                x_axis = np.linspace(x_min, x_max, resolution)

                y_axis = st.norm.pdf(x_axis, loc=g_mean, scale=std)*g_weight

                plt.plot(x_axis, y_axis, lw=3, color=colors[guas_idx], zorder=4)
                plt.axvline(g_mean, color=colors[guas_idx], lw=2, ls='--',zorder=3)

            if hist_kwargs is None:
                hist_kwargs = {"bins":100, 'color':'black', "density": True,}

            plt.hist(gene_x, **hist_kwargs, zorder=1)

            hist_y_lims = plt.ylim()
            lower_hist_x_lim , upper_hist_x_lim = plt.xlim()
             
            if lower_hist_x_lim < 0:
                hist_x_lims = (0, upper_hist_x_lim)
            else:
                hist_x_lims = plt.xlim()

            self.plot_linear_decision_boundaries(gene_adata,
                                                gene_x,
                                                labels,
                                                hist_x_lims,
                                                hist_y_lims,
                                                resolution,
                                                cmap,)

            patch_list = []
            for label, color in zip(labels, colors):
                patch = mpatches.Patch(facecolor=color, label=label, alpha=0.8,  edgecolor='black')
                patch_list.append(patch)

            plt.legend(handles = patch_list)

            plt.xlim(hist_x_lims)
            plt.title(f'{gene}')
        
        
    def merge_GMM_adata_labels(self, GMM_set_name_list):

        #self.adata.obs[self.GMM_obs_label] = np.repeat('NA', self.adata.shape[0])
        GMM_labels = np.repeat('NA', self.adata.shape[0])   

        for GMM_set_name in GMM_set_name_list:
            GMM_adata = self.cc_phase_info_dict[GMM_set_name]['gene_adata']

            GMM_cell_ids = GMM_adata.obs['CellID'].copy()

            GMM_idxs, _ = utils.get_str_idx(GMM_cell_ids, self.adata.obs['CellID'])
            
            GMM_labels[GMM_idxs] = GMM_adata.obs[self.GMM_obs_label]
        
        self.adata.obs[self.GMM_obs_label] = GMM_labels


    def plot_linear_decision_boundaries(self,
                                        gene_adata: ad.AnnData,
                                        gene_x: np.ndarray,
                                        labels: list,
                                        x_lims: tuple,
                                        y_lims: tuple,
                                        resolution: int,
                                        cmap: plt.cm,):

        """
        Plots linear decision boundaries using Support Vector Machine (SVM) for classification.

        Parameters:
        - gene_adata (ad.AnnData): Annotated data matrix containing gene expression data.
        - gene_x (np.ndarray): Array of gene expression values.
        - labels (list): List of class labels.
        - x_lims (tuple): Tuple specifying the x-axis limits for the plot.
        - y_lims (tuple): Tuple specifying the y-axis limits for the plot.
        - resolution (int): Number of points to generate along each axis for the decision boundaries.
        - cmap (plt.cm): Colormap for the plot.

        Returns:
        None
        """
        #encoding the labels from strings to integers
        encoder = LabelEncoder()
        encoder.fit(gene_adata.obs[self.GMM_obs_label].values)
        #Reassigning the classes to the encoder
        encoder.classes_ = np.array(labels)
        encoded_labels = encoder.transform(np.array(gene_adata.obs[self.GMM_obs_label]))

        #prepration of data to train the SVM
        dummy_feature = np.repeat(0, gene_adata.shape[0]).T
        svm_x = np.hstack([gene_x, dummy_feature.reshape(-1,1)])
        SVm = svm.SVC( random_state=0)
        SVm.fit(svm_x, encoded_labels)    

        #Creating the grid to predict the labels of the new data and stacking dummy variable
        test_x_axis = np.linspace(x_lims[0], x_lims[1], resolution)
        svm_pred_x = np.vstack([test_x_axis, np.repeat(0, resolution)]).T
        
        #Using the trained SVM to predict the labels of the new data
        predictions = SVm.predict(svm_pred_x)

        #Reshaping the predictions to be used in the contourf function
        predictions_contourf = np.repeat(predictions.reshape(-1,1), resolution, axis=1).T

        #Creating the array of the y axis to be used in the contourf function
        test_y_axis = np.linspace(y_lims[0], y_lims[1], resolution)

        plt.contourf(test_x_axis,
                    test_y_axis,
                    predictions_contourf,
                    alpha=0.15,
                    cmap=cmap,
                    zorder=2)

    def GMM_BIC_evaluation(self,
                            GMM_set_name: str,
                            bic_range:int = 5,
                            unit_size: int = 5,
                            ratio: tuple = (1,1),):
            """
            Evaluates the Bayesian Information Criterion (BIC) for Gaussian Mixture Models (GMMs).

            Parameters:
            - GMM_set_name (str): The name of the GMM set to evaluate.
            - bic_range (int): The range of number of components to test the BIC metric.
            - unit_size (int): The size of the plot units.
            - ratio (tuple): The ratio of the plot dimensions.

            Returns:
            None
            """
            gene_adata = self.cc_phase_info_dict[GMM_set_name]['gene_adata']
            n_components = self.cc_phase_info_dict[GMM_set_name]['n_components']


            cc_x = gene_adata.X.copy()

            bics = []
            counter=1
            for _ in range (bic_range): # test the AIC/BIC metric between 1 and 10 components
                gmm = GaussianMixture(n_components = counter, **self.GMM_kwargs)
                labels = gmm.fit(cc_x).predict(cc_x)
                bic = gmm.bic(cc_x)
                bics.append(bic)
                counter = counter + 1

            row_ratio, col_ratio = ratio
            fig = plt.figure(figsize=(col_ratio * unit_size, row_ratio * unit_size))

            plt.plot(np.arange(1,bic_range+1), bics, 'o-', lw=3, c='black', label='BIC')
            plt.xlabel('Number of components', fontsize=10)
            plt.ylabel('Information criterion', fontsize=10)
            plt.axvline(n_components, color='red', linestyle='--', lw=3)
            plt.xticks(np.arange(0,bic_range+1, 1))
            plt.title(f'GMM BIC evaluation')
            plt.tight_layout()

