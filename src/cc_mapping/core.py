from typing import Optional
from collections import OrderedDict
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
np.seterr(all="ignore")

import re
import anndata as ad
from scipy import stats as st
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from .utils import get_str_idx
from .preprocess import row_data_partitioning


def train_random_forest_model(
    features,
    labels,
    rf_params: dict,
    random_state: int,
    train_test_split_params: dict,
    feature_set_description: str = "",
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
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, random_state=random_state, **train_test_split_params
    )
    rf_classifier = RandomForestClassifier(random_state=random_state, **rf_params)

    rf_classifier.fit(train_features, train_labels)

    rf_pred_labels = rf_classifier.predict(test_features)

    accuracy = metrics.accuracy_score(test_labels, rf_pred_labels)

    if verbose:
        print(f"Classification Report for RF model trained with {feature_set_description} feature set")
        print("##################################################################")
        print()
        print(metrics.classification_report(test_labels, rf_pred_labels))

    return rf_classifier, accuracy


def __random_forest_increment_counter(
    acc_list, optim_feat_num, counter, stable_counter, threshold
):
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
    trun_acc_list = np.array(
        acc_list[optim_feat_num : optim_feat_num + stable_counter + 1]
    )
    acc_diff_list = trun_acc_list - acc_list[optim_feat_num]

    diff_surpass_threshold = np.where(acc_diff_list > threshold)[0]

    if len(diff_surpass_threshold) == 0:
        continue_bool = False
    else:
        if counter != 0:
            counter -= 1

        optim_feat_num += 1
        continue_bool = True

    return optim_feat_num, counter, continue_bool


def random_forest_feature_selection(
    adata: ad.AnnData,
    training_feature_set: str,
    training_labels: str,
    feature_set_name: str = None,
    method: str = "RF_min_max",
    random_state: int = 42,
    threshold: float = 0.01,
    stable_counter: int = 3,
    plot: bool = True,
    verbose: bool = True,
    save_path: str = None,
    cutoff_method: str = "increment",
    train_test_split_params: Optional[dict]=  None,
    rf_params: Optional[dict] = None,
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

    if rf_params is None:
        rf_params = { "min_samples_leaf": 50,
            "n_estimators": 150,
            "bootstrap": True,
            "oob_score": True,
            "n_jobs": -1,
        }
    if train_test_split_params is None:
        train_test_split_params = {"test_size": 0.25}

    if feature_set_name is None:
        feature_set_name = f"{method}_feature_set"

    # Get the indices of the features in the training feature set
    feature_set_idxs, _ = get_str_idx(training_feature_set, adata.var_names.values)

    # remove the nan values from the feature set
    # TODO: I need to make this more generalizable because there are other forms of nan in the data
    try:
        phase_nan_idx, _ = get_str_idx("nan", adata.obs[training_labels])
    except KeyError:
        phase_nan_idx = []

    # isolates the feature set from the adata object
    feature_set = adata.X[:, feature_set_idxs].copy()

    # removes the nan values from the feature set and labels
    feature_set = np.delete(feature_set, phase_nan_idx, axis=0)
    labels = np.delete(adata.obs[training_labels].values, phase_nan_idx, axis=0)

    # remove the inf values from the feature set
    feature_set[feature_set == np.inf] = np.nan
    feature_set[feature_set == -np.inf] = np.nan
    nan_data_idx = np.isnan(feature_set).any(axis=1)

    feature_set = feature_set[~nan_data_idx]
    labels = labels[~nan_data_idx]

    # train the random forest model on all the features to get the feature importances
    rf_classifier, _ = train_random_forest_model(
        features=feature_set,
        labels=labels,
        rf_params=rf_params,
        train_test_split_params=train_test_split_params,
        verbose=verbose,
        feature_set_description="intial",
        random_state=random_state,
    )

    # negative to have it sort from highest to lowest
    sorted_idxs = np.argsort(-rf_classifier.feature_importances_)
    sorted_feature_set = np.array(training_feature_set)[sorted_idxs]
    sorted_features = feature_set[:, sorted_idxs]

    # if the method is RF_min_30, then the optimal feature set is the top 30 features
    # this works with any number of features
    if re.search("(?<=RF_min_)[0-9]+", method):
        optim_feat_num = int(re.search("(?<=RF_min_)[0-9]+", method).group(0))
        optimum_random_forest_feature_set = sorted_feature_set[:optim_feat_num]

    elif method == "RF_min_max":

        counter = 0
        max_acc_arg = 0
        acc_list = [0]
        for num_feats in tqdm(
            range(1, sorted_features.shape[1] + 1),
            total=sorted_features.shape[1],
            desc=f"Training RF model iteratively using most important RF features until {stable_counter} stable iterations",
            disable=not verbose,
        ):

            if counter > stable_counter:
                break

            # truncates the feature set to the current number of features
            trunc_feature_set = sorted_features[:, :num_feats]

            # trains the random forest model on the truncated feature set
            _, accuracy = train_random_forest_model(
                features=trunc_feature_set,
                labels=labels,
                rf_params=rf_params,
                train_test_split_params=train_test_split_params,
                random_state=random_state,
                verbose=False,
            )
            acc_list.append(accuracy)

            # calulate the difference between the current accuracy and the maximum accuracy determined before this iteration
            acc_difference = np.abs(acc_list[max_acc_arg] - accuracy)

            # if the maximum accuracy is the same as the current accuracy, pass on
            if max_acc_arg == np.argmax(acc_list):
                pass

            # if the difference between the current accuracy and the maximum accuracy is greater than the threshold
            # and the number of features is greater than 1 (to avoid the 0th index in the accuracy list)
            # elif acc_difference > threshold and num_feats > 1:
            elif acc_difference > threshold:

                # if the cuttoff method is jump, then if the difference is greater than the threshold,
                # then set the maximum accuracy argument to the current number of features and reset counter
                if cutoff_method == "jump":
                    max_acc_arg = np.argmax(acc_list)
                    counter = 0

                # if the cuttoff method is increment, then if the difference is greater than the threshold,
                # then the optimal number of features is increased by 1 and the check occurs again until the condition is not met
                # when the acc_diccerece is less than the threshold, the counter also decreased by 1
                elif cutoff_method == "increment":

                    temp_max_acc_arg, temp_counter = max_acc_arg, counter
                    for _ in range(stable_counter):
                        temp_max_acc_arg, temp_counter, continue_bool = (
                            __random_forest_increment_counter(
                                acc_list,
                                temp_max_acc_arg,
                                temp_counter,
                                stable_counter,
                                threshold,
                            )
                        )

                        if not continue_bool:
                            break

                    if (temp_max_acc_arg, temp_counter) != (max_acc_arg, counter):
                        max_acc_arg = temp_max_acc_arg
                        counter = temp_counter

            counter += 1

        acc_list = np.array(acc_list)
        optim_feat_num = max_acc_arg

    optimum_random_forest_feature_set = sorted_feature_set[:optim_feat_num]

    optim_feature_set = sorted_features[:, :optim_feat_num]

    # train the random forest model on the optimal feature set for output performance metrics
    _ = train_random_forest_model(
        features=optim_feature_set,
        labels=labels,
        rf_params=rf_params,
        train_test_split_params=train_test_split_params,
        verbose=verbose,
        feature_set_description="optimal",
        random_state=random_state,
    )

    if verbose:
        print()
        print(
            "##################################################################################"
        )
        print()

        print("Optimal Feature Set sorted by RF feature importance")
        print("###################################################")
        print(optimum_random_forest_feature_set)

    # converts the optimal feature set to a boolean array
    feat_idxs, _ = get_str_idx(optimum_random_forest_feature_set, adata.var_names.values)
    fs_bool = np.repeat(False, adata.shape[1])
    fs_bool[feat_idxs] = True

    adata.var[feature_set_name] = fs_bool

    if plot and method == "RF_min_max":
        plt.figure(figsize=(10, 5))

        x_axis = np.arange(len(acc_list), dtype=int)
        plt.plot(x_axis, acc_list)
        plt.axvline(
            optim_feat_num,
            color="r",
            linestyle="--",
            label=f"Optimal Feature Set Size: {optim_feat_num}",
        )
        plt.title(
            f"Stable Counter {stable_counter} - Stable Threshold {threshold*100}% - Cutoff Method: {cutoff_method}"
        )
        plt.xticks(x_axis)

        percentages = np.arange(0, 110, 10, dtype=int).tolist()
        rounded_percentages = [f"{elem} %" for elem in percentages]
        plt.grid(visible=True, alpha=0.5, linestyle="--", which="both")
        plt.yticks(np.arange(0, 1.1, 0.1), rounded_percentages)

        xtick_labels = np.insert(sorted_feature_set[: len(acc_list) - 1], 0, "")
        plt.xticks(
            np.arange(0, len(acc_list)),
            xtick_labels,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        plt.ylabel("Accuracy")
        plt.xlim(0, len(acc_list) - 1)
        plt.ylim(0, 1.05)
        plt.legend(loc="lower right")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.close()

    return adata


class GaussianMixtureModelThersholdingSuite:

    def __init__(
        self,
        adata,
        obs_save_key: str = "labels",
        gaussian_mixture_model_parameters: dict = None,
    ):

        self.adata = adata
        self.default_label = "NA"
        self.obs_save_key = obs_save_key

        if self.obs_save_key in adata.obs.columns:
            del self.adata.obs[self.obs_save_key]

        if gaussian_mixture_model_parameters is None:
            self.gaussian_mixture_model_parameters = {}
        else:
            self.gaussian_mixture_model_parameters = gaussian_mixture_model_parameters

        if "random_state" not in self.gaussian_mixture_model_parameters.keys():
            raise ValueError(
                "The random_state parameter is required for reproducibility."
            )

        self.thrersholding_info_dict = {}

    def gene_adata_row_partitioning(
        self,
        set_name: str,
        obs_search_term: str = None,
        set_name_to_partition: str = None,
        phase_obs_label: str = None,
    ):
        if phase_obs_label is None:
            phase_obs_label = self.obs_save_key

        if obs_search_term is None:
            obs_search_term = self.default_label

        if set_name_to_partition is None:
            trunc_adata = row_data_partitioning(
                self.adata, obs_search_term, phase_obs_label
            )
        else:
            gene_adata = self.thrersholding_info_dict[set_name_to_partition][
                "gene_adata"
            ]

            gene_adata = row_data_partitioning(
                gene_adata, obs_search_term, phase_obs_label
            )
            cell_ids = gene_adata.obs["CellID"]

            trunc_adata = self.adata[self.adata.obs["CellID"].isin(cell_ids)].copy()

        self.thrersholding_info_dict[set_name] = {}
        self.thrersholding_info_dict[set_name]["trunc_adata"] = trunc_adata

    def define_gene_adata(
        self, gene: str, gene_adata_partitioning: bool = False, set_name: bool = None
    ) -> ad.AnnData:

        if not gene_adata_partitioning:
            return self.adata[:, gene].copy()

        trunc_adata = self.thrersholding_info_dict[set_name]["trunc_adata"]
        return trunc_adata[:, gene].copy()

    def fit_gaussian_mixture_model(
        self,
        gene: str,
        ordered_labels: list,
        n_components: int = None,
        set_name: str = None,
        gaussian_mixture_model_parameters: dict = None,
        duplicate_labels: bool = False,
        gene_adata_partitioning: bool = False,
    ):
        gene_adata = self.define_gene_adata(gene, gene_adata_partitioning, set_name)

        x = gene_adata.X.copy()

        if gaussian_mixture_model_parameters is None:
            gaussian_mixture_model_parameters = self.gaussian_mixture_model_parameters

        gmm = GaussianMixture(n_components=n_components, **gaussian_mixture_model_parameters)

        means = gmm.fit(x).means_.squeeze()
        covs = gmm.fit(x).covariances_.squeeze()
        weights = gmm.fit(x).weights_
        data_probs = gmm.fit(x).predict_proba(x)

        mean_argsort_idx = np.argsort(means)

        gaussian_mixture_model_results = {
            "means": means[mean_argsort_idx],
            "covs": covs[mean_argsort_idx],
            "weights": weights[mean_argsort_idx],
            "data_probs": data_probs[:, mean_argsort_idx],
            "n_components": n_components,
        }

        data_probs = gaussian_mixture_model_results["data_probs"].copy()

        if len(ordered_labels) != len(set(ordered_labels)):
            if not duplicate_labels:
                raise ValueError(
                    "The ordered GMM labels contain duplicate values. Please ensure that the labels are unique or set duplicate_labels to True."
                )

            dup_labels_list = set(
                [
                    label
                    for label in ordered_labels
                    if ordered_labels.count(label) > 1
                ]
            )

            temp_data_probs = data_probs.copy()
            label_idxs_to_keep = np.repeat(True, len(ordered_labels))
            cols_to_delete = []
            for dup_label in dup_labels_list:
                dup_idxs = [
                    idx
                    for idx, label in enumerate(ordered_labels)
                    if label == dup_label
                ]
                dup_data_probs = temp_data_probs.copy()[:, dup_idxs]

                max_dup_data_probs = np.max(dup_data_probs, axis=1)

                temp_data_probs[:, dup_idxs[0]] = max_dup_data_probs
                cols_to_delete.extend(dup_idxs[1:])

                label_idxs_to_keep[dup_idxs[1:]] = False

            temp_data_probs = np.delete(temp_data_probs, cols_to_delete, axis=1)
            condensed_labels = [
                label
                for idx, label in enumerate(ordered_labels)
                if label_idxs_to_keep[idx]
            ]

            data_probs = temp_data_probs

        argmax_data_probs = np.argmax(data_probs, axis=1)

        if duplicate_labels:
            phase_labels = [condensed_labels[i] for i in argmax_data_probs]
        else:
            phase_labels = [ordered_labels[i] for i in argmax_data_probs]

        gene_adata.obs[self.obs_save_key] = phase_labels

        if set_name is None:
            set_name = str(len(self.thrersholding_info_dict.keys()))

        gene_adata.uns[f"{gene}_gmm_results"] = gaussian_mixture_model_results

        self.thrersholding_info_dict[set_name] = {
            "gene": gene,
            "gene_adata": gene_adata,
            "n_components": n_components,
            "ordered_labels": ordered_labels,
            "row_data_partitioning": gene_adata_partitioning,
            "duplicate_labels": duplicate_labels,
        }

        if duplicate_labels:
            self.thrersholding_info_dict[set_name]["condensed_labels"] = condensed_labels
            self.thrersholding_info_dict[set_name][
                "condensed_data_probs"
            ] = temp_data_probs

    # labels for the GMM set names lave two labels, one with ~ and one without (ie. G0/~G0)
    def compare_set_labels(self, set_name_list: list, set_save_name: str):
        # TOTEST: Verify set names exist
        # TOTEST: Verify list is 2 elements long

        c1_label = set_name_list[0]
        c2_label = set_name_list[1]

        adata_size_list = []
        for key in set_name_list:
            c_phase_dict = self.thrersholding_info_dict[key]
            adata_size_list.append(c_phase_dict["gene_adata"].shape[0])

        if adata_size_list[0] != adata_size_list[1]:
            raise ValueError(
                "The two GMM sets have different number of cells. Please ensure that the GMM sets have the same number of cells."
            )

        genes = [self.thrersholding_info_dict[key]["gene"] for key in set_name_list]

        compare_dict = self.define_compare_parameters(set_name_list)

        labels = [compare_dict[key]["c_positive_label"] for key in set_name_list]

        c1_pidxs = compare_dict[c1_label]["c_pidxs"]
        c2_pidxs = compare_dict[c2_label]["c_pidxs"]

        c1c2_idxs = np.intersect1d(c1_pidxs, c2_pidxs)

        if len(c1c2_idxs) == 0:
            raise ValueError("The two GMM sets have no common cells.")

        c2_selected_data_probs = compare_dict[c2_label]["c_data_probs"][c1c2_idxs]
        c1_selected_data_probs = compare_dict[c1_label]["c_data_probs"][c1c2_idxs]

        c1c2_data_probs = np.concatenate(
            [
                c1_selected_data_probs[np.newaxis, :],
                c2_selected_data_probs[np.newaxis, :],
            ],
            axis=0,
        )

        max_c1c2_data_probs = np.max(c1c2_data_probs, axis=2)

        argmax_c1c2_data_probs = np.argmax(max_c1c2_data_probs, axis=0)

        c1c2_labels = [labels[arg_idx] for arg_idx in argmax_c1c2_data_probs]

        gene_adata = self.define_gene_adata(genes)
        gene_adata.obs[self.obs_save_key] = np.repeat(
            self.default_label, gene_adata.shape[0]
        )
        labels_from_thresholding = gene_adata.obs[self.obs_save_key].copy()

        labels_from_thresholding[c1_pidxs] = compare_dict[c1_label]["c_positive_label"]
        labels_from_thresholding[c2_pidxs] = compare_dict[c2_label]["c_positive_label"]
        labels_from_thresholding[c1c2_idxs] = c1c2_labels

        gene_adata.obs[self.obs_save_key] = labels_from_thresholding

        self.thrersholding_info_dict[set_save_name] = {
            "set_name_list": set_name_list,
            "gene_adata": gene_adata.copy(),
        }

    def define_compare_parameters(self, set_name_list: list):
        compare_dict = OrderedDict()

        for key in set_name_list:
            c_phase_dict = self.thrersholding_info_dict[key]
            c_gene = c_phase_dict["gene"]
            c_adata = c_phase_dict["gene_adata"]

            if c_phase_dict["duplicate_labels"]:
                c_data_probs = c_phase_dict["condensed_data_probs"]
            else:
                c_data_probs = c_adata.uns[f"{c_gene}_gmm_results"]["data_probs"]

            c_phase_labels = c_adata.obs[self.obs_save_key]
            c_positive_label = np.unique(c_phase_labels)[1].replace("~", "")
            c_pidxs, _ = get_str_idx(c_positive_label, c_phase_labels)

            compare_dict[key] = {
                "c_data_probs": c_data_probs,
                "c_positive_label": c_positive_label,
                "c_pidxs": c_pidxs,
            }

        return compare_dict

    def plot_thresholding_results(
        self,
        set_name: str,
        num_std: int = 3,
        hist_kwargs: dict = None,
        cmap: matplotlib.colors.LinearSegmentedColormap = cm.rainbow,
        unit_size: int = 5,
        ratio: tuple = (1, 1),
        x_lim_upper_percentile: int = 100,
        resolution: int = 1000,
        return_fig: bool = False,
    ) -> Optional[plt.figure]:

        cc_phase_dict = self.thrersholding_info_dict[set_name]

        gene_adata = cc_phase_dict["gene_adata"]

        if cc_phase_dict["duplicate_labels"] is True:
            labels = cc_phase_dict["condensed_labels"]
        else:
            labels = cc_phase_dict["ordered_labels"]

        gene = gene_adata.var_names[0]
        gene_x = gene_adata.X.copy()
        gmm_results = gene_adata.uns[f"{gene}_gmm_results"]

        n_components = gmm_results["n_components"]
        means = gmm_results["means"]
        covs = gmm_results["covs"]
        weights = gmm_results["weights"]

        colors = cmap(np.linspace(0, 1, n_components))

        row_ratio, col_ratio = ratio
        fig = plt.subplots(1, 1, figsize=(col_ratio * unit_size, row_ratio * unit_size))

        if hist_kwargs is None:
            hist_kwargs = {
                "bins": 100,
                "color": "black",
                "density": True,
            }

        threshold = np.percentile(gene_x, x_lim_upper_percentile, axis=0)
        trunc_gene_x = gene_x[gene_x < threshold]
        plt.hist(trunc_gene_x, **hist_kwargs, zorder=1)

        hist_y_lims = plt.ylim()
        lower_hist_x_lim, upper_hist_x_lim = plt.xlim()

        if lower_hist_x_lim < 0:
            hist_x_lims = (0, upper_hist_x_lim)
        else:
            hist_x_lims = plt.xlim()

        for guas_idx in range(n_components):
            g_mean = means[guas_idx]
            g_cov = covs[guas_idx]
            g_weight = weights[guas_idx]

            std = np.sqrt(g_cov)

            x_min = g_mean - num_std * std
            x_max = g_mean + num_std * std

            x_axis = np.linspace(x_min, x_max, resolution)

            y_axis = st.norm.pdf(x_axis, loc=g_mean, scale=std) * g_weight

            plt.plot(x_axis, y_axis, lw=3, color=colors[guas_idx], zorder=4)
            plt.axvline(g_mean, color=colors[guas_idx], lw=2, ls="--", zorder=3)

        self.plot_linear_decision_boundaries(
            gene_adata,
            gene_x,
            labels,
            hist_x_lims,
            hist_y_lims,
            resolution,
            cmap,
        )

        patch_list = []
        for label, color in zip(labels, colors):
            patch = mpatches.Patch(
                facecolor=color, label=label, alpha=0.8, edgecolor="black"
            )
            patch_list.append(patch)

        plt.legend(handles=patch_list)

        plt.xlim(hist_x_lims)
        plt.ylim(hist_y_lims)
        plt.title(f"{gene}")

        if return_fig:
            return fig
        
        return None

    def merge_gene_adata_labels(self,
                                set_name_list: list,
                                uniq_cell_id_obs_key: str,
                                new_labels: bool = True) -> None:
        if new_labels:
            labels_from_thresholding = np.repeat(self.default_label, self.adata.shape[0]).astype(
                object
            )
        else:
            labels_from_thresholding = self.adata.obs[self.obs_save_key].copy()

        for set_name in set_name_list:
            gene_adata = self.thrersholding_info_dict[set_name]["gene_adata"]

            cell_uniq_id_list = gene_adata.obs[uniq_cell_id_obs_key].copy()

            unique_cell_idxs, _ = get_str_idx(cell_uniq_id_list, self.adata.obs[uniq_cell_id_obs_key])

            labels_from_thresholding[unique_cell_idxs] = gene_adata.obs[self.obs_save_key]

        self.adata.obs[self.obs_save_key] = labels_from_thresholding

    def plot_linear_decision_boundaries(
        self,
        gene_adata: ad.AnnData,
        gene_x: np.ndarray,
        labels: list,
        x_lims: tuple,
        y_lims: tuple,
        resolution: int,
        cmap: plt.cm,
    ):
        # encoding the labels from strings to integers
        encoder = LabelEncoder()
        encoder.fit(gene_adata.obs[self.obs_save_key].values)

        # Reassigning the classes to the encoder
        encoder.classes_ = np.array(labels)
        encoded_labels = encoder.transform(np.array(gene_adata.obs[self.obs_save_key]))

        # prepration of data to train the SVM
        dummy_feature = np.repeat(0, gene_adata.shape[0]).T
        svm_x = np.hstack([gene_x, dummy_feature.reshape(-1, 1)])
        svc = SVC(random_state=0)
        svc.fit(svm_x, encoded_labels)

        # Creating the grid to predict the labels of the new data and stacking dummy variable
        test_x_axis = np.linspace(x_lims[0], x_lims[1], resolution)
        svm_pred_x = np.vstack([test_x_axis, np.repeat(0, resolution)]).T

        # Using the trained SVM to predict the labels of the new data
        predictions = svc.predict(svm_pred_x)

        # Reshaping the predictions to be used in the contourf function
        predictions_contourf = np.repeat(
            predictions.reshape(-1, 1), resolution, axis=1
        ).T

        # Creating the array of the y axis to be used in the contourf function
        test_y_axis = np.linspace(y_lims[0], y_lims[1], resolution)

        plt.contourf(
            test_x_axis,
            test_y_axis,
            predictions_contourf,
            alpha=0.15,
            cmap=cmap,
            zorder=2,
        )

    def plot_bayesian_information_criteria(
        self,
        set_name: str,
        bic_range: int = 5,
        unit_size: int = 5,
        ratio: tuple = (1, 1),
        return_fig: bool = False,
    ):
        gene_adata = self.thrersholding_info_dict[set_name]["gene_adata"]
        n_components = self.thrersholding_info_dict[set_name]["n_components"]

        cc_x = gene_adata.X.copy()

        bics = []
        counter = 1
        for _ in range(
            bic_range
        ):  # test the AIC/BIC metric between 1 and 10 components
            gmm = GaussianMixture(n_components=counter, **self.gaussian_mixture_model_parameters)
            gmm.fit(cc_x).predict(cc_x)
            bic = gmm.bic(cc_x)
            bics.append(bic)
            counter = counter + 1

        row_ratio, col_ratio = ratio
        fig = plt.figure(figsize=(col_ratio * unit_size, row_ratio * unit_size))

        plt.plot(np.arange(1, bic_range + 1), bics, "o-", lw=3, c="black", label="BIC")
        plt.xlabel("Number of components", fontsize=10)
        plt.ylabel("Information criterion", fontsize=10)
        plt.axvline(n_components, color="red", linestyle="--", lw=3)
        plt.xticks(np.arange(0, bic_range + 1, 1))
        plt.title("Bayesian Information Criteria Evaluation")
        plt.tight_layout()

        if return_fig:
            return fig
