from typing import Optional
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np  # noqa: E402

np.seterr(all="ignore")

import re  # noqa: E402
import anndata as ad  # noqa: E402
from tqdm import tqdm  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from sklearn import metrics  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


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
        print(
            f"Classification Report for RF model trained with {feature_set_description} feature set"
        )
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
    train_test_split_params: Optional[dict] = None,
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
        rf_params = {
            "min_samples_leaf": 50,
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
    feature_set_idxs = np.where(np.isin(adata.var_names.values, training_feature_set))[
        0
    ]

    # remove the nan values from the feature set
    # TODO: I need to make this more generalizable because there are other forms of nan in the data
    phase_nan_mask = adata.obs[training_labels].astype(str) == "nan"
    phase_nan_idx = np.where(phase_nan_mask)[0]

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
    fs_bool = np.isin(adata.var_names.values, optimum_random_forest_feature_set)

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
