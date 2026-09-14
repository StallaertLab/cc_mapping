from typing import Optional
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np  # noqa: E402

np.seterr(all="ignore")

import re  # noqa: E402
import anndata as ad  # noqa: E402

from sklearn import metrics  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from .feature_selection import (  # noqa: E402
    RFMinMaxSelector,
    RFTopNSelector,
    prepare_feature_matrix,
)


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

    .. deprecated:: 0.2.6
        Use :func:`cc_mapping.feature_selection.train_rf_model`. This function
        will be removed in 0.3.0.

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
    # DeprecationWarning, not FutureWarning: this module ignores FutureWarning.
    warnings.warn(
        "train_random_forest_model is deprecated and will be removed in cc-mapping "
        "0.3.0. Use cc_mapping.feature_selection.train_rf_model instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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


def random_forest_feature_selection(
    adata: ad.AnnData,
    training_feature_set: list[str],
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
    show: bool = True,
) -> ad.AnnData:
    """
    Trains a random forest classifier on the training feature set and labels using one of two methods:
    RF_min_30: Selects the top 30 features based on the random forest feature importance
    RF_min_max: Selects the minimum number of features that maximizes the accuracy of the random forest classifier
    This is done by iteratively adding features to the feature set until the accuracy of the classifier
    does not improve for x number of iterations

    .. deprecated:: 0.2.6
        Use :class:`~cc_mapping.feature_selection.RFMinMaxSelector` for
        ``method="RF_min_max"`` and :class:`~cc_mapping.feature_selection.RFTopNSelector`
        for ``method="RF_min_<N>"``. This function now wraps them and will be
        removed in 0.3.0; :mod:`cc_mapping.feature_selection` shows how the
        arguments map.

    Args:
        adata (ad.AnnData): The AnnData object containing the data.
        training_feature_set (list[str]): Names of the features (entries of adata.var_names) to train on. Order does not matter.
        training_labels (str): The name of the labels to be used for training.
        feature_set_name (str, optional): The name of the feature set to be added to the .var attribute of the adata object. Defaults to None.
        method (str, optional): The method to be used for feature selection. Defaults to 'RF_min_max'.
        random_state (int, optional): The random state for reproducibility. Defaults to 42.
        threshold (float, optional): The threshold for determining when to stop adding features. Defaults to 0.01.
        stable_counter (int, optional): The number of stable iterations before stopping. Defaults to 3.
        plot (bool, optional): Whether to plot the accuracy vs. number of features graph (RF_min_max only). Defaults to True.
        save_path (str, optional): Path to save the accuracy plot to. Defaults to None.
        cutoff_method (str, optional): The method for determining when to stop adding features. Defaults to 'increment'.
        train_test_split_params (dict, optional): The parameters for train test split. Defaults to {'test_size':0.25}.
        rf_params (dict, optional): The parameters for the random forest classifier. Defaults to {'min_samples_leaf':50, 'n_estimators':150, 'bootstrap':True, 'oob_score':True, 'n_jobs':-1}.
        show (bool, optional): Whether to display the accuracy plot with plt.show(). If False, the plot is closed after it is saved. Defaults to True.

    Returns:
        ad.AnnData: The adata object with the feature set added to the .var attribute.

    Raises:
        ValueError: If any name in training_feature_set is not in adata.var_names,
            or if method is neither 'RF_min_max' nor 'RF_min_<N>'.
    """
    # DeprecationWarning, not FutureWarning: this module ignores FutureWarning.
    warnings.warn(
        "random_forest_feature_selection is deprecated and will be removed in "
        "cc-mapping 0.3.0. Use RFMinMaxSelector (method='RF_min_max') or "
        "RFTopNSelector (method='RF_min_<N>') from cc_mapping.feature_selection.",
        DeprecationWarning,
        stacklevel=2,
    )

    top_n = re.search("(?<=RF_min_)[0-9]+", method)
    if top_n is None and method != "RF_min_max":
        raise ValueError(
            f"Unknown method {method!r}; use 'RF_min_max' or 'RF_min_<N>' (e.g. 'RF_min_30')."
        )

    if feature_set_name is None:
        feature_set_name = f"{method}_feature_set"

    prepared = prepare_feature_matrix(
        adata, training_feature_set, training_labels, verbose=verbose
    )
    shared = dict(
        rf_params=rf_params,
        train_test_split_params=train_test_split_params,
        random_state=random_state,
        verbose=verbose,
    )
    if top_n:
        # RFTopNSelector raises when N exceeds the number of features; this
        # function has always kept every feature instead.
        n_features = min(int(top_n.group(0)), prepared.n_features)
        selector = RFTopNSelector(n_features=n_features, **shared)
    else:
        selector = RFMinMaxSelector(
            threshold=threshold,
            stable_iterations=stable_counter,
            cutoff_method=cutoff_method,
            **shared,
        )
    selector.fit(prepared.X, prepared.y, prepared.feature_names)
    selector.transform_adata(adata, var_key=feature_set_name)

    if plot and not top_n:
        selector.plot_accuracy_curve(save_path=save_path, show=show)

    return adata
