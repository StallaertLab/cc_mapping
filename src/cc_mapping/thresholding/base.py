"""
Base classes and models for Gaussian Mixture Model (GMM) based thresholding.

This module provides:
- Pydantic models for storing GMM information, decision boundaries, and thresholding events
- GaussianMixtureModelBase: Base class with shared utilities for GMM operations

Classes:
    _GaussianMixtureModelInfo: Pydantic model for GMM parameters and results
    _DecisionBoundariesModel: Pydantic model for decision boundary thresholds
    _SingleThresholdingEventModel: Pydantic model for complete thresholding event data
    GaussianMixtureModelBase: Base class providing shared utilities and plotting methods
"""

import warnings
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from kneed import KneeLocator
from matplotlib import axes, gridspec
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, field_validator
from sklearn.mixture import GaussianMixture


def _validate_save_path(save_path: str | Path | None) -> Path | None:
    """
    Validate that the directory of the save_path exists.

    Parameters
    ----------
    save_path : str or Path or None
        Path where the figure should be saved. Can be str or Path.

    Returns
    -------
    Path or None
        Path object if save_path is provided, None otherwise.

    Raises
    ------
    FileNotFoundError
        If the parent directory of save_path doesn't exist.

    Examples
    --------
    >>> path = _validate_save_path('results/sequential/figure.png')
    >>> # Raises error if 'results/sequential/' doesn't exist
    """
    if save_path is None:
        return None

    save_path = Path(save_path)
    parent_dir = save_path.parent

    if not parent_dir.exists():
        raise FileNotFoundError(
            f"Directory '{parent_dir}' does not exist. "
            f"Please create it before saving the figure."
        )

    return save_path


class _GaussianMixtureModelInfo(BaseModel):
    """
    Model to store Gaussian Mixture Model (GMM) information.

    Attributes
    ----------
    gmm_kwargs : dict or None
        Keyword arguments used for the Gaussian Mixture Model.
    means : list of float or None
        Means of the GMM components.
    covs : list of float or None
        Covariances of the GMM components.
    weights : list of float or None
        Weights of the GMM components.
    n_components : int or None
        Number of GMM components.
    data_probs : list of list of float or None
        Probability of each data point belonging to each gmm component.
    condensed_data_probs : list of list of float or None
        Condensed data probabilities after handling duplicates (optional).
    """

    gmm_kwargs: dict | None = Field(
        default=None,
        description="Keyword arguments used for the Gaussian Mixture Model.",
    )
    means: list[float] | None = Field(
        default=None, description="Means of the GMM components."
    )
    covs: list[float] | None = Field(
        default=None, description="Covariances of the GMM components."
    )
    weights: list[float] | None = Field(
        default=None, description="Weights of the GMM components."
    )
    n_components: int | None = Field(
        default=None, description="Number of GMM components."
    )
    data_probs: list[list[float]] | None = Field(
        default=None,
        description="Probability of each data point belonging to each gmm component.",
    )
    condensed_data_probs: list[list[float]] | None = Field(
        default=None,
        description="Condensed data probabilities after handling duplicates (optional).",
    )

    @field_validator("*", mode="before")  # Apply validator to all fields
    @classmethod
    def convert_numpy_to_list(cls, value):
        """
        Converts NumPy arrays to lists for Pydantic validation/serialization.

        Parameters
        ----------
        value : any
            Value to convert.

        Returns
        -------
        any
            Converted value (list if input was numpy array, otherwise unchanged).
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value


class _DecisionBoundariesModel(BaseModel):
    """
    Model to store decision boundary information.

    Attributes
    ----------
    thresholds : list of float
        List of decision boundary thresholds.
    """

    thresholds: list[float] = Field(description="List of decision boundary thresholds.")


class _SingleThresholdingEventModel(BaseModel):
    """
    Model to store all information for a single thresholding event.

    Attributes
    ----------
    gmm_info : _GaussianMixtureModelInfo or None
        Gaussian Mixture Model information.
    ordered_gmm_labels : list of str or None
        Ordered labels corresponding to the gmm components.
    decision_boundaries : _DecisionBoundariesModel or None
        Decision boundary information (optional).
    condensed_labels : list of str or None
        Condensed labels after handling duplicates (optional).
    feature_name : str
        Name of the feature that was thresholded.
    gmm_obs_label : str
        Observation label in anndata object to store gmm phase labels.
    """

    gmm_info: _GaussianMixtureModelInfo | None = Field(
        description="Gaussian Mixture Model information.",
        default=None,
    )
    ordered_gmm_labels: list[str] | None = Field(
        description="Ordered labels corresponding to the gmm components.", default=None
    )
    decision_boundaries: _DecisionBoundariesModel | None = Field(
        default=None, description="Decision boundary information (optional)."
    )
    condensed_labels: list[str] | None = Field(
        default=None,
        description="Condensed labels after handling duplicates (optional).",
    )
    feature_name: str = Field(description="Name of the feature that was thresholded.")
    gmm_obs_label: str = Field(
        description="Observation label in anndata object to store gmm phase labels."
    )


class GaussianMixtureModelBase:
    """
    Base class providing shared utilities for GMM-based thresholding operations.

    This class contains utility methods and plotting functions that are common
    to both single-dataset thresholding and sequential refinement operations.
    It should not be instantiated directly; use GMMThresholding
    or SequentialGMM instead.

    The class provides:
    - Utility methods for data manipulation and label handling
    - BIC calculation and component optimization
    - Parameterized plotting methods for visualizing GMM results
    - Decision boundary calculation utilities
    """

    def _ensure_list(self, value: any) -> list:
        """
        Ensures that the input value is a list.

        Parameters
        ----------
        value : any
            Value to convert to list.

        Returns
        -------
        list
            Input value as a list.
        """
        if isinstance(value, list):
            return value
        return [value.item()] if hasattr(value, "item") else [value]

    def _handle_duplicate_labels(
        self, data_probs: np.ndarray, ordered_gmm_labels: list
    ) -> tuple:
        """
        Handles duplicate labels in the gmm results.

        Parameters
        ----------
        data_probs : np.ndarray
            Probability matrix (samples x components).
        ordered_gmm_labels : list
            List of label names for each component.

        Returns
        -------
        tuple of (np.ndarray, list)
            Condensed probability matrix and condensed label list.
        """
        df_probs = pd.DataFrame(data_probs, columns=ordered_gmm_labels)
        # Use sort=False to preserve the original order of labels
        grouped = df_probs.T.groupby(level=0, sort=False)
        condensed_data_probs = grouped.max().T.values
        condensed_labels = grouped.first().index.tolist()
        return condensed_data_probs, condensed_labels

    def _calculate_decision_boundaries_from_probs(
        self,
        feature_values: np.ndarray,
        data_probs: np.ndarray,
        ordered_labels: list[str] | None = None,
    ) -> _DecisionBoundariesModel:
        """
        Calculate decision boundaries from any probability array.

        This method works with both original GMM probabilities and condensed
        probabilities from collapsed labels. It finds class transitions and
        calculates threshold midpoints.

        Parameters
        ----------
        feature_values : np.ndarray
            Array of feature values for all samples.
        data_probs : np.ndarray
            Probability matrix (samples x components) from GMM or
            condensed from duplicate label collapsing.
        ordered_labels : list of str or None, optional
            List of label names corresponding to components.
            If provided, used to make warning messages more interpretable.

        Returns
        -------
        _DecisionBoundariesModel
            Contains calculated thresholds.

        Warns
        -----
        UserWarning
            If backward transitions are detected (likely due to outliers
            or overlapping components).
        """
        # Sort feature values and get sort indices
        sort_index = np.argsort(feature_values)
        sorted_feature_values = feature_values[sort_index]
        sorted_probabilities = data_probs[sort_index]

        # Determine the predicted class for each data point
        predicted_classes = np.argmax(sorted_probabilities, axis=1)

        # Calculate the difference between consecutive predicted classes
        class_differences = np.diff(predicted_classes)

        # Find indices where class changes occur (any non-zero change)
        change_indices = np.where(class_differences != 0)[0]

        # Filter out backward transitions (non-increasing component indices)
        # This handles outliers that cause the argmax to "jump back" to earlier components
        forward_transitions = []
        backward_transitions = []

        for idx in change_indices:
            from_class = predicted_classes[idx]
            to_class = predicted_classes[idx + 1]

            if to_class > from_class:
                # Forward transition - keep it
                forward_transitions.append(idx)
            else:
                # Backward transition - record but don't use
                transition_location = sorted_feature_values[idx]
                backward_transitions.append((from_class, to_class, transition_location))

        # Warn user if backward transitions were detected
        if backward_transitions:
            n_components = data_probs.shape[1]
            expected_thresholds = n_components - 1

            # Build detailed transition descriptions with label names if available
            transition_details = []
            for fr, to, loc in backward_transitions:
                if ordered_labels is not None:
                    from_label = ordered_labels[fr]
                    to_label = ordered_labels[to]
                    transition_details.append(
                        f"  • {from_label} (component {fr}) → {to_label} (component {to}) at x ≈ {loc:.3f}"
                    )
                else:
                    transition_details.append(
                        f"  • component {fr} → {to} at x ≈ {loc:.3f}"
                    )

            transition_str = "\n".join(transition_details)

            warnings.warn(
                f"\nDetected {len(backward_transitions)} backward transition(s) in sorted data:\n"
                f"{transition_str}\n"
                f"This suggests overlapping components or outliers.\n"
                f"These transitions are being ignored, keeping only {len(forward_transitions)} "
                f"forward transitions (expected {expected_thresholds}).\n"
                f"Review your plots and .uns metadata to verify the boundaries are appropriate.",
                UserWarning,
            )

        # Calculate midpoints at forward class transitions
        return _DecisionBoundariesModel(
            thresholds=[
                (sorted_feature_values[i] + sorted_feature_values[i + 1]) / 2
                for i in forward_transitions
            ]
        )

    def _calculate_bic_for_component_range(
        self,
        adata: ad.AnnData,
        feature: str,
        component_range: int,
        layer: str | None,
        gmm_kwargs: dict,
    ) -> list[int | float]:
        """
        Run Bayesian Information Criterion (BIC) on the gene expression data.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing the data.
        feature : str
            Name of the feature to analyze.
        component_range : int
            Maximum number of components to test (tests 1 to component_range).
        layer : str or None
            Optional layer name to use instead of .X.
        gmm_kwargs : dict
            Keyword arguments for GaussianMixture.

        Returns
        -------
        list of (int or float)
            BIC values for each number of components tested.
        """
        if not isinstance(component_range, int):
            raise ValueError("component_range must be an integer.")

        # Get feature data with layer support
        if layer is None:
            gene_x = adata[:, feature].X
        else:
            gene_x = adata[:, feature].layers[layer]

        bic_list = []
        for component_num in range(component_range):
            gmm = GaussianMixture(n_components=component_num + 1, **gmm_kwargs)
            _ = gmm.fit(gene_x).predict(gene_x)
            bic = gmm.bic(gene_x)
            bic_list.append(bic)

        return bic_list

    def determine_optimal_number_components(
        self,
        adata: ad.AnnData,
        feature: str,
        component_range: int,
        layer: str | None = None,
        gmm_kwargs: dict | None = None,
        metric: str = "bic",
        curve: str = "convex",
        direction: str = "decreasing",
        return_bic_list: bool = False,
    ) -> int | tuple[int, list[int | float]]:
        """
        Determine the optimal number of components for the GMM.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing the data.
        feature : str
            Name of the feature to analyze.
        component_range : int
            Maximum number of components to test.
        layer : str or None, optional
            Optional layer name to use instead of .X. Default is None.
        gmm_kwargs : dict or None, optional
            Keyword arguments for GaussianMixture. If None, uses defaults.
            Default is None.
        metric : str, optional
            Metric to use for optimization (currently only 'bic' supported).
            Default is 'bic'.
        curve : str, optional
            Type of curve for knee detection ('convex' or 'concave').
            Default is 'convex'.
        direction : str, optional
            Direction of curve ('decreasing' or 'increasing').
            Default is 'decreasing'.
        return_bic_list : bool, optional
            If True, returns tuple of (optimal_n, bic_list). Default is False.

        Returns
        -------
        int or tuple of (int, list of (int or float))
            Optimal number of components, or tuple of (optimal_n, bic_list)
            if return_bic_list=True.
        """
        if not isinstance(component_range, int):
            raise ValueError("component_range must be an integer.")

        if metric != "bic":
            raise ValueError("Currently, only BIC is supported.")

        if gmm_kwargs is None:
            gmm_kwargs = {
                "init_params": "k-means++",
                "n_init": 10,
                "max_iter": 1000,
                "random_state": 42,
            }

        bic_list = self._calculate_bic_for_component_range(
            adata, feature, component_range, layer, gmm_kwargs
        )
        optimal_component_number = KneeLocator(
            x=range(1, component_range + 1),
            y=bic_list,
            curve=curve,
            direction=direction,
        )
        if return_bic_list:
            return optimal_component_number.knee, bic_list
        else:
            return optimal_component_number.knee

    def plot_bayesian_information_criterion_curve(
        self,
        adata: ad.AnnData,
        feature: str,
        component_range: int,
        layer: str | None = None,
        gmm_kwargs: dict | None = None,
        curve: str = "convex",
        direction: str = "decreasing",
        ax: plt.Axes = None,
        save_path: str | Path | None = None,
    ) -> None:
        """
        Plot the BIC curve.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing the data.
        feature : str
            Name of the feature to analyze.
        component_range : int
            Maximum number of components to test.
        layer : str or None, optional
            Optional layer name to use instead of .X. Default is None.
        gmm_kwargs : dict or None, optional
            Keyword arguments for GaussianMixture. If None, uses defaults.
            Default is None.
        curve : str, optional
            Type of curve for knee detection ('convex' or 'concave').
            Default is 'convex'.
        direction : str, optional
            Direction of curve ('decreasing' or 'increasing').
            Default is 'decreasing'.
        ax : matplotlib.pyplot.Axes or None, optional
            Optional matplotlib axes to plot on. If None, creates new figure.
            Default is None.
        save_path : str or Path or None, optional
            Optional path to save the figure. Parent directory must exist.
            Default is None.

        Raises
        ------
        FileNotFoundError
            If save_path parent directory doesn't exist.
        """
        if not isinstance(component_range, int):
            raise ValueError("component_range must be an integer.")

        # Validate save path before generating the figure
        save_path = _validate_save_path(save_path)

        optimal_component_number, bic_list = self.determine_optimal_number_components(
            adata,
            feature,
            component_range,
            layer,
            gmm_kwargs,
            curve=curve,
            direction=direction,
            return_bic_list=True,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        ax.plot(range(1, component_range + 1), bic_list, marker="o", color="black")
        ax.axvline(
            optimal_component_number,
            color="red",
            linestyle="--",
            label=f"Optimal Number of Components = {optimal_component_number}",
        )

        ax.set_title("Bayesian Information Criterion (BIC) Curve")
        ax.set_xticks(np.arange(1, component_range + 1, 1))
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("BIC Value")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Figure saved to: {save_path}")

    def _plot_gmm_components(
        self,
        ax: axes.Axes,
        adata: ad.AnnData,
        feature: str,
        internal_data: _SingleThresholdingEventModel,
        num_std: int,
        resolution: int,
        cmap: plt.cm.ScalarMappable,
    ) -> axes.Axes:
        """
        Plot the Gaussian probability density functions (PDFs) and their means.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        adata : ad.AnnData
            AnnData object (not currently used but kept for consistency).
        feature : str
            Name of the feature being plotted.
        internal_data : _SingleThresholdingEventModel
            Thresholding event model containing GMM info and boundaries.
        num_std : int
            Number of standard deviations to plot for each component.
        resolution : int
            Number of points to use for plotting each Gaussian curve.
        cmap : matplotlib.pyplot.cm.ScalarMappable
            Colormap for coloring the categories.

        Returns
        -------
        matplotlib.axes.Axes
            Modified axes object with GMM components plotted.

        Raises
        ------
        ValueError
            If decision boundaries have not been calculated.
        """
        if internal_data.decision_boundaries is None:
            raise ValueError(
                "Decision boundaries have not been calculated. Please call calculate_decision_boundaries() first."
            )

        threshold_list = internal_data.decision_boundaries.thresholds

        means: list = self._ensure_list(internal_data.gmm_info.means)
        covs: list = self._ensure_list(internal_data.gmm_info.covs)
        weights: list = self._ensure_list(internal_data.gmm_info.weights)
        ordered_labels = internal_data.ordered_gmm_labels

        num_components = len(means)

        # Determine number of final categories for color mapping
        if internal_data.condensed_labels is not None:
            num_final_categories = len(internal_data.condensed_labels)
        elif ordered_labels is not None:
            num_final_categories = len(set(ordered_labels))
        else:
            num_final_categories = len(threshold_list) + 1

        # Use final category colors for background and mean lines
        category_colors = cmap(np.linspace(0, 1, num_final_categories))

        for gaus_idx in range(num_components):
            g_mean = means[gaus_idx]
            g_cov = covs[gaus_idx]
            g_weight = weights[gaus_idx]

            std = np.sqrt(g_cov)
            x_min = g_mean - num_std * std
            x_max = g_mean + num_std * std
            x_axis = np.linspace(x_min, x_max, resolution)
            y_axis = st.norm.pdf(x_axis, loc=g_mean, scale=std) * g_weight

            # Color the GMM curve segments by which category they fall into
            bin_indices = np.digitize(x_axis, threshold_list)
            bin_indices = np.clip(bin_indices, 0, num_final_categories - 1)
            x_colors = category_colors[bin_indices]

            ax.scatter(x_axis, y_axis, lw=1, c=x_colors, zorder=3, s=1)

            # Color the component mean line by spatial position (which region it falls in)
            # This ensures mean lines match the background shaded regions
            mean_bin_idx = np.digitize([g_mean], threshold_list)[0]
            mean_bin_idx = np.clip(mean_bin_idx, 0, num_final_categories - 1)
            mean_line_color = category_colors[mean_bin_idx]

            ax.axvline(g_mean, c=mean_line_color, lw=2, ls="--", zorder=4)

        return ax

    def _plot_sample_catergory_legend(
        self,
        ax: axes.Axes,
        internal_data: _SingleThresholdingEventModel,
        cmap: plt.cm.ScalarMappable,
        legend_kwargs: dict | None = None,
    ) -> axes.Axes:
        """
        Plot legend showing category labels and colors.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to add legend to.
        internal_data : _SingleThresholdingEventModel
            Thresholding event model containing label information.
        cmap : matplotlib.pyplot.cm.ScalarMappable
            Colormap used for category colors.
        legend_kwargs : dict or None, optional
            Optional kwargs to pass to ax.legend(). Default is None.

        Returns
        -------
        matplotlib.axes.Axes
            Modified axes object with legend added.
        """
        if internal_data.condensed_labels is not None:
            ordered_labels = internal_data.condensed_labels
        else:
            ordered_labels = internal_data.ordered_gmm_labels
        patch_list = []

        component_colors = cmap(np.linspace(0, 1, len(ordered_labels)))
        for label, color in zip(ordered_labels, component_colors):
            patch = mpatches.Patch(
                facecolor=color, label=label, alpha=0.8, edgecolor="black"
            )
            patch_list.append(patch)

        if legend_kwargs is None:
            legend_kwargs = {}

        ax.legend(handles=patch_list, **legend_kwargs)

        return ax

    def _plot_hist_base(
        self,
        adata: ad.AnnData,
        feature: str,
        layer: str | None,
        hist_kwargs: dict | None = None,
        ax: plt.Axes = None,
        x_axis_limits: tuple | None = None,
    ) -> axes.Axes:
        """
        Base function for plotting GMM distributions.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing the data.
        feature : str
            Name of the feature to plot.
        layer : str or None
            Optional layer name to use instead of .X.
        hist_kwargs : dict or None, optional
            Optional kwargs to pass to ax.hist(). Default is None.
        ax : matplotlib.pyplot.Axes or None, optional
            Optional matplotlib axes. If None, creates new figure. Default is None.
        x_axis_limits : tuple or None, optional
            Optional tuple of (x_min, x_max) for plot limits. Default is None.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with histogram plotted.
        """
        # Get feature data with layer support
        if layer is None:
            gene_x = adata[:, feature].X.copy()
        else:
            gene_x = adata[:, feature].layers[layer].copy()

        if gene_x.shape[1] != 1:
            raise ValueError("plot_gmm_distribution is only for 1d data.")

        if x_axis_limits is not None:
            x_lower_lim, x_upper_lim = x_axis_limits
            # Handle None values in limits by using data min/max
            if x_lower_lim is None:
                x_lower_lim = np.min(gene_x)
            if x_upper_lim is None:
                x_upper_lim = np.max(gene_x)
            # Validate limits after handling None values
            if x_lower_lim >= x_upper_lim:
                raise ValueError(
                    f"{x_lower_lim} must be less than {x_upper_lim} in x_lims."
                )
        else:
            x_lower_lim = np.min(gene_x)
            x_upper_lim = np.max(gene_x)

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        if hist_kwargs is None:
            hist_kwargs = {"bins": 100, "color": "black"}

        if "density" in hist_kwargs:
            hist_kwargs.pop("density")
            print("density is set to True by default and cannot be altered.")

        ax.hist(gene_x, density=True, zorder=2, **hist_kwargs)

        ax.set_xlim(x_lower_lim, x_upper_lim)

        return ax

    def _plot_vertical_linear_decision_boundaries(
        self,
        ax: axes.Axes,
        internal_data: _SingleThresholdingEventModel,
        resolution: int,
        cmap: plt.cm.ScalarMappable,
    ) -> axes.Axes:
        """
        Plots decision boundaries as a background with category colors.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        internal_data : _SingleThresholdingEventModel
            Thresholding event model containing boundaries and labels.
        resolution : int
            Number of points for smooth color gradients.
        cmap : matplotlib.pyplot.cm.ScalarMappable
            Colormap for category colors.

        Returns
        -------
        matplotlib.axes.Axes
            Modified axes object with decision boundaries plotted.

        Raises
        ------
        ValueError
            If decision boundaries have not been calculated.
        """
        if internal_data.decision_boundaries is None:
            raise ValueError(
                "Decision boundaries have not been calculated. Please call calculate_decision_boundaries() first."
            )

        thresholds = internal_data.decision_boundaries.thresholds

        # Determine number of final categories based on unique labels
        if internal_data.condensed_labels is not None:
            num_final_categories = len(internal_data.condensed_labels)
        elif internal_data.ordered_gmm_labels is not None:
            num_final_categories = len(set(internal_data.ordered_gmm_labels))
        else:
            num_final_categories = len(thresholds) + 1

        component_colors = cmap(np.linspace(0, 1, num_final_categories))

        x_min, x_max = ax.get_xlim()
        _, y_max = ax.get_ylim()
        x_axis = np.linspace(x_min, x_max, resolution)

        bin_indices = np.digitize(x_axis, thresholds)

        # Clamp bin indices to valid color range [0, num_final_categories-1]
        # This handles cases where we have more thresholds than final categories
        bin_indices = np.clip(bin_indices, 0, num_final_categories - 1)

        x_axis_colors_rgba = np.array(component_colors)[bin_indices]
        rgba_decision_boundary_grid = np.repeat(
            x_axis_colors_rgba[np.newaxis, :, :], 2, axis=0
        )

        ax.imshow(
            rgba_decision_boundary_grid,
            extent=[x_min, x_max, 0, y_max],
            origin="lower",
            aspect="auto",
            alpha=0.2,
        )

        for idx, threshold in enumerate(thresholds):
            # Clamp idx+1 to avoid index out of bounds when we have more thresholds than colors
            color_idx_right = min(idx + 1, num_final_categories - 1)
            axvline_color = (
                component_colors[idx] + component_colors[color_idx_right]
            ) / 2
            ax.axvline(threshold, color=axvline_color, zorder=5)

        return ax

    def _plot_horizontal_linear_decision_boundaries(
        self,
        ax: axes.Axes,
        internal_data: _SingleThresholdingEventModel,
        resolution: int,
        cmap: plt.cm.ScalarMappable,
    ) -> axes.Axes:
        """
        Plots horizontal decision boundaries with category colors.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        internal_data : _SingleThresholdingEventModel
            Thresholding event model containing boundaries and labels.
        resolution : int
            Number of points for smooth color gradients.
        cmap : matplotlib.pyplot.cm.ScalarMappable
            Colormap for category colors.

        Returns
        -------
        matplotlib.axes.Axes
            Modified axes object with horizontal decision boundaries plotted.
        """
        thresholds = internal_data.decision_boundaries.thresholds

        # Determine number of final categories based on unique labels
        if internal_data.condensed_labels is not None:
            num_final_categories = len(internal_data.condensed_labels)
        elif internal_data.ordered_gmm_labels is not None:
            num_final_categories = len(set(internal_data.ordered_gmm_labels))
        else:
            num_final_categories = len(thresholds) + 1

        component_colors = cmap(np.linspace(0, 1, num_final_categories))
        thresholds = sorted(thresholds)  # Ensure sorted

        # Get plot limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Create a grid for the Y-axis
        y_axis_grid = np.linspace(y_min, y_max, resolution)

        # Digitize based on Y-thresholds
        bin_indices = np.digitize(y_axis_grid, thresholds)

        # Clamp bin indices to valid color range [0, num_final_categories-1]
        # This handles cases where we have more thresholds than final categories
        bin_indices = np.clip(bin_indices, 0, num_final_categories - 1)

        # Get colors for the Y-axis grid
        y_axis_colors_rgba = np.array(component_colors)[bin_indices]

        color_grid = np.repeat(y_axis_colors_rgba[:, np.newaxis, :], 2, axis=1)

        ax.imshow(
            color_grid,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            aspect="auto",
            alpha=0.2,
        )

        # Plot horizontal lines at thresholds
        for idx, threshold in enumerate(thresholds):
            # Clamp idx+1 to avoid index out of bounds when we have more thresholds than colors
            color_idx_right = min(idx + 1, num_final_categories - 1)
            axhline_color = (
                component_colors[idx] + component_colors[color_idx_right]
            ) / 2
            ax.axhline(threshold, color=axhline_color, zorder=5)

        return ax

    def _plot_strip_plot_base(
        self,
        adata: ad.AnnData,
        feature: str,
        layer: str | None,
        obs_label: str | None = None,
        ordered_labels: list[str] | None = None,
        scatter_density: bool = True,
        y_axis_limits: tuple[float, float] | None = None,
        hist_kwargs: dict | None = None,
        strip_plot_kwargs: dict | None = None,
        cmap: mpl.cm.ScalarMappable = mpl.colormaps["plasma"],
        vmax: float | None = None,
    ) -> tuple[Figure, axes.Axes, axes.Axes]:
        """
        Base method for creating strip plot + histogram WITHOUT decision boundaries.

        This method handles the core plotting logic for strip plots with histograms,
        but does not add decision boundaries or legends. It's designed to be called
        by both exploratory methods (before thresholding) and final visualization
        methods (with thresholding).

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing the data.
        feature : str
            Feature name to plot.
        layer : str or None
            Layer to use for data access. If None, uses adata.X.
        obs_label : str or None, optional
            Optional column in adata.obs containing categorical labels.
            Required when scatter_density=False to color by labels.
        ordered_labels : list of str or None, optional
            Optional ordered list of label names (low to high).
            Required when scatter_density=False for label-colored scatter.
        scatter_density : bool, optional
            If True, color by density; if False, color by labels. Default is True.
        y_axis_limits : tuple of float or None, optional
            Y-axis limits (min, max). If None, uses data min/max. Default is None.
        hist_kwargs : dict or None, optional
            Kwargs for histogram. Default is None.
        strip_plot_kwargs : dict or None, optional
            Kwargs for strip plot scatter (e.g., s, alpha, marker).
            Only used when scatter_density=False. Default is None.
        cmap : matplotlib.pyplot.cm.ScalarMappable, optional
            Colormap for density or label colors. Default is 'plasma'.
        vmax : int or float or None, optional
            Maximum density value for colormap. If None, auto-calculated. Default is None.

        Returns
        -------
        tuple of (Figure, matplotlib.axes.Axes, matplotlib.axes.Axes)
            Figure and the two axes objects (scatter plot and histogram).

        Raises
        ------
        ValueError
            If scatter_density=False but obs_label or ordered_labels not provided.

        Warns
        -----
        UserWarning
            If vmax is provided when scatter_density=False.
        """
        # Get feature data
        if layer is None:
            feature_array = adata[:, feature].X.copy().squeeze()
        else:
            feature_array = adata[:, feature].layers[layer].copy().squeeze()

        # Validate inputs
        if not scatter_density:
            if obs_label is None or ordered_labels is None:
                raise ValueError(
                    "When scatter_density=False, both obs_label and ordered_labels must be provided."
                )
            if vmax is not None:
                warnings.warn(
                    "vmax is only used when scatter_density is set to True. Ignoring vmax.",
                    UserWarning,
                )

        if y_axis_limits is None:
            y_axis_limits = (np.min(feature_array), np.max(feature_array))

        # Auto-calculate vmax if needed for density mode
        if scatter_density and vmax is None:
            filtered_feature_array = feature_array[feature_array < y_axis_limits[1]]
            filtered_x = np.random.uniform(0, 1, size=filtered_feature_array.shape[0])
            temp_fig, density_ax = plt.subplots(
                figsize=(5, 5), subplot_kw={"projection": "scatter_density"}
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                density_object = density_ax.scatter_density(
                    filtered_x,
                    filtered_feature_array,
                    cmap=cmap,
                    dpi=100,
                    vmin=1,
                    zorder=1,
                    alpha=0.0,
                )

            temp_fig.canvas.draw()
            vmax = density_object.get_array().max()

            if len(feature_array) / vmax > 5000:
                print("Warning: The maximum density value is less than 1.")
                print(
                    "This may indicate that the y_axis limits are not set properly or your data is sparse."
                )

            plt.close(temp_fig)  # Close the temporary figure

        # Create figure with proper grid for colorbar
        fig = plt.figure(figsize=(6, 6))
        if scatter_density:
            # Create 3-column grid: strip plot, histogram, colorbar space
            # Increased colorbar width from 0.15 to 0.25 and spacing from 0.05 to 0.15
            gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 0.25], wspace=0.15)
        else:
            # Create 2-column grid: strip plot, histogram (no colorbar needed)
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0)

        # Add subplots
        ax_scatter = fig.add_subplot(gs[0, 0], projection="scatter_density")
        ax_hist_y = fig.add_subplot(gs[0, 1], sharey=ax_scatter)

        # Generate random x-coordinates for strip plot
        x = np.random.uniform(0, 1, size=feature_array.shape[0])

        cmap.set_under("white", alpha=1.0)

        # Plot scatter (density or label-colored)
        if scatter_density:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                density = ax_scatter.scatter_density(
                    x, feature_array, cmap=cmap, dpi=100, vmin=1, vmax=vmax, zorder=1
                )
        else:
            component_colors = mpl.colormaps["rainbow"](
                np.linspace(0, 1, len(ordered_labels))
            )

            sample_labels = adata.obs[obs_label].values
            sample_label_color_mapping = {
                label: color for label, color in zip(ordered_labels, component_colors)
            }
            sample_colors = np.array(
                [sample_label_color_mapping[label] for label in sample_labels]
            )

            # Set default strip plot kwargs
            if strip_plot_kwargs is None:
                strip_plot_kwargs = {"s": 5}

            # Validate strip_plot_kwargs
            if "c" in strip_plot_kwargs or "color" in strip_plot_kwargs:
                warnings.warn(
                    "Color parameters in strip_plot_kwargs will be ignored. Colors are determined by labels.",
                    UserWarning,
                )
                strip_plot_kwargs.pop("c", None)
                strip_plot_kwargs.pop("color", None)

            if "zorder" in strip_plot_kwargs:
                warnings.warn(
                    "zorder in strip_plot_kwargs will be ignored. zorder is set to 1 by default.",
                    UserWarning,
                )
                strip_plot_kwargs.pop("zorder", None)

            density = ax_scatter.scatter(
                x, feature_array, c=sample_colors, zorder=1, **strip_plot_kwargs
            )

        # Style scatter plot
        ax_scatter.tick_params(
            axis="x",
            which="both",
            right=False,
            labelright=False,
            bottom=False,
            labelbottom=False,
        )
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["bottom"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)
        ax_scatter.spines["left"].set_visible(False)
        ax_scatter.set_ylabel(f"{feature}")
        ax_scatter.set_ylim(y_axis_limits)

        # Plot histogram
        if hist_kwargs is None:
            hist_kwargs = {"bins": 1000, "color": "black"}

        if "orientation" in hist_kwargs:
            hist_kwargs.pop("orientation")
            print("orientation is set to horizontal by default and cannot be altered.")

        if "density" in hist_kwargs:
            hist_kwargs.pop("density")
            print("density is set to True by default and cannot be altered.")

        ax_hist_y.hist(
            feature_array, orientation="horizontal", density=True, **hist_kwargs
        )
        ax_hist_y.axis("off")

        # Add colorbar for density mode in dedicated third column
        if scatter_density:
            cax = fig.add_subplot(gs[0, 2])
            fig.colorbar(density, cax=cax, label="Sample Density", extend="min")

        return fig, ax_scatter, ax_hist_y

    def _plot_strip_plot_histogram_with_decision_boundaries(
        self,
        adata: ad.AnnData,
        feature: str,
        layer: str | None,
        obs_label: str,
        ordered_labels: list[str],
        internal_data: _SingleThresholdingEventModel,
        cmap: mpl.cm.ScalarMappable = mpl.colormaps["plasma"],
        y_axis_limits: tuple[float, float] | None = None,
        resolution: int = 1000,
        scatter_density: bool = True,
        vmax: float | None = None,
        hist_kwargs: dict | None = None,
        strip_plot_kwargs: dict | None = None,
        title: str | None = None,
    ) -> Figure:
        """
        Generate a 1D strip plot with histogram and decision boundaries.

        This is the base implementation used by both single and sequential classes.
        Creates a figure with a scatter density strip plot (or label-colored scatter)
        alongside a horizontal histogram showing the distribution and decision boundaries.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing the data.
        feature : str
            Feature name to plot.
        layer : str or None
            Layer to use for data access. If None, uses adata.X.
        obs_label : str
            Column in adata.obs containing categorical labels.
        ordered_labels : list of str
            Ordered list of label names (low to high).
        internal_data : _SingleThresholdingEventModel
            Thresholding event model with decision boundaries and GMM info.
        cmap : matplotlib.pyplot.cm.ScalarMappable, optional
            Colormap for density or label colors. Default is 'plasma'.
        y_axis_limits : tuple of float or None, optional
            Y-axis limits (min, max). If None, uses data min/max. Default is None.
        resolution : int, optional
            Resolution for boundary plotting. Default is 1000.
        scatter_density : bool, optional
            If True, color by density; if False, color by labels. Default is True.
        vmax : int or float or None, optional
            Maximum density value for colormap. If None, auto-calculated. Default is None.
        hist_kwargs : dict or None, optional
            Kwargs for histogram. Default is None.
        strip_plot_kwargs : dict or None, optional
            Kwargs for strip plot scatter (e.g., s, alpha, marker).
            Only used when scatter_density=False. Default is None.
        title : str or None, optional
            Title for the plot. If not provided, defaults to feature name.
            Pass empty string '' to suppress title. Default is None.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.
        """
        # Call base method to create strip plot and histogram
        fig, ax_scatter, ax_hist_y = self._plot_strip_plot_base(
            adata=adata,
            feature=feature,
            layer=layer,
            obs_label=obs_label if not scatter_density else None,
            ordered_labels=ordered_labels if not scatter_density else None,
            scatter_density=scatter_density,
            y_axis_limits=y_axis_limits,
            hist_kwargs=hist_kwargs,
            strip_plot_kwargs=strip_plot_kwargs,
            cmap=cmap,
            vmax=vmax,
        )

        # Add decision boundaries
        ax_hist_y = self._plot_horizontal_linear_decision_boundaries(
            ax=ax_hist_y,
            internal_data=internal_data,
            resolution=resolution,
            cmap=mpl.colormaps["rainbow"],
        )

        # Add legend
        ax_hist_y = self._plot_sample_catergory_legend(
            ax=ax_hist_y,
            internal_data=internal_data,
            cmap=mpl.colormaps["rainbow"],
            legend_kwargs={"loc": "upper center", "fontsize": 9},
        )

        # Add title (default to feature name, allow override or suppression)
        if title is None:
            title = feature
        if title:  # Only add title if not empty string
            plt.suptitle(title)

        return fig

    def generate_thresholding_report(
        self,
        output_format: str = "text",
    ) -> str | pd.DataFrame:
        """
        Generate a human-readable report of all thresholding operations.

        Reads thresholding metadata from adata.uns and creates a summary showing:
        - Operation names and order
        - Features used
        - Number of components
        - Thresholds calculated
        - Labels assigned
        - Parent operations (for refinements)
        - Cell counts per category (captured at operation time)

        Note: Cell counts reflect the state immediately after each operation was performed,
        not the current state of the data. This is important because subsequent refinement
        operations may change labels, but the historical counts are preserved.

        Parameters
        ----------
        output_format : str, default 'text'
            'text' for formatted string, 'dataframe' for pandas DataFrame.

        Returns
        -------
        Union[str, pd.DataFrame]
            Formatted report string or DataFrame.

        Raises
        ------
        KeyError
            If thresholding_events_key doesn't exist in adata.uns.
        ValueError
            If output_format is not 'text' or 'dataframe'.
        TypeError
            If adata.uns[thresholding_events_key] is not a dict.

        Examples
        --------
        Generate text report::

            >>> gmm = GMMThresholding(adata, feature='gene1', label_obs_save_str='gene1_cat')
            >>> gmm.fit(n_components=2)
            >>> gmm.categorize_samples(['Low', 'High'])
            >>> report = gmm.generate_thresholding_report()
            >>> print(report)
            Thresholding Report
            ==================================================

            1. gene1_thresholding (Standard Thresholding)
            ----------------------------------------------
               Feature: gene1
               Layer: None
               Obs column: gene1_cat
               Components: 2
               Thresholds: [0.0450]
               Labels: ['Low', 'High']
               Cell counts: Low=1234, High=5678

        Generate DataFrame report::

            >>> report_df = gmm.generate_thresholding_report(
            ...     output_format='dataframe'
            ... )
            >>> report_df.head()
        """
        from collections import OrderedDict

        # Use the instance's thresholding_events_key
        thresholding_events_key = self.thresholding_events_key

        # Validate thresholding_events_key exists
        if thresholding_events_key not in self.adata.uns:
            raise KeyError(
                f"thresholding_events_key '{thresholding_events_key}' not found in adata.uns. "
                f"Available keys: {list(self.adata.uns.keys())}"
            )

        # Validate it's a dict-like structure
        events = self.adata.uns[thresholding_events_key]
        if not isinstance(events, (dict, OrderedDict)):
            raise TypeError(
                f"adata.uns['{thresholding_events_key}'] must be a dict or OrderedDict, "
                f"got {type(events)}"
            )

        # Validate output_format
        valid_formats = ["text", "dataframe"]
        if output_format not in valid_formats:
            raise ValueError(
                f"output_format must be one of {valid_formats}, got '{output_format}'"
            )

        if len(events) == 0:
            if output_format == "text":
                return "No thresholding operations found."
            else:
                return pd.DataFrame()

        # Build report data
        report_data = []

        for idx, (op_name, op_data) in enumerate(events.items(), 1):
            # Extract basic info
            feature = op_data.get("feature_name", "N/A")
            layer = op_data.get("layer", None)
            obs_label = op_data.get("gmm_obs_label", "N/A")
            ordered_labels = op_data.get("ordered_gmm_labels", [])

            # Extract GMM info
            gmm_info = op_data.get("gmm_info", {})
            if gmm_info is not None:
                n_components = gmm_info.get("n_components", "N/A")
            else:
                n_components = "N/A (manual thresholds)"

            # Extract thresholds
            decision_boundaries = op_data.get("decision_boundaries", {})
            thresholds = (
                decision_boundaries.get("thresholds", []) if decision_boundaries else []
            )

            # Extract operation type and hierarchy
            operation_type = op_data.get("operation_type", "standard")
            parent_operation = op_data.get("parent_operation", None)
            refined_from_labels = op_data.get("refined_from_labels", None)

            # Get cell counts - prefer stored counts from operation time
            cell_counts = op_data.get("cell_counts_after_operation", {})

            # Fallback to current obs counts if not stored (backward compatibility)
            if not cell_counts and obs_label in self.adata.obs.columns:
                counts = self.adata.obs[obs_label].value_counts()
                # Only include labels from this operation
                for label in ordered_labels:
                    if label in counts.index:
                        cell_counts[label] = int(counts[label])

            # Store data for this operation
            op_info = {
                "operation_number": idx,
                "operation_name": op_name,
                "operation_type": operation_type,
                "feature": feature,
                "layer": str(layer),
                "obs_label": obs_label,
                "n_components": n_components,
                "thresholds": thresholds,
                "labels": ordered_labels,
                "parent_operation": parent_operation,
                "refined_from_labels": refined_from_labels,
                "cell_counts": cell_counts,
            }
            report_data.append(op_info)

        # Generate output based on format
        if output_format == "dataframe":
            # Create DataFrame
            df_data = []
            for op in report_data:
                df_data.append(
                    {
                        "Operation": f"{op['operation_number']}. {op['operation_name']}",
                        "Type": op["operation_type"],
                        "Feature": op["feature"],
                        "Layer": op["layer"],
                        "Obs Label": op["obs_label"],
                        "Components": str(op["n_components"]),
                        "Thresholds": ", ".join(f"{t:.4f}" for t in op["thresholds"])
                        if op["thresholds"]
                        else "N/A",
                        "Labels": ", ".join(op["labels"]),
                        "Parent": str(op["parent_operation"]),
                        "Refined From": ", ".join(op["refined_from_labels"])
                        if op["refined_from_labels"]
                        else "N/A",
                        "Total Cells": sum(op["cell_counts"].values())
                        if op["cell_counts"]
                        else "N/A",
                    }
                )
            return pd.DataFrame(df_data)

        else:  # text format
            lines = []
            lines.append("Thresholding Report")
            lines.append("=" * 50)
            lines.append("")

            for op in report_data:
                # Header
                if (
                    op["operation_type"] == "refinement"
                    or op["operation_type"] == "refinement_manual"
                ):
                    header = (
                        f"{op['operation_number']}. {op['operation_name']} (Refinement)"
                    )
                    if op["parent_operation"]:
                        header += f" of {op['parent_operation']}"
                else:
                    header = f"{op['operation_number']}. {op['operation_name']} (Standard Thresholding)"

                lines.append(header)
                lines.append("-" * len(header))

                # Basic info
                lines.append(f"   Feature: {op['feature']}")
                lines.append(f"   Layer: {op['layer']}")
                lines.append(f"   Obs column: {op['obs_label']}")

                # GMM info
                lines.append(f"   Components: {op['n_components']}")

                # Thresholds
                if op["thresholds"]:
                    threshold_str = ", ".join(f"{t:.4f}" for t in op["thresholds"])
                    lines.append(f"   Thresholds: [{threshold_str}]")
                else:
                    lines.append("   Thresholds: None")

                # Labels
                labels_str = ", ".join(f"'{label}'" for label in op["labels"])
                lines.append(f"   Labels: [{labels_str}]")

                # Refinement-specific info
                if op["refined_from_labels"]:
                    refined_str = ", ".join(
                        f"'{label}'" for label in op["refined_from_labels"]
                    )
                    lines.append(f"   Refined from: [{refined_str}]")

                # Cell counts
                if op["cell_counts"]:
                    count_strs = [
                        f"{label}={count}" for label, count in op["cell_counts"].items()
                    ]
                    lines.append(f"   Cell counts: {', '.join(count_strs)}")
                else:
                    lines.append(
                        "   Cell counts: Not available (obs column may have been modified)"
                    )

                lines.append("")

            # Summary
            lines.append("=" * 50)
            lines.append(f"Total operations: {len(report_data)}")

            # Count operation types
            type_counts = {}
            for op in report_data:
                op_type = op["operation_type"]
                type_counts[op_type] = type_counts.get(op_type, 0) + 1

            if type_counts:
                lines.append("Operation types:")
                for op_type, count in type_counts.items():
                    lines.append(f"  - {op_type}: {count}")

            return "\n".join(lines)
