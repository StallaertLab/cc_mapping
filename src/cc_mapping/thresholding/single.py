"""Single-feature GMM thresholding implementation.

This module provides the GMMThresholding class for performing
GMM-based thresholding on single features within AnnData objects.

Classes
-------
GMMThresholding
    Main class for single-feature GMM thresholding
"""

import warnings
import numbers
from string import ascii_uppercase
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from pathlib import Path

import anndata as ad
from kneed import KneeLocator
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from .base import (
    _GaussianMixtureModelInfo,
    _DecisionBoundariesModel,
    _SingleThresholdingEventModel,
    GaussianMixtureModelBase,
)


class GMMThresholding(GaussianMixtureModelBase):
    """A class to perform Gaussian Mixture Model (GMM) based thresholding on single-feature data.

    This class fits a GMM to the distribution of a specified feature (e.g., gene
    expression), calculates decision boundaries between the components (either
    automatically based on probability changes or manually specified), assigns
    categorical labels to observations based on these boundaries, and provides
    plotting functionalities to visualize the results.

    Attributes
    ----------
    adata : ad.AnnData
        A copy of the input AnnData object, modified during processing.
    feature : str
        The name of the feature (column in adata.var) being thresholded.
    gmm_obs_label : str
        The key in `adata.obs` where the resulting category labels will be stored.
    gmm_kwargs : Dict
        Default keyword arguments passed to the `sklearn.mixture.GaussianMixture` model during fitting.
    internal_data : _SingleThresholdingEventModel
        Pydantic model storing all results for the current feature.
    manual_decision_boundaries : bool
        Flag indicating if thresholds were set manually.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        feature: str,
        label_obs_save_str: str,
        thresholding_events_key: str = 'gmm_thresholding_events',
        layer: Optional[str] = None,
        gmm_kwargs: Optional[dict] = None,
        random_state: int = 42,
    ):
        """Initialize the GMMThresholding object.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data matrix (observations x features).
        feature : str
            The name of the feature (must exist in `adata.var_names`)
            to perform thresholding on.
        label_obs_save_str : str
            The column name in `adata.obs` where
            the resulting categorical labels will be saved.
        thresholding_events_key : str, optional
            The key in `adata.uns` where
            thresholding event information will be stored. Defaults to 'gmm_thresholding_events'.
        layer : str, optional
            The layer in `adata.layers` to use for thresholding.
            If None, uses `adata.X`. Defaults to None.
        gmm_kwargs : dict, optional
            Default keyword arguments to pass to
            `sklearn.mixture.GaussianMixture`. Defaults to None (becomes {}).
        random_state : int, optional
            Random state for reproducibility. Defaults to 42.
        
        Raises
        ------
        TypeError
            If `adata` is not an AnnData object.
        TypeError
            If `adata.X` is not a numeric type.
        TypeError
            If `adata.uns[thresholding_events_key]` exists but is not an OrderedDict.
        KeyError
            If `feature` is not found in `adata.var_names`.
        TypeError
            If `feature` is not a string.
        ValueError
            If `feature` is an empty string.
        TypeError
            If `label_obs_save_str` is not a string.
        KeyError
            If `label_obs_save_str` already exists as a column in `adata.obs`.
        TypeError
            If `gmm_kwargs` is not a dictionary.
        TypeError
            If `thresholding_events_key` is not a string.
        ValueError
            If `layer` is specified but doesn't exist in `adata.layers`.
        """

        # Validate adata
        if not isinstance(adata, ad.AnnData):
            raise TypeError("adata must be an AnnData.AnnData object.")
        elif np.issubdtype(adata.X.dtype, np.object_):
            raise TypeError(
                "adata.X must be a numeric type. Please convert the data to a numeric type."
                )

        # Validate thresholding_events_key
        if not isinstance(thresholding_events_key, str):
            raise TypeError("thresholding_events_key must be a string.")
        elif not thresholding_events_key:
            raise ValueError("thresholding_events_key cannot be an empty string.")

        if thresholding_events_key not in adata.uns:
            adata.uns[thresholding_events_key] = OrderedDict()
        elif not isinstance(adata.uns[thresholding_events_key], OrderedDict):
            raise TypeError(
                f"The '{thresholding_events_key}' key in the AnnData object's `.uns` attribute must be an OrderedDict."
            )

        # Validate feature
        if not isinstance(feature, str):
            raise TypeError("feature must be a string.")
        elif not feature:
            raise ValueError("feature cannot be an empty string.")
        elif feature not in adata.var_names:
            raise KeyError(
                f"Feature '{feature}' not found in adata.var_names. Please check the feature name."
            )
        
        # Validate label_obs_save_str
        if not isinstance(label_obs_save_str, str):
            raise TypeError("label_obs_save_str must be a string.")
        elif not label_obs_save_str:
            raise ValueError("label_obs_save_str cannot be an empty string.")
        elif label_obs_save_str in adata.obs.columns:
            raise KeyError(
                f"obs key '{label_obs_save_str}' already exists in the AnnData object. Please choose a different label."
            )
        
        # Validate layer
        if layer is not None:
            if not isinstance(layer, str):
                raise TypeError("layer must be a string or None.")
            elif layer not in adata.layers.keys():
                raise ValueError(
                    f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}"
                )

        self.adata = adata.copy()
        self.thresholding_events_key = thresholding_events_key
        self.label_obs_save_str = label_obs_save_str
        self.feature: str = feature
        self.layer: Optional[str] = layer
        self.random_state: int = random_state

        # variable initialization
        self._manual_decision_boundaries: bool = False

        # Initialize  and validate gmm_kwargs
        if gmm_kwargs is None:
            self.gmm_kwargs = {'init_params': 'k-means++', 'n_init':10, 'max_iter':1000, 'random_state': self.random_state}
        elif isinstance(gmm_kwargs, dict):
            self.gmm_kwargs = gmm_kwargs
            if "random_state" not in self.gmm_kwargs:
                self.gmm_kwargs["random_state"] = self.random_state
        else:
            raise TypeError("gmm_kwargs must be a dictionary or None")

        self._gmm_info: Optional[_GaussianMixtureModelInfo] = _GaussianMixtureModelInfo(
            gmm_kwargs=self.gmm_kwargs,
        )
        self._decision_boundaries: Optional[_DecisionBoundariesModel] = None
        self._internal_data: Optional[_SingleThresholdingEventModel] = _SingleThresholdingEventModel(
            gmm_info=self._gmm_info,
            feature_name=self.feature,
            gmm_obs_label=self.label_obs_save_str,
        )

    def _get_feature_data(self) -> np.ndarray:
        """Get feature data from appropriate layer or .X.
        
        Returns
        -------
        np.ndarray
            Feature data as numpy array.
        """
        if self.layer is None:
            return self.adata[:, self.feature].X
        else:
            return self.adata[:, self.feature].layers[self.layer]

    def fit(
        self,
        n_components: int,
    ) -> None:
        """Fit a Gaussian Mixture Model (GMM) to the specified gene expression data.

        Parameters
        ----------
        n_components : int
            Number of Gaussian components to fit. Must be > 1.
        
        Raises
        ------
        TypeError
            If n_components is not an integer.
        ValueError
            If n_components is <= 1.
        """

        if not isinstance(n_components, numbers.Integral):
            raise TypeError("n_components must be a positive integer.")
        if n_components <=1:
            raise ValueError("n_components must be a positive integer.")

        x = self._get_feature_data().copy()

        gaussian_mixure_model = GaussianMixture(
            n_components=n_components, **self.gmm_kwargs)

        gaussian_mixure_model.fit(x)
        means = gaussian_mixure_model.means_.squeeze()
        covs = gaussian_mixure_model.covariances_.squeeze()
        weights = gaussian_mixure_model.weights_
        data_probs = gaussian_mixure_model.predict_proba(x)

        mean_argsort_idx = np.argsort(means)

        means, covs, weights, data_probs = (
            means[mean_argsort_idx],
            covs[mean_argsort_idx],
            weights[mean_argsort_idx],
            data_probs[:, mean_argsort_idx],
        )

        self._gmm_info = _GaussianMixtureModelInfo(
            gmm_kwargs=self.gmm_kwargs,
            means=means,
            covs=covs,
            weights=weights,
            data_probs=data_probs,
            n_components=n_components,
        )

        self._internal_data.gmm_info = self._gmm_info


    def return_adata(self,
                     overwrite: bool = False) -> ad.AnnData:
        """Serialize internal Pydantic models and return the modified AnnData object.
        
        Parameters
        ----------
        overwrite : bool, optional
            If True, allows overwriting existing entries. Defaults to False.
        
        Returns
        -------
        ad.AnnData
            The modified AnnData object with the serialized GMM thresholding events.
                
        Raises
        ------
        ValueError
            If the feature already exists in `adata.uns[thresholding_events_key]`
            and overwrite is False. This prevents overwriting existing entries.
        """
        if self.feature in self.adata.uns[self.thresholding_events_key].keys() and not overwrite:
            raise ValueError(
                f"The feature '{self.feature}' already exists in `adata.uns['{self.thresholding_events_key}']`. "
                "Please choose a different feature or set `overwrite=True` to overwrite the existing entry."
            )

        # Capture cell counts after categorization if labels exist
        cell_counts_after_operation = {}
        if self.label_obs_save_str in self.adata.obs.columns:
            counts = self.adata.obs[self.label_obs_save_str].value_counts()
            cell_counts_after_operation = {str(k): int(v) for k, v in counts.items()}
        
        # Store the internal data with cell counts
        stored_data = self._internal_data.model_dump()
        stored_data['cell_counts_after_operation'] = cell_counts_after_operation
        
        self.adata.uns[self.thresholding_events_key][self.feature] = stored_data
        return self.adata

    def plot_hist_distribution_with_boundaries(
        self,
        num_std: int = 5,
        title: Optional[str] = None,
        hist_kwargs: Optional[Dict] = None,
        cmap: plt.cm.ScalarMappable = plt.get_cmap('rainbow'),
        ax: plt.Axes = None,
        x_axis_limits: Optional[tuple] = None,
        resolution: int = 1000,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Axes:
        """Plot the histogram and GMM components with decision boundaries.
        
        Parameters
        ----------
        num_std : int, optional
            Number of standard deviations for GMM plotting. Defaults to 5.
        title : str, optional
            Plot title. If not provided, defaults to feature name.
            Pass empty string '' to suppress title. Defaults to None.
        hist_kwargs : dict, optional
            Kwargs for histogram. Defaults to None.
        cmap : plt.cm.ScalarMappable, optional
            Colormap. Defaults to 'rainbow'.
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new. Defaults to None.
        x_axis_limits : tuple, optional
            X-axis limits (min, max). Defaults to None.
        resolution : int, optional
            Resolution for plotting. Defaults to 1000.
        save_path : str or Path, optional
            Path to save the figure. 
            Parent directory must exist. Defaults to None.
        
        Returns
        -------
        plt.Axes
            The matplotlib axes object. Call plt.show() to display it.
        
        Raises
        ------
        FileNotFoundError
            If save_path parent directory doesn't exist.
        """
        # Validate save path before generating the figure
        from .base import _validate_save_path
        save_path = _validate_save_path(save_path)

        if self._internal_data.decision_boundaries is None:
            raise ValueError(
                "Decision boundaries have not been calculated. Please call calculate_decision_boundaries() first."
            )
        
        if resolution <= 0:
            raise ValueError("Resolution must be a positive integer.")
        
        if resolution <= self._internal_data.gmm_info.n_components: # pylint: disable=E1101
            raise ValueError(
                "Resolution must be greater than the number of GMM components."
            )

        # Call base class plotting method with explicit parameters
        ax = super()._plot_hist_base(
            adata=self.adata,
            feature=self.feature,
            layer=self.layer,
            hist_kwargs=hist_kwargs,
            ax=ax,
            x_axis_limits=x_axis_limits,
        )
        
        if not self._manual_decision_boundaries:
            ax = super()._plot_gmm_components(
                ax=ax,
                adata=self.adata,
                feature=self.feature,
                internal_data=self._internal_data,
                num_std=num_std,
                resolution=resolution,
                cmap=cmap
            )

        ax = super()._plot_vertical_linear_decision_boundaries(
            ax=ax,
            internal_data=self._internal_data,
            resolution=resolution,
            cmap=cmap
        )
        
        ax = super()._plot_sample_catergory_legend(
            ax=ax,
            internal_data=self._internal_data,
            cmap=cmap
        )

        # Add title (default to feature name, allow override or suppression)
        if title is None:
            title = self.feature
        if title:  # Only add title if not empty string
            ax.set_title(title)
        
        ax.set_xlabel(f"{self.feature}")
        ax.set_ylabel("Density")
        
        # Save figure if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to: {save_path}")
        
        return ax

    def _calculate_decision_boundaries(self) -> None:
        """Calculate decision boundary thresholds using the fitted GMM model."""
        feature_values = self._get_feature_data().flatten()
        gmm_info: _GaussianMixtureModelInfo = self._internal_data.gmm_info
        probabilities = np.array(gmm_info.data_probs) # pylint: disable=E1101
        # Call base class method and store result
        # Note: ordered_labels not available in this context (called before categorize_samples)
        self._internal_data.decision_boundaries = super()._calculate_decision_boundaries_from_probs(
            feature_values, probabilities, ordered_labels=None
        )

    def categorize_samples(self,
                           manual_thresholds: Optional[List[Union[float,int]]] = None,
                           ordered_labels: Optional[list] = None,
                           duplicate_labels: bool = False,) -> None:
        """Categorize samples based on GMM-derived or manual thresholds.
        
        This method assigns categorical labels to samples based on either
        automatically calculated decision boundaries from GMM fitting or
        manually specified thresholds. Supports collapsing multiple GMM
        components into fewer categories for cross-dataset robustness.
        
        Parameters
        ----------
        manual_thresholds : list of float or int, optional
            Explicit threshold values. If None, thresholds
            are calculated automatically from GMM probabilities. Must have
            length equal to (number of unique labels - 1).
        ordered_labels : list, optional
            Label for each GMM component. Length must equal
            n_components fitted in GMM. Can contain duplicates if
            duplicate_labels=True to merge multiple components into single
            categories. If None, generates default labels.
        duplicate_labels : bool, optional
            If True, allows duplicate labels to collapse
            multiple GMM components into single categories. Useful for
            cross-dataset robustness where many components are fitted for
            adaptive boundary placement but fewer final categories are desired.
            Defaults to False.
        
        Raises
        ------
        ValueError
            If GMM model has not been fitted first.
        ValueError
            If number of ordered_labels doesn't match n_components.
        ValueError
            If duplicate labels provided but duplicate_labels=False.
        TypeError
            If manual_thresholds is not a list.
        ValueError
            If manual_thresholds length doesn't match unique labels - 1.
        TypeError
            If threshold values are not numeric.
        
        Examples
        --------
        Standard usage with 2 components and 2 labels::
        
            gmm.fit(n_components=2)
            gmm.categorize_samples(ordered_labels=['Low', 'High'])
        
        Cross-dataset robustness - fit many components, collapse to binary::
        
            gmm.fit(n_components=8)
            gmm.categorize_samples(
                ordered_labels=['Low', 'Low', 'Low', 'High', 'High', 'High', 'High', 'High'],
                duplicate_labels=True
            )
            # Boundary automatically placed based on 8-component GMM fit
            
        Using manual thresholds::
        
            gmm.fit(n_components=2)
            gmm.categorize_samples(
                ordered_labels=['Low', 'High'],
                manual_thresholds=[0.05]
            )
        """

        if self._gmm_info.data_probs is None:
            raise ValueError(
                "GMM model has not been fitted. Please call fit() first."
            )

        # Set default labels
        if not ordered_labels:
            warnings.warn(
                "ordered_labels is not set. Using default labels.",
                UserWarning,
            )
            ordered_labels  = [f'Label {i}' for i in ascii_uppercase[:self._gmm_info.n_components]] # pylint: disable=E1101

        # Validate label count matches fitted components
        if len(ordered_labels) != self._gmm_info.n_components: # pylint: disable=E1101
            raise ValueError(
                f"Number of labels ({len(ordered_labels)}) must equal number of "
                f"fitted GMM components ({self._gmm_info.n_components})." # pylint: disable=E1101
            )

        # Check for duplicate labels
        unique_labels = list(dict.fromkeys(ordered_labels))  # Preserves order
        has_duplicates = len(unique_labels) < len(ordered_labels)
        
        if has_duplicates and not duplicate_labels:
            raise ValueError(
                "The ordered GMM labels contain duplicate values. Please ensure that the "
                "labels are unique or set duplicate_labels=True."
            )
        
        # Validate that duplicate labels are contiguous (not alternating)
        if has_duplicates and duplicate_labels:
            # Check for non-contiguous patterns like ['Low', 'High', 'Low', 'High']
            seen_labels = []
            for label in ordered_labels:
                if label not in seen_labels:
                    seen_labels.append(label)
                elif seen_labels[-1] != label:
                    # We've seen this label before, but the previous label was different
                    raise ValueError(
                        f"Non-contiguous duplicate labels detected. Labels must be grouped "
                        f"contiguously when using duplicate_labels=True. "
                        f"For example, ['Low', 'Low', 'High', 'High'] is valid, but "
                        f"['Low', 'High', 'Low', 'High'] is not. "
                    f"Got labels: {ordered_labels}"
                )
        
        feature_values = self._get_feature_data().flatten().copy()

        # Handle label collapsing if duplicates exist
        if has_duplicates:
            condensed_data_probs, condensed_labels = self._handle_duplicate_labels(
                self._gmm_info.data_probs, ordered_labels # pylint: disable=E1101
            )
            self._internal_data.condensed_labels = condensed_labels 
            self._internal_data.gmm_info.condensed_data_probs = condensed_data_probs # pylint: disable=E1101
            final_labels = condensed_labels
        else:
            final_labels = ordered_labels

        # Validate manual thresholds if provided
        if manual_thresholds is not None:
            if not isinstance(manual_thresholds, list):
                raise TypeError("manual_thresholds must be a list.")
            if len(manual_thresholds) != len(unique_labels) - 1:
                raise ValueError(
                    f"Number of thresholds ({len(manual_thresholds)}) must be one less than "
                    f"unique labels ({len(unique_labels)}). Got unique labels: {unique_labels}"
                )
            if not all(isinstance(threshold, (int, float)) for threshold in manual_thresholds):
                raise TypeError(
                    "All thresholds in manual_thresholds must be integers or floats."
                )
            
            # Check for duplicates
            if len(manual_thresholds) != len(set(manual_thresholds)):
                raise ValueError(
                    f"manual_thresholds contains duplicate values. All thresholds must be unique. "
                    f"Got: {manual_thresholds}"
                )
            
            # Check for ascending order
            if manual_thresholds != sorted(manual_thresholds):
                raise ValueError(
                    f"manual_thresholds must be in ascending order. "
                    f"Got: {manual_thresholds}, Expected: {sorted(manual_thresholds)}"
                )
            
            # Use manual thresholds
            self._internal_data.decision_boundaries = _DecisionBoundariesModel(
                thresholds=manual_thresholds
            )
            self._manual_decision_boundaries = True
        else:
            # Calculate thresholds from GMM (using condensed probs if labels were collapsed)
            if has_duplicates:
                self._internal_data.decision_boundaries = super()._calculate_decision_boundaries_from_probs(
                    feature_values, condensed_data_probs, ordered_labels=final_labels
                )
            else:
                self._calculate_decision_boundaries()
            
            self._manual_decision_boundaries = False

        # Assign samples to categories
        thresholds = self._internal_data.decision_boundaries.thresholds
        bin_indices = np.digitize(feature_values, thresholds)
        sample_labels = np.array(final_labels)[bin_indices]

        self.adata.obs[self.label_obs_save_str] = sample_labels
        self._internal_data.ordered_gmm_labels = ordered_labels

    def return_thresholds(self) -> List[float]:
        """Return the decision boundary thresholds.

        Returns
        -------
        list of float
            The decision boundary thresholds.
        
        Raises
        ------
        ValueError
            If decision boundaries have not been calculated yet.
        """
        if self._internal_data.decision_boundaries is None:
            raise ValueError(
                "Decision boundaries have not been calculated. Please call calculate_decision_boundaries() first."
            )
        return self._internal_data.decision_boundaries.thresholds

    def plot_strip_plot_histogram_with_decision_boundaries(
        self,
        cmap: plt.cm.ScalarMappable = mpl.colormaps['plasma'],
        y_axis_limits: Optional[tuple] = None,
        resolution: int = 1000,
        scatter_density: bool = True,
        vmax: Optional[Union[int, float]] = None,
        hist_kwargs: Optional[dict] = None,
        strip_plot_kwargs: Optional[dict] = None,
        title: Optional[str] = None,
    ) -> Figure:
        """Generate a strip plot with a histogram and decision boundaries.
        
        This method wraps the base class implementation, providing a convenient
        interface for single-feature thresholding visualizations.

        Parameters
        ----------
        cmap : plt.cm.ScalarMappable, optional
            Colormap for density or labels. 
            Defaults to mpl.colormaps['plasma'].
        y_axis_limits : tuple, optional
            Y-axis limits (min, max). 
            If None, uses data min/max. Defaults to None.
        resolution : int, optional
            Resolution for boundary plotting. Defaults to 1000.
        scatter_density : bool, optional
            If True, color by density; if False, 
            color by labels. Defaults to True.
        vmax : int or float, optional
            Maximum density value for colormap. 
            If None, auto-calculated. Defaults to None.
        hist_kwargs : dict, optional
            Kwargs for histogram (bins, color, etc.). 
            Defaults to None.
        strip_plot_kwargs : dict, optional
            Kwargs for strip plot scatter 
            (e.g., s, alpha, marker). Only used when scatter_density=False. 
            Defaults to None.
        title : str or None, optional
            Title for the plot. If not provided, 
            defaults to feature name. Pass empty string '' to suppress title.
            Defaults to None.
        
        Returns
        -------
        Figure
            The matplotlib figure object. Call plt.show() to display it.
        
        Raises
        ------
        ValueError
            If decision boundaries have not been calculated yet.
        """
        # Validate that thresholding has been performed
        if self._internal_data.decision_boundaries is None:
            raise ValueError(
                "Decision boundaries have not been calculated. "
                "Please call categorize_samples() first."
            )
        
        # Call base class implementation and return figure
        return super()._plot_strip_plot_histogram_with_decision_boundaries(
            adata=self.adata,
            feature=self.feature,
            layer=self.layer,
            obs_label=self.label_obs_save_str,
            ordered_labels=self._internal_data.ordered_gmm_labels,
            internal_data=self._internal_data,
            cmap=cmap,
            y_axis_limits=y_axis_limits,
            resolution=resolution,
            scatter_density=scatter_density,
            vmax=vmax,
            hist_kwargs=hist_kwargs,
            strip_plot_kwargs=strip_plot_kwargs,
            title=title,
        )

    def plot_feature_distribution_exploratory(
        self,
        hist_kwargs: Optional[Dict] = None,
        ax: Optional[plt.Axes] = None,
        x_axis_limits: Optional[tuple] = None,
    ) -> plt.Axes:
        """Plot histogram of the feature distribution for exploratory analysis.
        
        This method allows you to visualize the feature distribution WITHOUT running
        any thresholding, so you can explore your data and decide on manual thresholds
        or the number of components to use for GMM.
        
        Parameters
        ----------
        hist_kwargs : dict, optional
            Keyword arguments for plt.hist().
            Defaults to {'bins': 50, 'color': 'black', 'alpha': 0.7}.
        ax : plt.Axes, optional
            Matplotlib axes to plot on. If None, uses current axes.
        x_axis_limits : tuple, optional
            (min, max) for x-axis. Use None for data-driven limits.
        
        Returns
        -------
        plt.Axes
            The matplotlib axes object.
        
        Examples
        --------
        Explore DNA content distribution before deciding on components::
        
            gmm = GMMThresholding(
                adata=adata,
                feature='DNA_content',
                label_obs_save_str='cell_cycle'
            )
            gmm.plot_feature_distribution_exploratory(
                hist_kwargs={'bins': 30, 'color': 'steelblue'},
                x_axis_limits=(0, 10)
            )
            plt.title('DNA Content Distribution - Exploratory')
            plt.show()
        """
        if ax is None:
            ax = plt.gca()
        
        # Use base class histogram plotting method
        ax = super()._plot_hist_base(
            adata=self.adata,
            feature=self.feature,
            layer=self.layer,
            hist_kwargs=hist_kwargs,
            ax=ax,
            x_axis_limits=x_axis_limits,
        )
        
        return ax

    def plot_feature_strip_plot_exploratory(
        self,
        hist_kwargs: Optional[Dict] = None,
        strip_plot_kwargs: Optional[Dict] = None,
        scatter_density: bool = True,
        x_axis_limits: Optional[tuple] = None,
    ) -> tuple:
        """Plot strip plot + histogram for exploratory analysis.
        
        Similar to plot_strip_plot_histogram_with_decision_boundaries() but WITHOUT
        decision boundaries, for exploring data before running threshold operations.
        
        Parameters
        ----------
        hist_kwargs : dict, optional
            Keyword arguments for histogram.
        strip_plot_kwargs : dict, optional
            Keyword arguments for strip plot.
        scatter_density : bool, optional
            If True, uses density-based coloring. Defaults to True.
        x_axis_limits : tuple, optional
            (min, max) for x-axis.
        
        Returns
        -------
        tuple
            (fig, (ax_strip, ax_hist)) - Figure and axes objects.
        
        Examples
        --------
        Explore DNA content distribution with density visualization::
        
            gmm = GMMThresholding(
                adata=adata,
                feature='DNA_content',
                label_obs_save_str='cell_cycle'
            )
            fig, (ax_strip, ax_hist) = gmm.plot_feature_strip_plot_exploratory(
                scatter_density=True,
                x_axis_limits=(0, 10)
            )
            plt.suptitle('DNA Content Distribution - Exploratory')
            plt.show()
        """
        # Call base method to create strip plot and histogram (without decision boundaries)
        fig, ax_strip, ax_hist = super()._plot_strip_plot_base(
            adata=self.adata,
            feature=self.feature,
            layer=self.layer,
            obs_label=None,  # No labels for exploratory
            ordered_labels=None,  # No labels for exploratory
            scatter_density=scatter_density,
            y_axis_limits=x_axis_limits,  # Note: x_axis becomes y_axis in vertical plot
            hist_kwargs=hist_kwargs,
            strip_plot_kwargs=strip_plot_kwargs,
            cmap=mpl.colormaps['plasma'],
            vmax=None,
        )
        
        return fig, (ax_strip, ax_hist)


