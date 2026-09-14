"""
Sequential GMM thresholding implementation.

This module provides the SequentialGMM class for
performing iterative refinement of categorical labels through multiple sequential
thresholding operations.

Classes:
    SequentialGMM: Sequential refinement class
"""

import warnings
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.mixture import GaussianMixture

from .base import (
    GaussianMixtureModelBase,
    _DecisionBoundariesModel,
    _GaussianMixtureModelInfo,
    _SingleThresholdingEventModel,
    _validate_save_path,
)
from .single import GMMThresholding


class SequentialGMM(GaussianMixtureModelBase):
    """
    Sequential GMM thresholding for iterative population refinement.

    This class enables performing multiple sequential GMM thresholding operations
    on subsets of cells, where each operation refines a specific categorical label
    from a previous thresholding event. This is useful for hierarchical cell type
    classification or iterative gating strategies.

    Unlike GMMThresholding which thresholds a single feature once,
    this class allows:

    - Initial thresholding on entire dataset
    - Refinement of specific label values through additional thresholding
    - Tracking operation provenance (parent-child relationships)
    - Multiple operations stored in a single .uns key

    Attributes
    ----------
    adata : ad.AnnData
        A copy of the input AnnData object, modified during processing.
    thresholding_events_key : str
        Key in adata.uns for storing all operations.
    gmm_kwargs : Dict
        Default GMM kwargs (can be overridden per operation).
    random_state : int
        Random state for reproducibility.

    Examples
    --------
    Example workflow::

        # Initialize
        seq_gmm = SequentialGMM(
            adata=adata,
            thresholding_events_key='sequential_thresholding'
        )

        # Create initial labels on entire dataset
        seq_gmm.threshold_entire_dataset(
            feature='DNA_content',
            label_obs_save_str='cell_cycle',
            n_components=2,
            ordered_labels=['Low', 'High'],
            operation_name='DNA_threshold'
        )

        # Refine 'Low' cells only
        seq_gmm.refine_labels_with_gmm(
            feature='Plk1',
            obs_label='cell_cycle',
            value_to_refine='Low',
            n_components=2,
            ordered_labels=['Low_neg', 'Low_pos'],
            operation_name='Plk1_refinement'
        )

        # Get modified adata
        adata = seq_gmm.return_adata()
    """

    def __init__(
        self,
        adata: ad.AnnData,
        thresholding_events_key: str = "sequential_gmm_thresholding_events",
        gmm_kwargs: dict | None = None,
        random_state: int = 42,
    ):
        """
        Initialize sequential thresholding object.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data matrix (observations x features).
        thresholding_events_key : str, optional
            The key in `adata.uns` where all thresholding event information will be stored.
            Defaults to 'sequential_gmm_thresholding_events'.
        gmm_kwargs : Optional[Dict], optional
            Default keyword arguments to pass to `sklearn.mixture.GaussianMixture`.
            Can be overridden per operation. Defaults to None (becomes {}).
        random_state : int, optional
            Random state for reproducibility. Defaults to 42.

        Raises
        ------
        TypeError
            If `adata` is not an AnnData object.
        TypeError
            If `adata.X` is not a numeric type.
        TypeError
            If `thresholding_events_key` is not a string.
        ValueError
            If `thresholding_events_key` is an empty string.
        TypeError
            If `adata.uns[thresholding_events_key]` exists but is not an OrderedDict.
        TypeError
            If `gmm_kwargs` is not a dictionary.
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

        # Create or validate .uns key
        if thresholding_events_key not in adata.uns:
            adata.uns[thresholding_events_key] = OrderedDict()
        elif not isinstance(adata.uns[thresholding_events_key], OrderedDict):
            raise TypeError(
                f"The '{thresholding_events_key}' key in the AnnData object's `.uns` attribute must be an OrderedDict."
            )

        # Validate gmm_kwargs
        if gmm_kwargs is None:
            gmm_kwargs = {}
        elif not isinstance(gmm_kwargs, dict):
            raise TypeError("gmm_kwargs must be a dictionary.")

        # Store attributes
        self.adata = adata.copy()
        self.thresholding_events_key = thresholding_events_key
        self.gmm_kwargs = gmm_kwargs
        self.random_state = random_state

    def threshold_entire_dataset(
        self,
        feature: str,
        label_obs_save_str: str,
        n_components: int,
        ordered_labels: list[str],
        manual_thresholds: list[float | int] | None = None,
        duplicate_labels: bool = False,
        operation_name: str | None = None,
        layer: str | None = None,
        gmm_kwargs: dict | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Threshold entire dataset to create initial categorical labels.

        This method creates a new obs column with categorical labels based on
        GMM thresholding of a single feature across all cells. It's a wrapper
        around GMMThresholding that stores results in the
        sequential thresholding framework.

        Parameters
        ----------
        feature : str
            Feature name to threshold on (must exist in adata.var_names).
        label_obs_save_str : str
            New column name in adata.obs for labels.
        n_components : int
            Number of GMM components to fit.
        ordered_labels : List[str]
            Labels to assign (length = n_components).
        manual_thresholds : Optional[List[Union[float, int]]], optional
            Manual threshold values. If None, calculated automatically from GMM.
            Length must be n_components - 1. Defaults to None.
        duplicate_labels : bool, optional
            Allow duplicate labels for label collapsing. Defaults to False.
        operation_name : str
            Required name for tracking this operation.
        layer : Optional[str], optional
            Layer to use for data access. If None, uses adata.X. Defaults to None.
        gmm_kwargs : Optional[dict], optional
            GMM kwargs for this operation. Overrides default if provided. Defaults to None.
        overwrite : bool, optional
            If True, allows overwriting an existing operation with the same name.
            Useful for updating n_components or thresholds. Defaults to False.

        Raises
        ------
        ValueError
            If operation_name is None or empty.
        KeyError
            If operation_name already exists in .uns and overwrite=False.

        Notes
        -----
        Other exceptions raised by GMMThresholding.

        Examples
        --------
        ::

            seq_gmm.threshold_entire_dataset(
                feature='DNA_content',
                label_obs_save_str='cell_cycle',
                n_components=3,
                ordered_labels=['G0', 'G1', 'S'],
                operation_name='DNA_initial_threshold'
            )
        """
        # Validate operation_name
        if operation_name is None or not operation_name:
            raise ValueError("operation_name is required and cannot be empty.")

        if operation_name in self.adata.uns[self.thresholding_events_key]:
            if not overwrite:
                raise KeyError(
                    f"Operation name '{operation_name}' already exists in "
                    f"adata.uns['{self.thresholding_events_key}']. Use overwrite=True to update it."
                )
            else:
                warnings.warn(
                    f"Overwriting existing operation '{operation_name}'.",
                    UserWarning,
                    stacklevel=2,
                )

        # Use default gmm_kwargs if not provided
        if gmm_kwargs is None:
            gmm_kwargs = self.gmm_kwargs

        # Create temporary single thresholding instance
        # Use a temporary .uns key to avoid conflicts
        temp_key = f"_temp_{operation_name}"
        gmm_single = GMMThresholding(
            adata=self.adata,
            feature=feature,
            label_obs_save_str=label_obs_save_str,
            thresholding_events_key=temp_key,
            layer=layer,
            gmm_kwargs=gmm_kwargs,
            random_state=self.random_state,
        )

        # Fit and categorize
        gmm_single.fit(n_components=n_components)
        gmm_single.categorize_samples(
            ordered_labels=ordered_labels,
            manual_thresholds=manual_thresholds,
            duplicate_labels=duplicate_labels,
        )

        # Extract results from temporary instance
        self.adata = gmm_single.return_adata()

        # Move operation data from temp key to our key with metadata
        # Single class stores with feature name as key
        operation_data = self.adata.uns[temp_key][feature]

        # Add sequential-specific metadata
        operation_data["operation_type"] = "standard"
        operation_data["parent_operation"] = None
        operation_data["refined_from_labels"] = None
        operation_data["layer"] = layer

        # Capture cell counts immediately after this operation
        cell_counts_after_operation = {
            str(k): int(v)
            for k, v in self.adata.obs[label_obs_save_str].value_counts().items()
        }
        operation_data["cell_counts_after_operation"] = cell_counts_after_operation

        # Store in our key with the operation_name
        self.adata.uns[self.thresholding_events_key][operation_name] = operation_data

        # Clean up temp key
        del self.adata.uns[temp_key]

    def refine_labels_with_gmm(
        self,
        feature: str,
        obs_label: str,
        value_to_refine: str,
        n_components: int,
        ordered_labels: list[str],
        duplicate_labels: bool = False,
        operation_name: str | None = None,
        layer: str | None = None,
        gmm_kwargs: dict | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Refine existing categorical labels by thresholding a subset with GMM.

        Modifies adata.obs[obs_label] in-place (within the copy), replacing cells
        with value_to_refine with new labels based on GMM thresholding.

        Parameters
        ----------
        feature : str
            Feature to threshold on (e.g., 'Plk1').
        obs_label : str
            Obs column to modify in-place (e.g., 'cell_cycle').
        value_to_refine : str
            Which label value to refine (e.g., 'G0').
        n_components : int
            Number of GMM components to fit.
        ordered_labels : List[str]
            New labels to assign (e.g., ['G0_low', 'G0_high']).
        duplicate_labels : bool, optional
            Allow duplicate labels for label collapsing. Defaults to False.
        operation_name : str
            Required name for tracking this operation.
        layer : Optional[str], optional
            Layer to use for data access. If None, uses adata.X. Defaults to None.
        gmm_kwargs : Optional[dict], optional
            GMM kwargs for this operation. Overrides default if provided. Defaults to None.
        overwrite : bool, optional
            If True, allows overwriting an existing operation with the same name.
            Useful for updating n_components or thresholds. Defaults to False.

        Raises
        ------
        ValueError
            If operation_name is None or empty.
        KeyError
            If operation_name already exists in .uns and overwrite=False.
        KeyError
            If obs_label doesn't exist in adata.obs.
        ValueError
            If value_to_refine is not present in adata.obs[obs_label].
        ValueError
            If no cells have the value_to_refine.

        Examples
        --------
        ::

            # Before: adata.obs['cell_cycle'] = ['G0', 'G0', 'G1', 'S', 'G0']
            seq_gmm.refine_labels_with_gmm(
                feature='Plk1',
                obs_label='cell_cycle',
                value_to_refine='G0',
                n_components=2,
                ordered_labels=['G0_low', 'G0_high'],
                operation_name='Plk1_G0_refinement'
            )
            # After: adata.obs['cell_cycle'] = ['G0_low', 'G0_high', 'G1', 'S', 'G0_low']
        """
        # Validate operation_name
        if operation_name is None or not operation_name:
            raise ValueError("operation_name is required and cannot be empty.")

        if operation_name in self.adata.uns[self.thresholding_events_key]:
            if not overwrite:
                raise KeyError(
                    f"Operation name '{operation_name}' already exists in "
                    f"adata.uns['{self.thresholding_events_key}']. Use overwrite=True to update it."
                )
            else:
                warnings.warn(
                    f"Overwriting existing operation '{operation_name}'.",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate obs_label exists
        if obs_label not in self.adata.obs.columns:
            raise KeyError(
                f"obs_label '{obs_label}' not found in adata.obs. "
                f"Available columns: {list(self.adata.obs.columns)}"
            )

        # Partition data by label
        mask, subset_data = self._partition_data_by_label(
            obs_label=obs_label,
            value_to_refine=value_to_refine,
            feature=feature,
            layer=layer,
        )

        # Use default gmm_kwargs if not provided
        if gmm_kwargs is None:
            gmm_kwargs = self.gmm_kwargs.copy()

        # Ensure random_state is in gmm_kwargs (but don't override if explicitly provided)
        if "random_state" not in gmm_kwargs:
            gmm_kwargs["random_state"] = self.random_state

        # Fit GMM on subset only
        gmm = GaussianMixture(n_components=n_components, **gmm_kwargs)
        gmm.fit(subset_data.reshape(-1, 1))

        # Extract GMM info
        gmm_info = _GaussianMixtureModelInfo(
            gmm_kwargs=gmm_kwargs,
            means=gmm.means_.flatten(),
            covs=gmm.covariances_.flatten(),
            weights=gmm.weights_,
            n_components=n_components,
            data_probs=gmm.predict_proba(subset_data.reshape(-1, 1)),
        )

        # Handle duplicate labels if needed
        if duplicate_labels:
            ordered_labels_processed, condensed_data_probs = (
                self._handle_duplicate_labels(ordered_labels, gmm_info.data_probs)
            )
            gmm_info.condensed_data_probs = condensed_data_probs
        else:
            ordered_labels_processed = ordered_labels

        # Calculate decision boundaries from probabilities
        # Convert data_probs back to numpy array (Pydantic stores as list)
        probs_array = np.array(
            gmm_info.condensed_data_probs if duplicate_labels else gmm_info.data_probs
        )
        decision_boundaries = self._calculate_decision_boundaries_from_probs(
            feature_values=subset_data,
            data_probs=probs_array,
            ordered_labels=ordered_labels_processed,
        )

        # Assign new labels to subset cells
        new_labels = self._assign_labels_from_thresholds(
            data=subset_data,
            thresholds=decision_boundaries.thresholds,
            ordered_labels=ordered_labels_processed,
        )

        # Update labels in-place
        self._update_labels_in_place(
            obs_label=obs_label,
            mask=mask,
            new_labels=new_labels,
        )

        # Capture cell counts immediately after this operation
        cell_counts_after_operation = {
            str(k): int(v) for k, v in pd.Series(new_labels).value_counts().items()
        }

        # Store operation metadata
        internal_data = _SingleThresholdingEventModel(
            gmm_info=gmm_info,
            ordered_gmm_labels=ordered_labels,
            decision_boundaries=decision_boundaries,
            condensed_labels=ordered_labels_processed if duplicate_labels else None,
            feature_name=feature,
            gmm_obs_label=obs_label,
        )

        self._store_refinement_operation(
            operation_name=operation_name,
            internal_data=internal_data,
            obs_label=obs_label,
            value_to_refine=value_to_refine,
            operation_type="refinement",
            layer=layer,
            cell_counts_after_operation=cell_counts_after_operation,
        )

    def refine_labels_with_manual_thresholds(
        self,
        feature: str,
        obs_label: str,
        value_to_refine: str,
        manual_thresholds: list[float | int],
        ordered_labels: list[str],
        operation_name: str | None = None,
        layer: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Refine existing categorical labels using manual thresholds.

        Similar to refine_labels_with_gmm() but uses explicit threshold values
        instead of fitting a GMM.

        Parameters
        ----------
        feature : str
            Feature to threshold on.
        obs_label : str
            Obs column to modify in-place.
        value_to_refine : str
            Which label value to refine.
        manual_thresholds : List[Union[float, int]]
            Threshold values. Length must be len(ordered_labels) - 1.
        ordered_labels : List[str]
            New labels to assign.
        operation_name : str
            Required name for tracking this operation.
        layer : Optional[str], optional
            Layer to use for data access. If None, uses adata.X. Defaults to None.
        overwrite : bool, optional
            If True, allows overwriting an existing operation with the same name.
            Useful for updating thresholds. Defaults to False.

        Raises
        ------
        ValueError
            If operation_name is None or empty.
        KeyError
            If operation_name already exists in .uns and overwrite=False.
        KeyError
            If obs_label doesn't exist in adata.obs.
        ValueError
            If value_to_refine is not present in adata.obs[obs_label].
        ValueError
            If no cells have the value_to_refine.
        ValueError
            If len(manual_thresholds) != len(ordered_labels) - 1.

        Examples
        --------
        ::

            seq_gmm.refine_labels_with_manual_thresholds(
                feature='Plk1',
                obs_label='cell_cycle',
                value_to_refine='G0',
                manual_thresholds=[1.5],
                ordered_labels=['G0_low', 'G0_high'],
                operation_name='Plk1_G0_manual'
            )
        """
        # Validate operation_name
        if operation_name is None or not operation_name:
            raise ValueError("operation_name is required and cannot be empty.")

        if operation_name in self.adata.uns[self.thresholding_events_key]:
            if not overwrite:
                raise KeyError(
                    f"Operation name '{operation_name}' already exists in "
                    f"adata.uns['{self.thresholding_events_key}']. Use overwrite=True to update it."
                )
            else:
                warnings.warn(
                    f"Overwriting existing operation '{operation_name}'.",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate obs_label exists
        if obs_label not in self.adata.obs.columns:
            raise KeyError(
                f"obs_label '{obs_label}' not found in adata.obs. "
                f"Available columns: {list(self.adata.obs.columns)}"
            )

        # Validate manual_thresholds length
        if len(manual_thresholds) != len(ordered_labels) - 1:
            raise ValueError(
                f"Number of thresholds ({len(manual_thresholds)}) must be "
                f"len(ordered_labels) - 1 = {len(ordered_labels) - 1}"
            )

        # Partition data by label
        mask, subset_data = self._partition_data_by_label(
            obs_label=obs_label,
            value_to_refine=value_to_refine,
            feature=feature,
            layer=layer,
        )

        # Assign new labels based on manual thresholds
        new_labels = self._assign_labels_from_thresholds(
            data=subset_data,
            thresholds=manual_thresholds,
            ordered_labels=ordered_labels,
        )

        # Update labels in-place
        self._update_labels_in_place(
            obs_label=obs_label,
            mask=mask,
            new_labels=new_labels,
        )

        # Capture cell counts immediately after this operation
        cell_counts_after_operation = {
            str(k): int(v) for k, v in pd.Series(new_labels).value_counts().items()
        }

        # Store operation metadata (no GMM info for manual thresholds)
        decision_boundaries = _DecisionBoundariesModel(thresholds=manual_thresholds)

        internal_data = _SingleThresholdingEventModel(
            gmm_info=None,  # No GMM for manual thresholds
            ordered_gmm_labels=ordered_labels,
            decision_boundaries=decision_boundaries,
            condensed_labels=None,
            feature_name=feature,
            gmm_obs_label=obs_label,
        )

        self._store_refinement_operation(
            operation_name=operation_name,
            internal_data=internal_data,
            obs_label=obs_label,
            value_to_refine=value_to_refine,
            operation_type="refinement_manual",
            layer=layer,
            cell_counts_after_operation=cell_counts_after_operation,
        )

    def _partition_data_by_label(
        self, obs_label: str, value_to_refine: str, feature: str, layer: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract subset of data for cells with specific label value.

        Parameters
        ----------
        obs_label : str
            Obs column name to check.
        value_to_refine : str
            Label value to select.
        feature : str
            Feature name to extract data for.
        layer : Optional[str]
            Layer to use (None = .X).

        Returns
        -------
        mask : np.ndarray
            Boolean array indicating which cells to refine.
        feature_data : np.ndarray
            Feature values for those cells (1D array).

        Raises
        ------
        ValueError
            If value_to_refine is not present in obs_label.
        ValueError
            If no cells have the value_to_refine.
        """
        # Check if value_to_refine exists
        unique_values = self.adata.obs[obs_label].unique()
        if value_to_refine not in unique_values:
            # Check if this operation exists in metadata and may have been previously refined
            error_msg = (
                f"value_to_refine '{value_to_refine}' not found in adata.obs['{obs_label}'].\n"
                f"Available values: {list(unique_values)}"
            )

            # Check if operation with this value_to_refine exists in history
            if (
                hasattr(self.adata, "uns")
                and self.thresholding_events_key in self.adata.uns
            ):
                metadata = self.adata.uns[self.thresholding_events_key]
                for op_name, op_data in metadata.items():
                    if op_data.get("label_to_refine") == value_to_refine:
                        refined_labels = op_data.get("ordered_gmm_labels", [])
                        # Check if those refined labels exist now
                        existing_refined = [
                            lbl for lbl in refined_labels if lbl in unique_values
                        ]
                        if existing_refined:
                            error_msg += (
                                f"\n\nNote: Operation '{op_name}' previously refined '{value_to_refine}' "
                                f"into {refined_labels}.\n"
                                f"These labels currently exist: {existing_refined}\n\n"
                                f"To re-run this analysis:\n"
                                f"  1. Restart from the initialization of the SequentialGMM object, or\n"
                                f"  2. Use the exploratory plotting functions to visualize current labels:\n"
                                f"     seq_gmm.plot_feature_distribution_exploratory(...)\n"
                                f"     seq_gmm.plot_feature_strip_plot_exploratory(...)"
                            )
                        break

            raise ValueError(error_msg)

        # Create mask for cells with value_to_refine
        mask = self.adata.obs[obs_label] == value_to_refine

        # Check if any cells match
        if not mask.any():
            raise ValueError(
                f"No cells found with value '{value_to_refine}' in adata.obs['{obs_label}']"
            )

        # Extract feature data for subset
        if layer is None:
            feature_data = self.adata[mask, feature].X
        else:
            feature_data = self.adata[mask, feature].layers[layer]

        # Ensure 1D array
        if hasattr(feature_data, "toarray"):
            feature_data = feature_data.toarray()
        feature_data = np.asarray(feature_data).flatten()

        return mask, feature_data

    def _assign_labels_from_thresholds(
        self,
        data: np.ndarray,
        thresholds: list[float | int],
        ordered_labels: list[str],
    ) -> np.ndarray:
        """
        Assign categorical labels based on threshold values.

        Parameters
        ----------
        data : np.ndarray
            1D array of feature values.
        thresholds : List[Union[float, int]]
            List of threshold values (sorted low to high).
        ordered_labels : List[str]
            Labels corresponding to threshold bins.

        Returns
        -------
        np.ndarray
            Array of labels (same length as data).

        Raises
        ------
        ValueError
            If number of labels doesn't match number of thresholds.
        """
        # Validate that we have the right number of labels
        expected_labels = len(thresholds) + 1
        if len(ordered_labels) != expected_labels:
            raise ValueError(
                f"Number of labels ({len(ordered_labels)}) must equal number of thresholds + 1 ({expected_labels}). "
                f"Thresholds: {thresholds}, Labels: {ordered_labels}"
            )

        # Initialize with first label
        labels = np.full(len(data), ordered_labels[0], dtype=object)

        # Assign labels based on thresholds
        for i, threshold in enumerate(thresholds):
            labels[data > threshold] = ordered_labels[i + 1]

        return labels

    def _update_labels_in_place(
        self, obs_label: str, mask: np.ndarray, new_labels: np.ndarray
    ) -> None:
        """
        Update obs column in-place for masked cells.

        Parameters
        ----------
        obs_label : str
            Obs column to modify.
        mask : np.ndarray
            Boolean mask indicating which cells to update.
        new_labels : np.ndarray
            New label values for masked cells.
        """
        # Convert to categorical if not already
        if not isinstance(self.adata.obs[obs_label].dtype, pd.CategoricalDtype):
            self.adata.obs[obs_label] = self.adata.obs[obs_label].astype("category")

        # Add new categories if they don't exist
        existing_categories = self.adata.obs[obs_label].cat.categories
        new_categories = set(new_labels) - set(existing_categories)
        if new_categories:
            self.adata.obs[obs_label] = self.adata.obs[obs_label].cat.add_categories(
                list(new_categories)
            )

        # Update values for masked cells
        self.adata.obs.loc[mask, obs_label] = new_labels

    def _store_refinement_operation(
        self,
        operation_name: str,
        internal_data: _SingleThresholdingEventModel,
        obs_label: str,
        value_to_refine: str,
        operation_type: str,
        layer: str | None,
        cell_counts_after_operation: dict[str, int],
    ) -> None:
        """
        Store operation metadata in .uns.

        Parameters
        ----------
        operation_name : str
            Name for this operation.
        internal_data : _SingleThresholdingEventModel
            Pydantic model with thresholding data.
        obs_label : str
            Which obs column was modified.
        value_to_refine : str
            Which label value was refined.
        operation_type : str
            'refinement' or 'refinement_manual'.
        layer : Optional[str]
            Which layer was used (None = .X).
        cell_counts_after_operation : Dict[str, int]
            Dictionary of label counts immediately after operation.
        """
        # Convert to dict for storage
        operation_dict = internal_data.model_dump()

        # Add sequential-specific metadata
        operation_dict["operation_type"] = operation_type
        operation_dict["parent_operation"] = obs_label
        operation_dict["refined_from_labels"] = [value_to_refine]
        operation_dict["layer"] = layer
        operation_dict["cell_counts_after_operation"] = cell_counts_after_operation

        # Store in .uns
        self.adata.uns[self.thresholding_events_key][operation_name] = operation_dict

    def return_adata(self) -> ad.AnnData:
        """
        Return the modified AnnData object.

        Returns
        -------
        ad.AnnData
            Modified AnnData object with all operations applied.

        Examples
        --------
        ::

            seq_gmm = SequentialGMM(adata)
            seq_gmm.threshold_entire_dataset(...)
            seq_gmm.refine_labels_with_gmm(...)
            adata_modified = seq_gmm.return_adata()
        """
        return self.adata

    def _get_descendant_labels(
        self, operation_name: str, original_labels: list[str]
    ) -> list[str]:
        """
        Get all labels that descended from an operation's labels.

        This handles cases where subsequent operations refined labels from this operation.
        For example, if 'separate_M_phase' created ['M', 'G1/S/G2'], and a later
        operation refined 'G1/S/G2' into ['G1', 'S', 'G2'], this returns
        ['M', 'G1', 'S', 'G2'].

        Parameters
        ----------
        operation_name : str
            Name of the operation.
        original_labels : List[str]
            The labels created by this operation.

        Returns
        -------
        List[str]
            List of all current labels that descended from original_labels.
        """
        all_labels = set(original_labels)

        # Check all subsequent operations to see if they refined any of our labels
        metadata = self.adata.uns[self.thresholding_events_key]
        for op_name, op_data in metadata.items():
            # Skip the current operation and operations before it
            if op_name == operation_name:
                continue

            # Check if this operation refined one of our labels
            if op_data.get("refined_from_labels"):
                refined_from = op_data["refined_from_labels"][
                    0
                ]  # Should be single label
                if refined_from in all_labels:
                    # Replace the parent label with child labels
                    all_labels.discard(refined_from)
                    all_labels.update(op_data["ordered_gmm_labels"])

        return list(all_labels)

    def plot_feature_distribution_exploratory(
        self,
        feature: str,
        obs_label: str | None = None,
        value_to_subset: str | None = None,
        layer: str | None = None,
        hist_kwargs: dict | None = None,
        ax: Axes | None = None,
        x_axis_limits: tuple | None = None,
    ) -> Axes:
        """
        Plot histogram of a feature distribution for exploratory analysis.

        This method allows you to visualize feature distributions WITHOUT running
        any thresholding, so you can explore your data and decide on manual thresholds
        or the number of components to use for GMM.

        Parameters
        ----------
        feature : str
            Feature name to plot (must exist in adata.var_names).
        obs_label : Optional[str], optional
            Obs column to use for subsetting. If provided with value_to_subset,
            only plots cells with that label value. If None, plots all cells.
            Defaults to None.
        value_to_subset : Optional[str], optional
            Specific label value to plot. Requires obs_label to be specified.
            If None, plots all cells (or all cells in obs_label if provided).
            Defaults to None.
        layer : Optional[str], optional
            Layer to use for data. If None, uses adata.X. Defaults to None.
        hist_kwargs : Optional[Dict], optional
            Keyword arguments for plt.hist().
            Defaults to {'bins': 50, 'color': 'black', 'alpha': 0.7}.
        ax : Optional[plt.Axes], optional
            Matplotlib axes to plot on. If None, uses current axes. Defaults to None.
        x_axis_limits : Optional[tuple], optional
            (min, max) for x-axis. Use None for data-driven limits. Defaults to None.

        Returns
        -------
        Axes
            The matplotlib axes object.

        Raises
        ------
        ValueError
            If value_to_subset is provided without obs_label.
        KeyError
            If obs_label doesn't exist in adata.obs.
        ValueError
            If value_to_subset is not present in adata.obs[obs_label].

        Examples
        --------
        Explore entire dataset::

            seq_gmm.plot_feature_distribution_exploratory(
                feature='Int_Intg_DNA_nuc',
                hist_kwargs={'bins': 30, 'color': 'steelblue'},
                x_axis_limits=(5, 15)
            )
            plt.title('DNA Content Distribution - All Cells')
            plt.show()

        Explore specific subset::

            seq_gmm.plot_feature_distribution_exploratory(
                feature='Int_Intg_DNA_nuc',
                obs_label='cell_cycle_phase',
                value_to_subset='G1/S/G2',
                hist_kwargs={'bins': 30, 'color': 'steelblue'},
                x_axis_limits=(5, 15)
            )
            plt.title('DNA Content in G1/S/G2 Cells - Exploratory')
            plt.show()
        """
        if ax is None:
            ax = plt.gca()

        # Validate parameters
        if value_to_subset is not None and obs_label is None:
            raise ValueError(
                "obs_label must be provided when value_to_subset is specified."
            )

        # Get subset of adata if value_to_subset provided
        if value_to_subset is not None:
            mask, _ = self._partition_data_by_label(
                obs_label=obs_label,
                value_to_refine=value_to_subset,
                feature=feature,
                layer=layer,
            )
            adata_subset = self.adata[mask, :]
        else:
            adata_subset = self.adata

        # Use base class histogram plotting method
        ax = super()._plot_hist_base(
            adata=adata_subset,
            feature=feature,
            layer=layer,
            hist_kwargs=hist_kwargs,
            ax=ax,
            x_axis_limits=x_axis_limits,
        )

        return ax

    def plot_feature_strip_plot_exploratory(
        self,
        feature: str,
        obs_label: str | None = None,
        value_to_subset: str | None = None,
        layer: str | None = None,
        hist_kwargs: dict | None = None,
        strip_plot_kwargs: dict | None = None,
        scatter_density: bool = True,
        x_axis_limits: tuple | None = None,
        vmax: float | None = None,
    ) -> tuple:
        """
        Plot strip plot + histogram for exploratory analysis.

        Similar to plot_strip_plot_histogram_with_decision_boundaries() but WITHOUT
        decision boundaries, for exploring data before running threshold operations.

        This method allows you to visualize feature distributions WITHOUT running
        any thresholding, so you can explore your data and decide on manual thresholds
        or the number of components to use for GMM.

        Parameters
        ----------
        feature : str
            Feature name to plot (must exist in adata.var_names).
        obs_label : Optional[str], optional
            Obs column to use for subsetting. If provided with value_to_subset,
            only plots cells with that label value. If None, plots all cells.
            Defaults to None.
        value_to_subset : Optional[str], optional
            Specific label value to plot. Requires obs_label to be specified.
            If None, plots all cells (or all cells in obs_label if provided).
            Defaults to None.
        layer : Optional[str], optional
            Layer to use for data. If None, uses adata.X. Defaults to None.
        hist_kwargs : Optional[Dict], optional
            Keyword arguments for histogram. Defaults to None.
        strip_plot_kwargs : Optional[Dict], optional
            Keyword arguments for strip plot. Only used when scatter_density=False.
            Defaults to None.
        scatter_density : bool, optional
            If True, uses density-based coloring. If False, uses uniform scatter plot.
            Defaults to True.
        x_axis_limits : Optional[tuple], optional
            (min, max) for x-axis. Use None for data-driven limits. Defaults to None.
        vmax : Optional[Union[int, float]], optional
            Maximum density value for colormap. Only used when scatter_density=True.
            If None, auto-calculated. Defaults to None.

        Returns
        -------
        tuple
            (fig, (ax_strip, ax_hist)) - Figure and axes objects.

        Raises
        ------
        ValueError
            If value_to_subset is provided without obs_label.
        KeyError
            If obs_label doesn't exist in adata.obs.
        ValueError
            If value_to_subset is not present in adata.obs[obs_label].

        Examples
        --------
        Explore entire dataset::

            fig, (ax_strip, ax_hist) = seq_gmm.plot_feature_strip_plot_exploratory(
                feature='Int_Intg_DNA_nuc',
                scatter_density=True,
                x_axis_limits=(5, 15)
            )
            plt.suptitle('DNA Content Distribution - All Cells')
            plt.show()

        Explore specific subset::

            fig, (ax_strip, ax_hist) = seq_gmm.plot_feature_strip_plot_exploratory(
                feature='Int_Intg_DNA_nuc',
                obs_label='cell_cycle_phase',
                value_to_subset='G1/S/G2',
                scatter_density=True,
                x_axis_limits=(5, 15)
            )
            plt.suptitle('DNA Content Distribution - G1/S/G2 Cells')
            plt.show()
        """
        # Validate parameters
        if value_to_subset is not None and obs_label is None:
            raise ValueError(
                "obs_label must be provided when value_to_subset is specified."
            )

        # Get subset of adata if value_to_subset provided
        if value_to_subset is not None:
            mask, _ = self._partition_data_by_label(
                obs_label=obs_label,
                value_to_refine=value_to_subset,
                feature=feature,
                layer=layer,
            )
            adata_subset = self.adata[mask, :]
        else:
            adata_subset = self.adata

        # Call base method to create strip plot and histogram (without decision boundaries)
        fig, ax_strip, ax_hist = super()._plot_strip_plot_base(
            adata=adata_subset,
            feature=feature,
            layer=layer,
            obs_label=None,  # No labels for exploratory
            ordered_labels=None,  # No labels for exploratory
            scatter_density=scatter_density,
            y_axis_limits=x_axis_limits,  # Note: x_axis becomes y_axis in vertical plot
            hist_kwargs=hist_kwargs,
            strip_plot_kwargs=strip_plot_kwargs,
            cmap=mpl.colormaps["plasma"],
            vmax=vmax,
        )

        return fig, (ax_strip, ax_hist)

    def plot_hist_distribution_with_boundaries(
        self,
        operation_name: str,
        num_std: int = 5,
        title: str | None = None,
        hist_kwargs: dict | None = None,
        cmap: mpl_colors.Colormap | None = None,
        ax: Axes | None = None,
        x_axis_limits: tuple | None = None,
        resolution: int = 1000,
        save_path: str | Path | None = None,
    ) -> Axes:
        """
        Plot histogram with boundaries for a specific operation.

        Parameters
        ----------
        operation_name : str
            Name of operation to plot (from .uns keys).
        num_std : int, optional
            Number of standard deviations for GMM plotting. Defaults to 5.
        title : Optional[str], optional
            Plot title. If not provided, defaults to feature name.
            Pass empty string '' to suppress title. Defaults to None.
        hist_kwargs : Optional[Dict], optional
            Kwargs for histogram. Defaults to None.
        cmap : plt.cm.ScalarMappable, optional
            Colormap. Defaults to 'rainbow'.
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new. Defaults to None.
        x_axis_limits : Optional[tuple], optional
            X-axis limits (min, max). Defaults to None.
        resolution : int, optional
            Resolution for plotting. Defaults to 1000.
        save_path : Optional[Union[str, Path]], optional
            Path to save the figure. Parent directory must exist. Defaults to None.

        Returns
        -------
        Axes
            The matplotlib axes object. Call plt.show() to display it.

        Raises
        ------
        KeyError
            If operation_name not found in .uns.
        ValueError
            If operation has no decision boundaries.
        ValueError
            If resolution <= 0 or <= n_components.
        FileNotFoundError
            If save_path parent directory doesn't exist.

        Examples
        --------
        ::

            ax = seq_gmm.plot_hist_distribution_with_boundaries('Plk1_refinement')
            plt.show()
        """
        # Validate save path before generating the figure
        save_path = _validate_save_path(save_path)

        # Set default colormap if not provided
        if cmap is None:
            cmap = plt.get_cmap("rainbow")

        # Validate operation_name exists
        if operation_name not in self.adata.uns[self.thresholding_events_key]:
            raise KeyError(
                f"Operation '{operation_name}' not found in "
                f"adata.uns['{self.thresholding_events_key}']. "
                f"Available operations: {list(self.adata.uns[self.thresholding_events_key].keys())}"
            )

        # Load operation data from .uns
        op_data = self.adata.uns[self.thresholding_events_key][operation_name]
        internal_data = _SingleThresholdingEventModel(**op_data)
        feature = op_data["feature_name"]
        layer = op_data.get("layer", None)
        obs_label = op_data["gmm_obs_label"]
        ordered_labels = op_data["ordered_gmm_labels"]

        # Validate decision boundaries exist
        if internal_data.decision_boundaries is None:
            raise ValueError(
                f"Decision boundaries not found for operation '{operation_name}'. "
                "This should not happen."
            )

        # Validate resolution
        if resolution <= 0:
            raise ValueError("Resolution must be a positive integer.")

        if internal_data.gmm_info is not None:
            if resolution <= internal_data.gmm_info.n_components:
                raise ValueError(
                    "Resolution must be greater than the number of GMM components."
                )

        # Filter adata to only include cells with labels from this operation
        # For refinement operations, we need to include all descendant labels
        # since subsequent operations may have further refined the labels
        if op_data.get("refined_from_labels"):
            # This was a refinement operation - find all labels that descended from it
            labels_to_plot = self._get_descendant_labels(operation_name, ordered_labels)
        else:
            # This was an initial threshold operation - use ordered_labels directly
            labels_to_plot = ordered_labels

        mask = self.adata.obs[obs_label].isin(labels_to_plot)
        adata_subset = self.adata[mask, :]

        # Call base class plotting method with explicit parameters
        ax = super()._plot_hist_base(
            adata=adata_subset,
            feature=feature,
            layer=layer,
            hist_kwargs=hist_kwargs,
            ax=ax,
            x_axis_limits=x_axis_limits,
        )

        # Plot GMM components if this was GMM-based (not manual)
        if internal_data.gmm_info is not None:
            ax = super()._plot_gmm_components(
                ax=ax,
                adata=self.adata,
                feature=feature,
                internal_data=internal_data,
                num_std=num_std,
                resolution=resolution,
                cmap=cmap,
            )

        # Plot decision boundaries
        ax = super()._plot_vertical_linear_decision_boundaries(
            ax=ax, internal_data=internal_data, resolution=resolution, cmap=cmap
        )

        # Plot legend
        super()._plot_sample_catergory_legend(
            ax=ax, internal_data=internal_data, cmap=cmap, legend_kwargs=None
        )

        # Add title (default to feature name, allow override or suppression)
        if title is None:
            title = feature
        if title:  # Only add title if not empty string
            ax.set_title(title)

        # Save figure if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Figure saved to: {save_path}")

        return ax

    def plot_strip_plot_histogram_with_decision_boundaries(
        self,
        operation_name: str,
        cmap: mpl_colors.Colormap | None = None,
        y_axis_limits: tuple[float, float] | None = None,
        resolution: int = 1000,
        scatter_density: bool = True,
        vmax: float | None = None,
        hist_kwargs: dict | None = None,
        strip_plot_kwargs: dict | None = None,
        title: str | None = None,
    ) -> Figure:
        """
        Plot 1D strip plot with histogram and decision boundaries for a specific operation.

        This method wraps the base class implementation to provide visualization
        for sequential thresholding operations. It creates a density strip plot
        (or label-colored scatter) alongside a horizontal histogram showing the
        distribution and decision boundaries for the specified operation.

        Parameters
        ----------
        operation_name : str
            Name of operation to plot (from .uns keys).
        cmap : plt.cm.ScalarMappable, optional
            Colormap for density or labels. Defaults to mpl.colormaps['plasma'].
        y_axis_limits : Optional[Tuple[float, float]], optional
            Y-axis limits (min, max). If None, uses data min/max. Defaults to None.
        resolution : int, optional
            Resolution for boundary plotting. Defaults to 1000.
        scatter_density : bool, optional
            If True, color by density; if False, color by labels. Defaults to True.
        vmax : Optional[Union[int, float]], optional
            Maximum density value for colormap. If None, auto-calculated. Defaults to None.
        hist_kwargs : Optional[Dict], optional
            Kwargs for histogram (bins, color, etc.). Defaults to None.
        strip_plot_kwargs : Optional[Dict], optional
            Kwargs for strip plot scatter (e.g., s, alpha, marker).
            Only used when scatter_density=False. Defaults to None.
        title : Optional[str], optional
            Title for the plot. If not provided, defaults to feature name.
            Pass empty string '' to suppress title. Defaults to None.

        Returns
        -------
        Figure
            The matplotlib figure object. Call plt.show() to display it.

        Raises
        ------
        KeyError
            If operation_name not found in .uns.
        ValueError
            If operation has no decision boundaries.

        Examples
        --------
        Basic usage with label-colored scatter::

            fig = seq_gmm.plot_strip_plot_histogram_with_decision_boundaries(
                operation_name='separate_M_phase',
                scatter_density=False
            )
            plt.show()

        Custom title::

            fig = seq_gmm.plot_strip_plot_histogram_with_decision_boundaries(
                operation_name='separate_M_phase',
                scatter_density=False,
                title='M Phase Separation'
            )
            plt.show()

        Customize strip plot appearance::

            fig = seq_gmm.plot_strip_plot_histogram_with_decision_boundaries(
                operation_name='separate_M_phase',
                scatter_density=False,
                strip_plot_kwargs={'s': 5, 'alpha': 0.8, 'marker': 'o'}
            )
            plt.show()
        """
        # Validate operation_name exists
        if operation_name not in self.adata.uns[self.thresholding_events_key]:
            raise KeyError(
                f"Operation '{operation_name}' not found in "
                f"adata.uns['{self.thresholding_events_key}']. "
                f"Available operations: {list(self.adata.uns[self.thresholding_events_key].keys())}"
            )

        # Load operation data from .uns
        op_data = self.adata.uns[self.thresholding_events_key][operation_name]
        internal_data = _SingleThresholdingEventModel(**op_data)
        feature = op_data["feature_name"]
        layer = op_data.get("layer", None)
        obs_label = op_data["gmm_obs_label"]
        ordered_labels = op_data["ordered_gmm_labels"]

        # Validate decision boundaries exist
        if internal_data.decision_boundaries is None:
            raise ValueError(
                f"Decision boundaries not found for operation '{operation_name}'."
            )

        # Filter adata to only include cells with labels from this operation
        # For refinement operations, we need to include all descendant labels
        # since subsequent operations may have further refined the labels
        if op_data.get("refined_from_labels"):
            # This was a refinement operation - find all labels that descended from it
            labels_to_plot = self._get_descendant_labels(operation_name, ordered_labels)
        else:
            # This was an initial threshold operation - use ordered_labels directly
            labels_to_plot = ordered_labels

        mask = self.adata.obs[obs_label].isin(labels_to_plot)
        adata_subset = self.adata[mask, :]

        # Set default colormap if not provided
        if cmap is None:
            cmap = mpl.colormaps["plasma"]

        # Call base class implementation and return figure
        return super()._plot_strip_plot_histogram_with_decision_boundaries(
            adata=adata_subset,
            feature=feature,
            layer=layer,
            obs_label=obs_label,
            ordered_labels=ordered_labels,
            internal_data=internal_data,
            cmap=cmap,
            y_axis_limits=y_axis_limits,
            resolution=resolution,
            scatter_density=scatter_density,
            vmax=vmax,
            hist_kwargs=hist_kwargs,
            strip_plot_kwargs=strip_plot_kwargs,
            title=title,
        )
