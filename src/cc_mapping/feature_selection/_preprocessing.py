"""
Data preprocessing utilities for feature selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import anndata as ad
import numpy as np
from scipy import sparse


@dataclass
class PreparedData:
    """
    Container for prepared feature matrix and labels.
    
    Attributes
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    feature_names : np.ndarray
        Names of features corresponding to columns in X.
    valid_mask : np.ndarray
        Boolean mask of shape (n_original_samples,) indicating kept rows.
    dropped_indices : np.ndarray
        Integer indices of dropped rows in the original data.
    drop_reasons : dict
        Dictionary with keys like 'nan_labels', 'nan_features', 'inf_values'
        mapping to arrays of dropped indices for each reason.
    """
    X: np.ndarray
    y: np.ndarray
    feature_names: np.ndarray
    valid_mask: np.ndarray
    dropped_indices: np.ndarray
    drop_reasons: dict = field(default_factory=dict)
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the prepared data."""
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features in the prepared data."""
        return self.X.shape[1]
    
    @property
    def n_dropped(self) -> int:
        """Number of dropped samples."""
        return len(self.dropped_indices)
    
    def __repr__(self) -> str:
        return (
            f"PreparedData(n_samples={self.n_samples}, "
            f"n_features={self.n_features}, n_dropped={self.n_dropped})"
        )


def _get_feature_indices(
    adata: ad.AnnData,
    feature_set: str | list[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get feature indices and names from AnnData.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object.
    feature_set : str or list[str] or np.ndarray
        Either a key in adata.var (boolean column marking features),
        or a list/array of feature names.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (indices, feature_names).
        
    Raises
    ------
    KeyError
        If feature_set is a string but not found in adata.var.
    ValueError
        If feature names in list are not found in adata.var_names.
    """
    var_names = adata.var_names.values
    
    if isinstance(feature_set, str):
        # Feature set is a key in adata.var (boolean column)
        if feature_set not in adata.var.columns:
            raise KeyError(
                f"Feature set key '{feature_set}' not found in adata.var. "
                f"Available columns: {list(adata.var.columns)}"
            )
        mask = adata.var[feature_set].values.astype(bool)
        indices = np.where(mask)[0]
        feature_names = var_names[indices]
    else:
        # Feature set is a list/array of feature names
        feature_set = np.asarray(feature_set)
        
        # Check all features exist
        missing = set(feature_set) - set(var_names)
        if missing:
            raise ValueError(
                f"Features not found in adata.var_names: {missing}"
            )
        
        # Take the columns in adata.var_names order, not list order: the forest's
        # feature subsampling depends on column order, so the selection would
        # otherwise change with how the caller happened to order the names.
        indices = np.where(np.isin(var_names, feature_set))[0]
        feature_names = var_names[indices]
    
    return indices, feature_names


def prepare_feature_matrix(
    adata: ad.AnnData,
    feature_set: str | list[str] | np.ndarray,
    labels: str,
    drop_na: bool = True,
    drop_inf: bool = True,
    nan_label_values: list[str] | None = None,
    verbose: bool = True,
) -> PreparedData:
    """
    Extract and validate feature matrix from AnnData.
    
    This function extracts features and labels from an AnnData object,
    optionally removing rows with NaN or infinite values. It provides
    detailed information about which rows were dropped and why.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing the data.
    feature_set : str or list[str] or np.ndarray
        Either a key in adata.var (boolean column marking features),
        or a list/array of feature names to use. Order does not matter:
        the columns always follow adata.var_names.
    labels : str
        Key in adata.obs containing target labels.
    drop_na : bool, default=True
        Whether to drop rows with NaN values in features.
    drop_inf : bool, default=True
        Whether to drop rows with infinite values in features.
    nan_label_values : list[str], optional
        Label values to treat as NaN (e.g., ['nan', 'unknown']).
        These rows will be dropped. Default is ['nan'].
    verbose : bool, default=True
        Whether to print information about dropped rows.
        
    Returns
    -------
    PreparedData
        Container with X, y, feature_names, and drop information.
        
    Raises
    ------
    KeyError
        If labels key not found in adata.obs.
    KeyError
        If feature_set key not found in adata.var (when str).
    ValueError
        If feature names not found in adata.var_names (when list).
        
    Examples
    --------
    >>> prepared = prepare_feature_matrix(
    ...     adata,
    ...     feature_set="intensity_features",  # boolean column in adata.var
    ...     labels="cell_cycle_phase",
    ... )
    >>> print(f"Prepared {prepared.n_samples} samples with {prepared.n_features} features")
    >>> print(f"Dropped {prepared.n_dropped} rows")
    
    >>> # Using explicit feature list
    >>> prepared = prepare_feature_matrix(
    ...     adata,
    ...     feature_set=["gene_A", "gene_B", "gene_C"],
    ...     labels="treatment",
    ... )
    """
    if nan_label_values is None:
        nan_label_values = ["nan"]
    
    # Validate labels column exists
    if labels not in adata.obs.columns:
        raise KeyError(
            f"Labels key '{labels}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    # Get feature indices and names
    feature_indices, feature_names = _get_feature_indices(adata, feature_set)
    
    # Extract feature matrix and labels. A sparse X is densified so the NaN/inf
    # checks and the forest see the same values as for a dense X.
    X = adata.X[:, feature_indices]
    X = X.toarray() if sparse.issparse(X) else X.copy()
    y = adata.obs[labels].values.copy()
    
    n_original = len(y)
    drop_reasons: dict[str, np.ndarray] = {}
    
    # Track all indices to drop
    to_drop = np.zeros(n_original, dtype=bool)
    
    # 1. Drop rows with NaN labels
    if nan_label_values:
        y_str = y.astype(str)
        nan_label_mask = np.isin(y_str, nan_label_values)
        if nan_label_mask.any():
            drop_reasons["nan_labels"] = np.where(nan_label_mask)[0]
            to_drop |= nan_label_mask
    
    # 2. Drop rows with NaN in features
    if drop_na:
        nan_feature_mask = np.isnan(X).any(axis=1)
        if nan_feature_mask.any():
            drop_reasons["nan_features"] = np.where(nan_feature_mask & ~to_drop)[0]
            to_drop |= nan_feature_mask
    
    # 3. Drop rows with infinite values in features
    if drop_inf:
        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.any():
            drop_reasons["inf_values"] = np.where(inf_mask & ~to_drop)[0]
            to_drop |= inf_mask
    
    # Apply mask
    valid_mask = ~to_drop
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    dropped_indices = np.where(to_drop)[0]
    
    if verbose and len(dropped_indices) > 0:
        print(f"Prepared feature matrix: {n_original} -> {len(y_clean)} samples")
        print(f"  Dropped {len(dropped_indices)} rows total:")
        for reason, indices in drop_reasons.items():
            print(f"    - {reason}: {len(indices)} rows")
    elif verbose:
        print(f"Prepared feature matrix: {len(y_clean)} samples, {len(feature_names)} features")
    
    return PreparedData(
        X=X_clean,
        y=y_clean,
        feature_names=np.asarray(feature_names),
        valid_mask=valid_mask,
        dropped_indices=dropped_indices,
        drop_reasons=drop_reasons,
    )


def validate_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: np.ndarray | None = None,
) -> None:
    """
    Validate that data is ready for feature selection.
    
    Raises descriptive errors if data contains NaN or infinite values.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    feature_names : np.ndarray, optional
        Feature names for better error messages.
        
    Raises
    ------
    ValueError
        If X or y contain NaN or infinite values.
    ValueError
        If X and y have mismatched number of samples.
    ValueError
        If feature_names length doesn't match X columns.
    """
    # Check shapes match
    if X.shape[0] != len(y):
        raise ValueError(
            f"X has {X.shape[0]} samples but y has {len(y)} samples. "
            "They must have the same number of rows."
        )
    
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X has {X.shape[1]} columns."
        )
    
    # Check for NaN in X
    nan_mask = np.isnan(X)
    if nan_mask.any():
        nan_rows = np.where(nan_mask.any(axis=1))[0]
        nan_cols = np.where(nan_mask.any(axis=0))[0]
        
        col_info = ""
        if feature_names is not None:
            col_info = f" (features: {feature_names[nan_cols][:5]}...)"
        
        raise ValueError(
            f"X contains NaN values in {len(nan_rows)} rows and {len(nan_cols)} columns{col_info}. "
            "Use prepare_feature_matrix() with drop_na=True, or handle NaN values manually."
        )
    
    # Check for infinite values in X
    inf_mask = np.isinf(X)
    if inf_mask.any():
        inf_rows = np.where(inf_mask.any(axis=1))[0]
        inf_cols = np.where(inf_mask.any(axis=0))[0]
        
        col_info = ""
        if feature_names is not None:
            col_info = f" (features: {feature_names[inf_cols][:5]}...)"
        
        raise ValueError(
            f"X contains infinite values in {len(inf_rows)} rows and {len(inf_cols)} columns{col_info}. "
            "Use prepare_feature_matrix() with drop_inf=True, or handle infinite values manually."
        )
    
    # Check for NaN in labels (only if y is numeric). Pandas Categorical labels
    # have no numpy dtype, so look at them as a plain array.
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.floating):
        if np.isnan(y).any():
            n_nan = np.isnan(y).sum()
            raise ValueError(
                f"y contains {n_nan} NaN values. "
                "Remove these rows before feature selection."
            )
