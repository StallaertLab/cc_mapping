"""
Refactored manifold module with clean separation of PHATE computation and visualization.

Key improvements:
- Separated PHATE computation from side effects
- Dataclasses for configuration instead of dict-passing
- Decoupled from problematic plot.py functions
- Clear return type contracts
- Easier to test and maintain
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
from tqdm import tqdm

from .plot import (
    HyperparamGridBuilder,
    HyperparamGridConfig,
    get_legend,
)

np.seterr(all="ignore")


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class PHATEConfig:
    """Configuration for PHATE algorithm."""

    n_components: int = 2
    knn: int = 5
    decay: int = 40
    t: str = "auto"
    gamma: float = 1.0
    n_pca: int = 100
    n_jobs: int = -1
    random_state: Optional[int] = None
    verbose: bool = False

    def to_dict(self) -> dict:
        """Convert to dict for passing to phate.PHATE()."""
        return {
            "n_components": self.n_components,
            "knn": self.knn,
            "decay": self.decay,
            "t": self.t,
            "gamma": self.gamma,
            "n_pca": self.n_pca,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


@dataclass
class PHATEHyperparamGrid:
    """Configuration for PHATE hyperparameter grid search."""

    row_param_name: str
    col_param_name: str
    constant_param_name: str
    row_param_values: list
    col_param_values: list
    constant_param_values: list

    def iter_constant_values(self):
        """Iterate over constant parameter values."""
        for value in self.constant_param_values:
            yield value

    def get_params_for_cell(
        self,
        row_idx: int,
        col_idx: int,
        constant_value: any,
        base_config: PHATEConfig,
    ) -> dict:
        """Get PHATE parameters for a specific grid cell."""
        params = base_config.to_dict()
        params[self.row_param_name] = self.row_param_values[row_idx]
        params[self.col_param_name] = self.col_param_values[col_idx]
        params[self.constant_param_name] = constant_value
        return params


@dataclass
class PHATEPlotContext:
    """Context for plotting PHATE embeddings in hyperparam search."""

    phate_coords: np.ndarray
    adata: ad.AnnData
    color_name: str
    kwargs: dict = field(default_factory=dict)


# ============================================================================
# PHATE Computation (Pure Functions)
# ============================================================================


class PHATERunner:
    """Handles PHATE computation without side effects."""

    def __init__(
        self,
        adata: ad.AnnData,
        feature_set: str,
        layer: str,
    ):
        """
        Initialize PHATE runner.

        Args:
            adata: AnnData object containing the data.
            feature_set: Name of boolean column in adata.var indicating features to use.
            layer: Name of the layer to use from adata.layers.
        """
        self.adata = adata
        self.feature_set = feature_set
        self.layer = layer
        self._validate()

    def _validate(self):
        """Validate inputs."""
        if self.feature_set not in self.adata.var.columns:
            raise ValueError(f"Feature set '{self.feature_set}' not found in adata.var")
        if self.layer not in self.adata.layers:
            raise ValueError(f"Layer '{self.layer}' not found in adata.layers")

    def _get_data(self) -> np.ndarray:
        """Extract and prepare data for PHATE."""
        feature_set_bool = self.adata.var[self.feature_set].values
        data = self.adata.layers[self.layer][:, feature_set_bool].copy()
        return data

    def compute(self, config: PHATEConfig) -> np.ndarray:
        """
        Compute PHATE embedding.

        Args:
            config: PHATE configuration.

        Returns:
            PHATE coordinates as numpy array of shape (n_cells, n_components).
        """
        data = self._get_data()
        phate_operator = phate.PHATE(**config.to_dict())
        phate_coords = phate_operator.fit_transform(data)
        return phate_coords

    def compute_with_params(self, **phate_params) -> np.ndarray:
        """
        Compute PHATE embedding with custom parameters.

        Args:
            **phate_params: PHATE parameters to override defaults.

        Returns:
            PHATE coordinates as numpy array.
        """
        data = self._get_data()
        phate_operator = phate.PHATE(**phate_params)
        phate_coords = phate_operator.fit_transform(data)
        return phate_coords


def add_phate_to_adata(
    adata: ad.AnnData,
    phate_coords: np.ndarray,
    obsm_key: str = "X_phate",
) -> ad.AnnData:
    """
    Add PHATE coordinates to AnnData object.

    Args:
        adata: AnnData object to modify.
        phate_coords: PHATE coordinates to add.
        obsm_key: Key to store coordinates in adata.obsm.

    Returns:
        Modified AnnData object (same object, modified in place).
    """
    adata.obsm[obsm_key] = phate_coords
    return adata


def run_phate(
    adata: ad.AnnData,
    feature_set: str,
    layer: str,
    phate_config: Optional[PHATEConfig] = None,
    obsm_save_key: str = "X_phate",
) -> ad.AnnData:
    """
    Run PHATE and add results to AnnData object.

    This is a convenience function that combines computation and storage.
    For more control, use PHATERunner directly.

    Args:
        adata: Annotated data object.
        feature_set: Name of the feature set to use.
        layer: Name of the layer to use.
        phate_config: PHATE configuration. If None, uses defaults.
        obsm_save_key: Key to save the PHATE coordinates in adata.obsm.

    Returns:
        Modified AnnData object with PHATE coordinates in adata.obsm[obsm_save_key].
    """
    if phate_config is None:
        phate_config = PHATEConfig()

    runner = PHATERunner(adata, feature_set, layer)
    phate_coords = runner.compute(phate_config)
    add_phate_to_adata(adata, phate_coords, obsm_save_key)

    return adata


# ============================================================================
# PHATE Visualization
# ============================================================================


class PHATEVisualizer:
    """Handles PHATE embedding visualization."""

    @staticmethod
    def plot_embedding(
        ax: plt.Axes,
        phate_coords: np.ndarray,
        colors: np.ndarray | pd.Series,
        kwargs: Optional[dict] = None,
    ) -> plt.Axes:
        """
        Plot PHATE embedding on given axes.

        Args:
            ax: Matplotlib axes to plot on.
            phate_coords: PHATE coordinates (n_cells, 2).
            colors: Color values for each cell.
            kwargs: Additional kwargs for scatter plot.

        Returns:
            Modified axes object.
        """
        if kwargs is None:
            kwargs = {}
        else:
            kwargs = kwargs.copy()

        # Handle continuous vs categorical colors
        if not isinstance(colors, (pd.Categorical, pd.CategoricalDtype)):
            if colors.dtype != "object":
                vmin = np.percentile(colors, 1)
                vmax = np.percentile(colors, 99)
                kwargs.update(
                    {
                        "vmin": vmin,
                        "vmax": vmax,
                        "cmap": "rainbow",
                    }
                )

        ax.scatter(phate_coords[:, 0], phate_coords[:, 1], c=colors, **kwargs)

        # Clean up axes
        ax.axis("off")
        ax.set_yticks([])
        ax.set_xticks([])

        return ax

    @staticmethod
    def plot_from_adata(
        adata: ad.AnnData,
        color_name: str,
        obsm_embedding: str = "X_phate",
        ax: Optional[plt.Axes] = None,
        unit_size: int = 5,
        kwargs: Optional[dict] = None,
        return_fig: bool = False,
    ) -> plt.Axes | tuple[plt.Figure, plt.Axes]:
        """
        Plot PHATE embedding from AnnData object.

        Args:
            adata: AnnData object containing PHATE coordinates.
            color_name: Name of column in adata.obs to use for colors.
            obsm_embedding: Key for PHATE coordinates in adata.obsm.
            ax: Matplotlib axes to plot on. If None, creates new figure.
            unit_size: Size of the figure if creating new one.
            kwargs: Additional kwargs for scatter plot.
            return_fig: If True and ax is None, return both fig and ax.

        Returns:
            Axes object, or (Figure, Axes) tuple if return_fig=True.
        """
        if kwargs is None:
            kwargs = {}

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(unit_size, unit_size))
            created_fig = True

        phate_coords = adata.obsm[obsm_embedding]
        colors = adata.obs_vector(color_name)

        PHATEVisualizer.plot_embedding(ax, phate_coords, colors, kwargs)

        if created_fig and return_fig:
            return fig, ax

        return ax


# ============================================================================
# Hyperparameter Search
# ============================================================================


def plot_phate_hyperparam_cell(ax: plt.Axes, context: PHATEPlotContext) -> plt.Axes:
    """Plot a single cell in PHATE hyperparameter search grid."""
    colors = context.adata.obs_vector(context.color_name)
    PHATEVisualizer.plot_embedding(ax, context.phate_coords, colors, context.kwargs)
    return ax


def perform_phate_hyperparameter_search(
    adata: ad.AnnData,
    feature_set: str,
    layer: str,
    hyperparam_grid: PHATEHyperparamGrid,
    base_config: PHATEConfig,
    color_name: str,
    final_grid_dims: tuple[int, int],
    unit_size: int = 10,
    plot_kwargs: Optional[dict] = None,
    save_path: Optional[str] = None,
    show_legend: bool = False,
) -> list[plt.Figure]:
    """
    Perform hyperparameter search for PHATE visualization.

    Args:
        adata: Annotated data object.
        feature_set: Name of the feature set.
        layer: Name of the layer.
        hyperparam_grid: Hyperparameter grid configuration.
        base_config: Base PHATE configuration.
        color_name: Name of color column for visualization.
        final_grid_dims: Dimensions (rows, cols) for final combined figure.
        unit_size: Size of each subplot unit.
        plot_kwargs: Additional kwargs for scatter plots.
        save_path: Path to save figures.
        show_legend: Whether to show legend.

    Returns:
        List of generated figures (one per constant parameter value).
    """
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            raise ValueError(f"{save_dir} does not exist")

    if plot_kwargs is None:
        plot_kwargs = {}

    # Store original backend and switch to non-interactive
    backend = mpl.get_backend()
    mpl.use("agg")

    try:
        runner = PHATERunner(adata, feature_set, layer)
        figure_list = []

        num_constant_values = len(hyperparam_grid.constant_param_values)

        for const_idx, const_value in enumerate(
            tqdm(
                hyperparam_grid.constant_param_values,
                desc="Generating hyperparameter search plots",
                total=num_constant_values,
            )
        ):
            # Create grid config
            grid_config = HyperparamGridConfig(
                row_param_name=hyperparam_grid.row_param_name,
                col_param_name=hyperparam_grid.col_param_name,
                constant_param_name=hyperparam_grid.constant_param_name,
                row_param_values=hyperparam_grid.row_param_values,
                col_param_values=hyperparam_grid.col_param_values,
                constant_param_value=const_value,
                unit_size=unit_size,
            )

            # Build grid
            builder = HyperparamGridBuilder(
                num_rows=len(hyperparam_grid.row_param_values),
                num_cols=len(hyperparam_grid.col_param_values),
                unit_size=unit_size,
            )

            # Create context factory
            def context_factory(row_idx, col_idx, row_param_value, col_param_value):
                # Compute PHATE for this parameter combination
                params = hyperparam_grid.get_params_for_cell(
                    row_idx, col_idx, const_value, base_config
                )
                phate_coords = runner.compute_with_params(**params)

                return PHATEPlotContext(
                    phate_coords=phate_coords,
                    adata=adata,
                    color_name=color_name,
                    kwargs=plot_kwargs.copy(),
                )

            fig = builder.build(
                grid_config, plot_phate_hyperparam_cell, context_factory
            )
            figure_list.append(fig)

        # Add legend if requested
        if color_name is not None and show_legend:
            color_vector = adata.obs_vector(color_name)
            if (
                isinstance(color_vector, (pd.Categorical, pd.CategoricalDtype))
                or color_vector.dtype == "object"
            ):
                patches, _ = get_legend(adata, color_name)
                plt.legend(handles=patches, fontsize=unit_size)

        # Save if requested
        if save_path is not None:
            for idx, fig in enumerate(figure_list):
                path_base, path_ext = os.path.splitext(save_path)
                save_file = f"{path_base}_{idx}{path_ext}"
                fig.savefig(save_file, dpi=300, bbox_inches="tight")

        return figure_list

    finally:
        # Restore original backend
        mpl.use(backend)


# ============================================================================
# Convenience Functions
# ============================================================================


def quick_phate_plot(
    adata: ad.AnnData,
    feature_set: str,
    layer: str,
    color_name: str,
    phate_config: Optional[PHATEConfig] = None,
    save_path: Optional[str] = None,
) -> tuple[ad.AnnData, plt.Figure]:
    """
    Quick PHATE computation and visualization.

    Args:
        adata: AnnData object.
        feature_set: Feature set name.
        layer: Layer name.
        color_name: Color column name.
        phate_config: PHATE configuration (optional).
        save_path: Path to save figure (optional).

    Returns:
        Tuple of (modified adata with PHATE coords, figure).
    """
    # Compute PHATE
    adata = run_phate(adata, feature_set, layer, phate_config)

    # Plot
    fig, ax = PHATEVisualizer.plot_from_adata(
        adata,
        color_name,
        return_fig=True,
        unit_size=10,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return adata, fig
