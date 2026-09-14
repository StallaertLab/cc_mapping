"""
Refactored plotting module with improved structure and separation of concerns.

Key improvements:
- Dataclasses instead of dict-passing for type safety
- Separated grid builders for different use cases
- Native matplotlib composition instead of array manipulation
- Clear separation between data prep and visualization
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import anndata as ad
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class PlotConfig:
    """Base configuration for plotting."""

    unit_size: int = 10
    kwargs: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class RowPartitionConfig(PlotConfig):
    """Configuration for row partition plots."""

    adata: ad.AnnData
    obs_search_term: str
    colors: list | np.ndarray
    column_labels: Optional[list | np.ndarray] = None
    obs_embedding_key: str = "X_phate"
    plot_all: bool = True
    plot_background: bool = True


@dataclass(kw_only=True)
class HyperparamGridConfig(PlotConfig):
    """Configuration for hyperparameter search grids."""

    row_param_name: str
    col_param_name: str
    constant_param_name: str
    row_param_values: list
    col_param_values: list
    constant_param_value: any = None


# ============================================================================
# Grid Label Rendering
# ============================================================================


class GridLabelRenderer:
    """Handles rendering of row and column labels on grid plots."""

    def __init__(self, fontsize: int = 35):
        self.fontsize = fontsize
        self.row_cmap = mpl.colormaps["tab20"]
        self.col_cmap = mpl.colormaps["Dark2"]

    def add_column_labels(
        self,
        ax: plt.Axes,
        labels: list,
        param_name: Optional[str] = None,
    ) -> plt.Axes:
        """Add colored column labels to the top of the grid."""
        num_cols = len(labels)
        col_limits = np.linspace(0, 1, num_cols + 1)

        for col_idx, label in enumerate(labels):
            left = col_limits[col_idx]
            right = col_limits[col_idx + 1]
            center = (left + right) / 2

            ax.axvspan(left, right, facecolor=self.row_cmap(col_idx), alpha=0.5)

            label_text = f"{param_name} = {label}" if param_name else str(label)
            ax.annotate(
                label_text,
                xy=(center, 0.5),
                xycoords="axes fraction",
                va="center",
                ha="center",
                fontsize=self.fontsize,
            )

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, 1)

        return ax

    def add_row_labels(
        self,
        ax: plt.Axes,
        labels: list,
        param_name: Optional[str] = None,
    ) -> plt.Axes:
        """Add colored row labels to the left of the grid."""
        num_rows = len(labels)
        row_limits = np.linspace(0, 1, num_rows + 1)

        # Reverse labels to go top-to-bottom
        for row_idx, label in enumerate(reversed(labels)):
            upper = row_limits[row_idx]
            lower = row_limits[row_idx + 1]
            center = (upper + lower) / 2

            ax.axhspan(lower, upper, facecolor=self.col_cmap(row_idx), alpha=0.5)

            label_text = f"{param_name} = {label}" if param_name else str(label)
            ax.annotate(
                label_text,
                xy=(0.5, center),
                xycoords="axes fraction",
                va="center",
                ha="center",
                fontsize=self.fontsize,
                rotation=90,
            )

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, 1)

        return ax

    def add_corner_label(
        self,
        ax: plt.Axes,
        text: str,
    ) -> plt.Axes:
        """Add label to corner (top-left) cell."""
        ax.annotate(
            text,
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            va="center",
            ha="center",
            fontsize=25,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        return ax


# ============================================================================
# Grid Layout Builders
# ============================================================================


class GridLayoutBuilder:
    """Base class for building grid layouts."""

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        unit_size: int = 10,
        label_proportion: float = 0.20,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.unit_size = unit_size
        self.label_proportion = label_proportion
        self.label_renderer = GridLabelRenderer(fontsize=5 * unit_size)

    def create_figure(self) -> tuple[plt.Figure, GridSpec]:
        """Create figure and grid specification."""
        width_ratios = [self.label_proportion] + [1] * self.num_cols
        height_ratios = [self.label_proportion] + [1] * self.num_rows

        fig = plt.figure(
            figsize=(self.unit_size * self.num_cols, self.unit_size * self.num_rows),
            constrained_layout=True,
        )

        gs = GridSpec(
            self.num_rows + 1,
            self.num_cols + 1,
            figure=fig,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        return fig, gs


class RowPartitionGridBuilder(GridLayoutBuilder):
    """Builder for row partition grid layouts."""

    def build(
        self,
        config: RowPartitionConfig,
        plotting_function: Callable,
    ) -> plt.Figure:
        """Build the complete row partition grid."""
        # Setup column labels
        if config.column_labels is None:
            col_labels = np.unique(config.adata.obs[config.obs_search_term])
            if config.plot_all:
                col_labels = np.append(col_labels, "ALL")
        else:
            col_labels = config.column_labels

        # Update dimensions
        self.num_rows = len(config.colors)
        self.num_cols = len(col_labels)

        fig, gs = self.create_figure()

        # Add column labels (top row)
        col_label_ax = fig.add_subplot(gs[0, 1:])
        self.label_renderer.add_column_labels(col_label_ax, col_labels)

        # Add row labels (left column)
        row_label_ax = fig.add_subplot(gs[1:, 0])
        self.label_renderer.add_row_labels(row_label_ax, config.colors)

        # Create grid cells
        for row_idx in range(self.num_rows):
            for col_idx in range(self.num_cols):
                ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])

                # Call plotting function with context
                context = RowPartitionPlotContext(
                    adata=config.adata,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    color_name=config.colors[row_idx],
                    column_label=col_labels[col_idx],
                    obs_search_term=config.obs_search_term,
                    obs_embedding_key=config.obs_embedding_key,
                    plot_background=config.plot_background,
                    kwargs=config.kwargs.copy(),
                )

                plotting_function(ax, context)

        return fig


class HyperparamGridBuilder(GridLayoutBuilder):
    """Builder for hyperparameter search grid layouts."""

    def build(
        self,
        config: HyperparamGridConfig,
        plotting_function: Callable,
        plotting_context_factory: Callable,
    ) -> plt.Figure:
        """Build the complete hyperparameter grid."""
        self.num_rows = len(config.row_param_values)
        self.num_cols = len(config.col_param_values)

        fig, gs = self.create_figure()

        # Add column labels (top row)
        col_label_ax = fig.add_subplot(gs[0, 1:])
        self.label_renderer.add_column_labels(
            col_label_ax,
            config.col_param_values,
            param_name=config.col_param_name,
        )

        # Add row labels (left column)
        row_label_ax = fig.add_subplot(gs[1:, 0])
        self.label_renderer.add_row_labels(
            row_label_ax,
            config.row_param_values,
            param_name=config.row_param_name,
        )

        # Add corner label for constant parameter
        if config.constant_param_value is not None:
            corner_ax = fig.add_subplot(gs[0, 0])
            label_text = f"{config.constant_param_name}={config.constant_param_value}"
            self.label_renderer.add_corner_label(corner_ax, label_text)

        # Create grid cells
        for row_idx in range(self.num_rows):
            for col_idx in range(self.num_cols):
                ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])

                # Create context using factory
                context = plotting_context_factory(
                    row_idx=row_idx,
                    col_idx=col_idx,
                    row_param_value=config.row_param_values[row_idx],
                    col_param_value=config.col_param_values[col_idx],
                )

                plotting_function(ax, context)

        return fig


# ============================================================================
# Plot Context Dataclasses
# ============================================================================


@dataclass
class RowPartitionPlotContext:
    """Context passed to row partition plotting functions."""

    adata: ad.AnnData
    row_idx: int
    col_idx: int
    color_name: str
    column_label: str
    obs_search_term: str
    obs_embedding_key: str
    plot_background: bool
    kwargs: dict


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_row_partition_cell(ax: plt.Axes, context: RowPartitionPlotContext) -> plt.Axes:
    """Plot a single cell in a row partition grid."""
    phate_df = context.adata.obsm[context.obs_embedding_key]
    # Resolve colors on the full dataset so every column maps a category to the same color
    colors, is_continuous = to_scatter_colors(
        context.adata.obs_vector(context.color_name),
        context.adata.uns.get(f"{context.color_name}_colors"),
    )

    # Plot background if requested
    if context.plot_background:
        bg_kwargs = {
            k: v for k, v in context.kwargs.items() if k not in ("cmap", "vmin", "vmax")
        }
        ax.scatter(phate_df[:, 0], phate_df[:, 1], c="lightgrey", **bg_kwargs)

    # Determine which data to plot based on column label
    if context.column_label == "ALL":
        plot_df = phate_df
        plot_colors = colors
    else:
        # Use native pandas for filtering
        mask = context.adata.obs[context.obs_search_term] == context.column_label
        label_idxs = np.where(mask)[0]
        plot_df = phate_df[label_idxs, :]
        plot_colors = colors[label_idxs]

    # Handle continuous vs categorical colors
    kwargs = context.kwargs.copy()
    if is_continuous:
        kwargs.setdefault("vmin", np.percentile(plot_colors, 1))
        kwargs.setdefault("vmax", np.percentile(plot_colors, 99))
        kwargs.setdefault("cmap", "rainbow")

    ax.scatter(plot_df[:, 0], plot_df[:, 1], c=plot_colors, **kwargs)

    # Clean up axes
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis("off")
    ax.axis("tight")

    return ax


def plot_row_partitions(
    adata: ad.AnnData,
    obs_search_term: str,
    colors: list | np.ndarray,
    column_labels: Optional[list | np.ndarray] = None,
    obs_embedding_key: str = "X_phate",
    kwargs: Optional[dict] = None,
    plot_all: bool = True,
    plot_background: bool = True,
    unit_size: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot row partitions of the given AnnData object.

    Args:
        adata: The AnnData object containing the data.
        obs_search_term: The search term for selecting the observations.
        colors: Names of the obs columns / var names to color each row by. A
            categorical column (e.g. "phase") is colored with
            ``adata.uns[f"{name}_colors"]`` when present (scanpy convention),
            otherwise a tab10/tab20 palette; a column of literal colors (the
            ``*_colors`` convention, e.g. "phase_colors") is used as-is; numeric
            values are colormapped.
        column_labels: List or array of column labels.
        obs_embedding_key: The key for the observation embedding.
        kwargs: Additional keyword arguments for the plotting function.
        plot_all: Whether to plot all partitions.
        plot_background: Whether to plot the background.
        unit_size: The size of each unit in the plot.
        save_path: The path to save the plot.

    Returns:
        The created figure.
    """
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            raise ValueError(f"{save_dir} does not exist")

    if kwargs is None:
        kwargs = {}

    config = RowPartitionConfig(
        adata=adata,
        obs_search_term=obs_search_term,
        colors=colors,
        column_labels=column_labels,
        obs_embedding_key=obs_embedding_key,
        plot_all=plot_all,
        plot_background=plot_background,
        unit_size=unit_size,
        kwargs=kwargs,
    )

    builder = RowPartitionGridBuilder(
        num_rows=len(colors),
        num_cols=1,  # Will be updated in build()
        unit_size=unit_size,
    )

    fig = builder.build(config, plot_row_partition_cell)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ============================================================================
# Utility Functions
# ============================================================================


NA_COLOR = "lightgrey"


def _default_palette(n_colors: int) -> list:
    """tab10 / tab20 for up to 10 / 20 categories, evenly sampled "turbo" beyond that."""
    if n_colors <= 10:
        return [mpl.colormaps["tab10"](i) for i in range(n_colors)]
    if n_colors <= 20:
        return [mpl.colormaps["tab20"](i) for i in range(n_colors)]
    return [mpl.colormaps["turbo"](i / (n_colors - 1)) for i in range(n_colors)]


def _is_literal_color(value) -> bool:
    """True for strings matplotlib reads as a color, except bare numbers.

    Matplotlib reads "0" or "0.5" as grayscale, but in obs columns those are
    almost always cluster labels.
    """
    if not isinstance(value, str):
        return False
    try:
        float(value)
    except ValueError:
        return mcolors.is_color_like(value)
    return False


def _is_color_column(values) -> bool:
    """True when every non-missing value is a literal color (the ``*_colors`` convention)."""
    present = [v for v in pd.unique(np.asarray(values, dtype=object)) if not pd.isna(v)]
    return bool(present) and all(_is_literal_color(v) for v in present)


def category_color_map(values, palette: Optional[list] = None) -> dict:
    """
    Map each category of ``values`` to a color.

    Categories are taken in ``categories`` order when ``values`` is categorical,
    otherwise in sorted order. ``palette`` (for example ``adata.uns["phase_colors"]``,
    the scanpy convention) is used when it has exactly one color per category;
    otherwise a tab10/tab20 palette is used.

    Args:
        values: Per-cell category labels.
        palette: Optional list of colors, one per category.

    Returns:
        Dict mapping each category to a color.
    """
    if isinstance(getattr(values, "dtype", None), pd.CategoricalDtype):
        categories = list(values.dtype.categories)
    else:
        present = pd.unique(np.asarray(values, dtype=object))
        categories = sorted((v for v in present if not pd.isna(v)), key=str)

    if palette is None or len(palette) != len(categories):
        palette = _default_palette(len(categories))

    return dict(zip(categories, palette))


def to_scatter_colors(values, palette: Optional[list] = None) -> tuple[np.ndarray, bool]:
    """
    Convert per-cell values into something ``Axes.scatter(c=...)`` accepts.

    - Numeric values are returned unchanged and flagged as continuous, so the
      caller can apply a colormap.
    - Values that are all literal colors (the ``*_colors`` obs convention, e.g.
      ``"orchid"`` or ``"#ff0000"``) are used as-is.
    - Any other categorical, string or boolean values get one color per category
      from :func:`category_color_map`.

    Missing values are drawn in light grey.

    Args:
        values: Per-cell values, e.g. ``adata.obs_vector(color_name)``.
        palette: Optional list of colors, one per category, for categorical values.

    Returns:
        Tuple of (colors, is_continuous).
    """
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        values = np.asarray(values)
        dtype = values.dtype
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        return np.asarray(values), True

    labels = np.asarray(values, dtype=object)
    missing = pd.isna(labels)
    if _is_color_column(labels):
        return np.where(missing, NA_COLOR, labels), False

    lookup = {
        label: mcolors.to_rgba(color)
        for label, color in category_color_map(values, palette).items()
    }
    na_rgba = mcolors.to_rgba(NA_COLOR)
    rgba = [na_rgba if is_na else lookup[label] for label, is_na in zip(labels, missing)]
    return np.array(rgba), False


def get_legend(
    adata: ad.AnnData,
    color_name: str,
    label_name: Optional[str] = None,
) -> tuple[list[mpatches.Patch], np.ndarray]:
    """
    Get patches from adata.obs[color_name] for creating a legend.

    ``color_name`` can name either

    - a column of literal colors (the ``*_colors`` convention, e.g. ``"phase_colors"``);
      the labels then come from ``label_name``, which defaults to ``color_name``
      without the ``_colors`` suffix; or
    - a categorical/string column (e.g. ``"phase"``); each category then gets the
      color the plotting functions use for it (``adata.uns[f"{color_name}_colors"]``
      when present, otherwise a tab10/tab20 palette) and ``label_name`` is ignored.

    Args:
        adata: AnnData object.
        color_name: Name of the anndata obs column to use for coloring.
        label_name: The name of the label column (``*_colors`` columns only).

    Returns:
        A tuple containing a list of patches for the legend and the colors array.

    Raises:
        ValueError: If ``color_name`` holds continuous (numeric) values.
    """
    values = adata.obs_vector(color_name)
    palette = adata.uns.get(f"{color_name}_colors")
    colors, is_continuous = to_scatter_colors(values, palette)
    if is_continuous:
        raise ValueError(
            f"Cannot build a legend for continuous values in '{color_name}'"
        )

    if not _is_color_column(values):
        patch_list = [
            mpatches.Patch(color=color, label=str(label))
            for label, color in category_color_map(values, palette).items()
        ]
        return patch_list, colors

    if label_name is None:
        label_name = color_name.removesuffix("_colors")

    labels = adata.obs[label_name].values

    col_lab_array = np.array([colors, labels], dtype=str).T
    uni_col_lab_matches = np.unique(col_lab_array, axis=0)

    patch_list = []
    for color, label in uni_col_lab_matches:
        patch = mpatches.Patch(color=color, label=label)
        patch_list.append(patch)

    return patch_list, colors


def _render_figure(fig: plt.Figure, dpi: int) -> np.ndarray:
    """Render ``fig`` to an RGBA image array."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return plt.imread(buffer)


def combine_figures_with_gridspec(
    figures: list[plt.Figure],
    grid_rows: int,
    grid_cols: int,
    unit_size: int = 5,
    title: Optional[str] = None,
    title_kwargs: Optional[dict] = None,
    save_path: Optional[str] = None,
    raster_dpi: int = 150,
) -> plt.Figure:
    """
    Combine multiple figures into a single figure using GridSpec.

    Each source figure is rendered to an image and drawn into its own grid cell,
    so figures with any content (several axes, labels, legends, colorbars) can be
    combined, and the source figures are left untouched. Matplotlib artists cannot
    be moved between figures, which is why the figures are rasterized rather than
    copied.

    Args:
        figures: List of matplotlib figures to combine.
        grid_rows: Number of rows in the final grid.
        grid_cols: Number of columns in the final grid.
        unit_size: Size of each unit in the plot.
        title: Title for the combined figure.
        title_kwargs: Keyword arguments for the title.
        save_path: Path to save the combined figure.
        raster_dpi: Resolution used to render each source figure.

    Returns:
        Combined figure.
    """
    combined_fig = plt.figure(
        figsize=(unit_size * grid_cols, unit_size * grid_rows),
        constrained_layout=True,
    )

    gs = GridSpec(grid_rows, grid_cols, figure=combined_fig)

    for idx, fig in enumerate(figures[: grid_rows * grid_cols]):
        ax_combined = combined_fig.add_subplot(gs[idx // grid_cols, idx % grid_cols])
        ax_combined.imshow(_render_figure(fig, dpi=raster_dpi))
        ax_combined.axis("off")

    if title:
        if title_kwargs is None:
            title_kwargs = {
                "fontsize": unit_size * grid_cols,
                "fontweight": "bold",
            }
        combined_fig.suptitle(title, **title_kwargs)

    if save_path:
        combined_fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return combined_fig
