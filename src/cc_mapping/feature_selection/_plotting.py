"""
Plotting utilities for feature selection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_curve(
    accuracy_curve: np.ndarray,
    optimal_n: int,
    feature_names: np.ndarray,
    threshold: float,
    stable_iterations: int,
    cutoff_method: str,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
    show: bool = True,
) -> plt.Figure:
    """
    Plot accuracy vs number of features for RFMinMaxSelector.
    
    Parameters
    ----------
    accuracy_curve : np.ndarray
        Accuracy at each iteration (index 0 is placeholder).
    optimal_n : int
        Optimal number of features selected.
    feature_names : np.ndarray
        Feature names in order of importance.
    threshold : float
        Threshold used for selection.
    stable_iterations : int
        Number of stable iterations used.
    cutoff_method : str
        Cutoff method used ("increment" or "jump").
    save_path : str or Path, optional
        Path to save the figure.
    figsize : tuple, default=(12, 6)
        Figure size.
    show : bool, default=True
        Whether to display the figure.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_axis = np.arange(len(accuracy_curve))
    
    # Plot accuracy curve
    ax.plot(x_axis, accuracy_curve, "b-", linewidth=2, label="Accuracy")
    
    # Mark optimal point
    ax.axvline(
        optimal_n,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Optimal: {optimal_n} features",
    )
    
    # Mark the optimal point on the curve
    if optimal_n < len(accuracy_curve):
        ax.scatter(
            [optimal_n],
            [accuracy_curve[optimal_n]],
            color="r",
            s=100,
            zorder=5,
        )
    
    # Formatting
    ax.set_title(
        f"Feature Selection: stable_iterations={stable_iterations}, "
        f"threshold={threshold*100:.1f}%, method={cutoff_method}",
        fontsize=12,
    )
    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    
    # Y-axis as percentages
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{int(p*100)}%" for p in np.arange(0, 1.1, 0.1)])
    
    # X-axis with feature names (if not too many)
    max_features_to_show = min(len(accuracy_curve), 50)
    if max_features_to_show <= 30:
        # Show feature names on x-axis
        xtick_labels = [""] + list(feature_names[:max_features_to_show-1])
        ax.set_xticks(range(max_features_to_show))
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
    else:
        # Just show numbers
        ax.set_xlim(0, max_features_to_show)
    
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_feature_importances(
    feature_names: np.ndarray,
    importances: np.ndarray,
    n_features: int = 20,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    show: bool = True,
) -> plt.Figure:
    """
    Plot horizontal bar chart of feature importances.
    
    Parameters
    ----------
    feature_names : np.ndarray
        Feature names (should be sorted by importance).
    importances : np.ndarray
        Importance values (same order as feature_names).
    n_features : int, default=20
        Number of top features to show.
    save_path : str or Path, optional
        Path to save the figure.
    figsize : tuple, default=(10, 8)
        Figure size.
    show : bool, default=True
        Whether to display the figure.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Take top n features
    n_to_show = min(n_features, len(feature_names))
    names = feature_names[:n_to_show][::-1]  # Reverse for horizontal bar
    imps = importances[:n_to_show][::-1]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, imps, align="center", color="steelblue", alpha=0.8)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"Top {n_to_show} Feature Importances", fontsize=12)
    
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    
    # Add value labels on bars
    for bar, imp in zip(bars, imps):
        width = bar.get_width()
        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{imp:.3f}",
            va="center",
            fontsize=8,
        )
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
