"""Configuration for the API reference documentation."""


def _get_guide(*refs):
    """Get the rst to refer to user guide."""
    if len(refs) == 1:
        ref_desc = f":ref:`{refs[0]}` section"
    elif len(refs) == 2:
        ref_desc = f":ref:`{refs[0]}` and :ref:`{refs[1]}` sections"
    else:
        ref_desc = ", ".join(f":ref:`{ref}`" for ref in refs[:-1])
        ref_desc += f", and :ref:`{refs[-1]}` sections"

    return f"**User guide.** See the {ref_desc} for further details."


API_REFERENCE = {
    "cc_mapping.thresholding": {
        "short_summary": "Thresholding tools for cell cycle analysis.",
        "description": "Tools for Gaussian Mixture Model-based thresholding to identify cell cycle phases.",
        "sections": [
            {
                "title": "Classes",
                "autosummary": [
                    "GMMThresholding",
                    "SequentialGMM",
                ],
            },
        ],
    },
    "cc_mapping.manifold": {
        "short_summary": "PHATE dimensionality reduction and visualization.",
        "description": "Tools for computing PHATE embeddings and performing hyperparameter searches.",
        "sections": [
            {
                "title": "Configuration",
                "autosummary": [
                    "PHATEConfig",
                    "PHATEHyperparamGrid",
                    "PHATEPlotContext",
                ],
            },
            {
                "title": "Classes",
                "autosummary": [
                    "PHATERunner",
                    "PHATEVisualizer",
                ],
            },
            {
                "title": "Functions",
                "autosummary": [
                    "run_phate",
                    "add_phate_to_adata",
                    "quick_phate_plot",
                    "perform_phate_hyperparameter_search",
                    "plot_phate_hyperparam_cell",
                ],
            },
        ],
    },
    "cc_mapping.plot": {
        "short_summary": "Plotting utilities for grid-based visualizations.",
        "description": "Grid layout builders and plotting functions for embedding visualizations.",
        "sections": [
            {
                "title": "Configuration",
                "autosummary": [
                    "PlotConfig",
                    "RowPartitionConfig",
                    "HyperparamGridConfig",
                    "RowPartitionPlotContext",
                ],
            },
            {
                "title": "Grid Builders",
                "autosummary": [
                    "GridLabelRenderer",
                    "GridLayoutBuilder",
                    "RowPartitionGridBuilder",
                    "HyperparamGridBuilder",
                ],
            },
            {
                "title": "Functions",
                "autosummary": [
                    "plot_row_partitions",
                    "plot_row_partition_cell",
                    "get_legend",
                    "combine_figures_with_gridspec",
                ],
            },
        ],
    },
    "cc_mapping.core": {
        "short_summary": "Random forest feature selection.",
        "description": "Random forest-based feature selection for identifying optimal feature sets.",
        "sections": [
            {
                "title": "Functions",
                "autosummary": [
                    "train_random_forest_model",
                    "random_forest_feature_selection",
                ],
            },
        ],
    },
    "cc_mapping.utils": {
        "short_summary": "General utility functions.",
        "description": "General-purpose utility functions for data manipulation and analysis.",
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "create_boolean_label_combination",
                ],
            },
        ],
    },
}
