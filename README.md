# cc_mapping

[![PyPI version](https://img.shields.io/pypi/v/cc-mapping.svg)](https://pypi.org/project/cc-mapping/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/cc-mapping/badge/?version=latest)](https://cc-mapping.readthedocs.io)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/StallaertLab/cc_mapping/blob/main/LICENSE)

**Gaussian Mixture Model-based thresholding and cell cycle mapping for single-cell data**

`cc_mapping` categorizes cells by fitting Gaussian Mixture Models (GMMs) to
single-cell feature measurements stored in an [AnnData](https://anndata.readthedocs.io)
object. It was developed for cell cycle analysis, and it works for any
single-cell feature you want to split into populations (for example, low and high).

## Features

- **Single-feature thresholding**: fit a GMM to one feature and label each cell from the fitted decision boundaries
- **Sequential thresholding**: refine one label at a time with further features, like a gating strategy
- **Manual overrides**: set thresholds by hand when you need to
- **Plots**: histograms and strip plots with the fitted components and decision boundaries
- **Feature selection**: random forest selectors that find the features that best separate your labels
- **Cell cycle maps**: PHATE embeddings of the selected features (optional dependency)
- **AnnData throughout**: labels go to `adata.obs`, and the fitted models and thresholds go to `adata.uns`

## Installation

```bash
pip install cc-mapping
```

Optional extras:

```bash
pip install "cc-mapping[manifold]"      # PHATE, for cell cycle maps
pip install "cc-mapping[persistence]"   # skops, to save and load fitted feature selectors
```

To install from source:

```bash
git clone https://github.com/StallaertLab/cc_mapping.git
cd cc_mapping
pip install -e .
```

## Quick start

```python
import anndata as ad
from cc_mapping import GMMThresholding

adata = ad.read_h5ad("your_data.h5ad")

# Threshold one feature (a name in adata.var_names)
gmm = GMMThresholding(
    adata=adata,
    feature="PCNA",
    label_obs_save_str="PCNA_category",
)

# Fit a two-component GMM and label the cells
gmm.fit(n_components=2)
gmm.categorize_samples(ordered_labels=["Low", "High"])

# Plot the histogram with the fitted components and the decision boundary
gmm.plot_hist_distribution_with_boundaries(save_path="pcna_thresholding.png")

# Get a copy of the AnnData with the labels in adata.obs["PCNA_category"]
adata = gmm.return_adata()
```

`GMMThresholding` works on a copy of your AnnData, so call `return_adata()`
to get the labelled result.

To set the threshold yourself, pass `manual_thresholds`. No fit is needed:

```python
gmm.categorize_samples(ordered_labels=["Low", "High"], manual_thresholds=[0.05])
```

## Sequential thresholding

`SequentialGMM` thresholds the whole dataset once, then refines one label at
a time using other features. Each step has a name and is recorded in `adata.uns`.

```python
from cc_mapping import SequentialGMM

seq_gmm = SequentialGMM(adata=adata)

# Step 1: split every cell on DNA content
seq_gmm.threshold_entire_dataset(
    feature="DNA_content",
    label_obs_save_str="cell_cycle",
    n_components=2,
    ordered_labels=["2N", "4N"],
    operation_name="DNA_threshold",
)

# Step 2: split only the 2N cells on pRB
seq_gmm.refine_labels_with_gmm(
    feature="pRB",
    obs_label="cell_cycle",
    value_to_refine="2N",
    n_components=2,
    ordered_labels=["G0", "G1"],
    operation_name="pRB_refinement",
)

adata = seq_gmm.return_adata()
```

Use `refine_labels_with_manual_thresholds` to refine a label with thresholds
you choose.

## Combining labels

`create_boolean_label_combination` combines two categorical `adata.obs`
columns with boolean logic:

```python
from cc_mapping import create_boolean_label_combination

adata = create_boolean_label_combination(
    adata=adata,
    obs_key_1="treatment",
    match_values_1=["control"],
    obs_key_2="cell_cycle",
    match_values_2=["G0"],
    operator="AND",
    output_obs_key="control_G0",
    true_label="control_G0",
    false_label="other",
)
```

## Feature selection and cell cycle maps

`cc_mapping.feature_selection` selects the features that best separate a set
of labels, and `cc_mapping.manifold` embeds the cells on those features with PHATE:

```python
from cc_mapping.feature_selection import RFMinMaxSelector
from cc_mapping.manifold import run_phate

# Find the smallest feature set that keeps classification accuracy high
selector = RFMinMaxSelector(threshold=0.05, stable_iterations=5)
selector.fit_adata(adata, adata.var_names, "cell_cycle")
selector.transform_adata(adata, var_key="selected_features")

# Embed the cells with PHATE (needs the "manifold" extra)
adata = run_phate(adata, feature_set="selected_features", layer="z_score")
# coordinates are in adata.obsm["X_phate"]
```

`run_phate` reads from `adata.layers`, so `layer` must name a layer in your AnnData.

## Documentation

The full documentation, tutorials and API reference are at
[cc-mapping.readthedocs.io](https://cc-mapping.readthedocs.io).

Example notebooks are in [`notebooks/`](https://github.com/StallaertLab/cc_mapping/tree/main/notebooks):

- [Single thresholding](https://github.com/StallaertLab/cc_mapping/blob/main/notebooks/Single_Thresholding_Workflow.ipynb)
- [Sequential thresholding](https://github.com/StallaertLab/cc_mapping/blob/main/notebooks/Sequential_Thresholding_Workflow.ipynb)
- [CSV to AnnData](https://github.com/StallaertLab/cc_mapping/blob/main/notebooks/CSV_to_Anndata.ipynb)
- [Cell cycle maps](https://github.com/StallaertLab/cc_mapping/blob/main/notebooks/Cell_Cycle_Map_Workflow.ipynb) (random forest feature selection and PHATE)

## Requirements

Python 3.10 or later. `pip` installs the dependencies: numpy, pandas, scipy,
scikit-learn, matplotlib, anndata, pydantic, kneed, mpl-scatter-density and tqdm.

## Citation

If you use `cc_mapping` in your research, please cite this repository:
<https://github.com/StallaertLab/cc_mapping>

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

`cc_mapping` is released under the BSD 3-Clause License. See
[LICENSE](https://github.com/StallaertLab/cc_mapping/blob/main/LICENSE).

## Support

- **Bug reports and questions**: [GitHub Issues](https://github.com/StallaertLab/cc_mapping/issues)
- **Email**: dap182@pitt.edu
