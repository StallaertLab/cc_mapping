# cc_mapping# Cell Cycle Mapping Package

[![PyPI version](https://badge.fury.io/py/cc-mapping.svg)](https://badge.fury.io/py/cc-mapping)## Step 1: Install Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)From the root directory of this repository:

**Gaussian Mixture Model-based thresholding for single-cell gene expression analysis**```

conda env create -f .\environments\cc_mapping.yml

`cc_mapping` provides robust statistical methods for categorizing cells based on gene expression levels using Gaussian Mixture Models (GMMs). Originally developed for cell cycle analysis, it's applicable to any single-cell RNA-seq thresholding task.```

## Features## Step 2: Update Global Variables

- 🎯 **Automatic thresholding** using GMM-based statistical inferenceDue to the fact this is not an actual package, whenever you want to use it, you will have to tell your computer where to look. You will need to update these two files:

- 📊 **Single & sequential thresholding** for simple or complex categorization schemes

- 🔄 **Sequential refinement** to progressively narrow down cell populations* cc_mapping\GLOBAL_VARIABLES\GLOBAL_VARIABLES.py

- 📈 **Built-in visualization** with density plots and QC metrics* notebooks\GLOBAL_VARIABLES\GLOBAL_VARIABLES.py

- 🧪 **AnnData integration** for seamless single-cell workflow compatibility

- ⚙️ **Flexible configuration** with manual threshold overrides when neededReplace the variable 'cc_mapping_package_dir' with the path to the root directory for the cc_mapping repository.

## InstallationThis means that if you want to use the cc_mapping package in another folder, you should copy this GLOBAL VARIABLES folder into that directory and add this to the imports of your python scripts

Install from PyPI using pip:```

import sys

````bashsys.path.append(os.getcwd())

pip install cc-mapping

```from GLOBAL_VARIABLES.GLOBAL_VARIABLES import cc_mapping_package_dir

sys.path.append(cc_mapping_package_dir)

Or using Poetry:```



```bash## Step 3: Use the cc_mapping package!

poetry add cc-mapping

```There is a test notebook in the notebooks directory of this repository that can be used as an example.

## Quick Start

```python
import anndata as ad
from cc_mapping import GMMThresholding

# Load your AnnData object
adata = ad.read_h5ad('your_data.h5ad')

# Initialize thresholding for a gene
gmm = GMMThresholding(
    adata=adata,
    feature='PCNA',  # Gene name
    label_obs_save_str='PCNA_categories'
)

# Fit GMM with automatic component selection
gmm.fit(n_components=2)

# Categorize cells
gmm.categorize_samples(ordered_labels=['Low', 'High'])

# Get updated AnnData with new categories
adata = gmm.return_adata()

# Visualize results
fig = gmm.plot_density()
fig.savefig('pcna_thresholding.png')
````

## Sequential Thresholding

For more complex categorization schemes (e.g., cell cycle phases):

```python
from cc_mapping import SequentialGMM

# Initialize sequential thresholding
seq_gmm = SequentialGMM(
    adata=adata,
    features=["PCNA", "CDK1"],
    parent_labels=["All"],
    ordered_labels_list=[["PCNA-", "PCNA+"], ["CDK1-", "CDK1+"]],
)

# Run sequential refinement
seq_gmm.fit_all(n_components_list=[2, 2])
adata = seq_gmm.return_adata()

# Collapse labels to final categories
seq_gmm.collapse_labels(
    final_labels=["G0", "G1", "S", "G2M"],
    collapse_map={
        "PCNA-_CDK1-": "G0",
        "PCNA+_CDK1-": "G1",
        "PCNA+_CDK1+": ["S", "G2M"],
    },
)
```

## Boolean Label Operations

Combine categorical observations with boolean logic:

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

## Documentation

For detailed documentation, tutorials, and API reference, visit [our documentation](https://github.com/StallaertLab/cc_mapping).

### Examples

- **Single thresholding**: See `notebooks/Single_Thresholding_Workflow.ipynb`
- **Sequential thresholding**: See `notebooks/Sequential_Thresholding_Workflow.ipynb`
- **CSV to AnnData**: See `notebooks/CSV_to_Anndata.ipynb`

## Requirements

- Python ≥ 3.10
- AnnData ≥ 0.10.0
- NumPy < 2.0.0
- scikit-learn ≥ 1.3.0
- pandas ≥ 2.0.0
- matplotlib ≥ 3.7.0
- scipy ≥ 1.11.0
- pydantic ≥ 2.10.0

## Citation

If you use `cc_mapping` in your research, please cite:

```bibtex
@software{cc_mapping,
  author = {Your Name},
  title = {cc_mapping: GMM-based thresholding for single-cell analysis},
  year = {2025},
  url = {https://github.com/StallaertLab/cc_mapping}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/StallaertLab/cc_mapping/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/StallaertLab/cc_mapping/discussions)
- 📧 **Email**: dap182@pitt.edu
