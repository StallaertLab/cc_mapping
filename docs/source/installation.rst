.. _installation:

============
Installation
============

Requirements
------------

* Python 3.10+
* numpy
* pandas
* scikit-learn
* matplotlib
* anndata
* pydantic

Install from PyPI
-----------------

Once published, you can install cc-mapping using pip::

    pip install cc-mapping

The PHATE-based cell cycle map functions in ``cc_mapping.manifold`` need the
optional ``phate`` dependency::

    pip install "cc-mapping[manifold]"

Saving and loading fitted selectors from ``cc_mapping.feature_selection``
needs the optional ``skops`` dependency::

    pip install "cc-mapping[persistence]"

Install from Source
-------------------

Clone the repository and install with Poetry::

    git clone https://github.com/StallaertLab/cc_mapping.git
    cd cc_mapping
    poetry install

Or with pip::

    git clone https://github.com/StallaertLab/cc_mapping.git
    cd cc_mapping
    pip install -e .

Development Installation
------------------------

For development with testing and documentation dependencies::

    poetry install --with test,docs

Verify Installation
-------------------

To verify the installation, run::

    python -c "from cc_mapping.thresholding import GMMThresholding; print('Success!')"
