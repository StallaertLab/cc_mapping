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
