Quick Start Guide
=================

This guide will help you get started with the thresholding module.

Basic Usage
-----------

Single Feature Thresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~cc_mapping.thresholding.GMMThresholding` for thresholding a single feature:

.. code-block:: python

    import scanpy as sc
    from cc_mapping.thresholding import GMMThresholding

    # Load your data
    adata = sc.read_h5ad("your_data.h5ad")

    # Create thresholder
    gmm = GMMThresholding()

    # Fit the model
    gmm.fit(adata, feature_key="PCNA")

    # Access results
    print(f"Threshold: {gmm.threshold}")
    print(f"Categories: {gmm.categories}")

    # Visualize
    gmm.plot()

Sequential Thresholding
~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~cc_mapping.thresholding.SequentialGMM` for multiple features:

.. code-block:: python

    from cc_mapping.thresholding import SequentialGMM

    # Create sequential thresholder
    seq_gmm = SequentialGMM()

    # Define features to threshold sequentially
    features = ["PCNA", "CDK2", "Geminin"]

    # Fit sequentially
    seq_gmm.fit(adata, feature_keys=features)

    # Visualize all steps
    seq_gmm.plot()

Working with Results
--------------------

The thresholding results are stored in the AnnData object:

.. code-block:: python

    # Access categorized cells
    categories = adata.obs["PCNA_categories"]

    # Filter for high-expressing cells
    high_cells = adata[adata.obs["PCNA_categories"] == "High"]

    # Continue with downstream analysis
    sc.pl.umap(adata, color="PCNA_categories")

Next Steps
----------

* Check out the :doc:`api/index` for detailed API documentation
* See example notebooks in the repository
* Learn about advanced features and customization options
