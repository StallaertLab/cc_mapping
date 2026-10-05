Single Feature Thresholding
===========================

Use :class:`~cc_mapping.thresholding.GMMThresholding` to threshold a single marker.

When to Use
-----------

Use single feature thresholding when you want to:

* Categorize cells based on one marker (e.g., PCNA levels)
* Identify cell populations with distinct expression levels
* Perform quality control on individual features

Step-by-Step Guide
------------------

1. Import and Setup
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import scanpy as sc
    from cc_mapping.thresholding import GMMThresholding

    # Load your data
    adata = sc.read_h5ad("your_data.h5ad")

2. Create the Thresholder
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize with default parameters
    gmm = GMMThresholding()

    # Or customize parameters
    gmm = GMMThresholding(
        gmm_kwargs={'n_components': 3, 'random_state': 42}
    )

3. Fit the Model
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fit on your feature of interest
    gmm.fit(adata, feature_key="PCNA")

4. Visualize Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create comprehensive visualization
    gmm.plot()

    # Or use specific plots
    gmm.plot_hist_distribution_with_boundaries()
    gmm.plot_bayesian_information_criterion_curve()

5. Access Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get categorized cells
    categories = adata.obs["PCNA_categories"]

    # Get thresholds
    thresholds = gmm.return_thresholds()

    # Get the updated AnnData object
    adata_result = gmm.return_adata()

Advanced Options
----------------

Manual Thresholds
~~~~~~~~~~~~~~~~~

If you want to set custom thresholds instead of using GMM:

.. code-block:: python

    gmm = GMMThresholding(
        manual_decision_boundaries=[100, 500]
    )
    gmm.fit(adata, feature_key="PCNA")

Customizing GMM
~~~~~~~~~~~~~~~

Control the Gaussian Mixture Model behavior:

.. code-block:: python

    gmm = GMMThresholding(
        gmm_kwargs={
            'n_components': 4,  # Try 4 components
            'covariance_type': 'full',
            'random_state': 42
        }
    )

Complete Example
----------------

See the :doc:`../tutorials/single_thresholding` tutorial for a complete workflow with real data.

API Reference
-------------

For detailed parameter descriptions, see:

* :class:`~cc_mapping.thresholding.GMMThresholding`
