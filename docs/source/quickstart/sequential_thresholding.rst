Sequential Thresholding
=======================

Use :class:`~cc_mapping.thresholding.SequentialGMM` to threshold multiple markers in sequence.

When to Use
-----------

Use sequential thresholding when you want to:

* Analyze cell cycle phases using multiple markers (e.g., PCNA → CDK2 → Geminin)
* Apply hierarchical filtering based on sequential expression patterns
* Refine cell populations through multi-step thresholding

How It Works
------------

Sequential thresholding applies GMM thresholding to multiple features **in order**:

1. Threshold the first feature (e.g., PCNA)
2. Within each category, threshold the second feature (e.g., CDK2)
3. Continue for additional features
4. Optionally refine with additional manual thresholds

Step-by-Step Guide
------------------

1. Import and Setup
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import scanpy as sc
    from cc_mapping.thresholding import SequentialGMM

    # Load your data
    adata = sc.read_h5ad("your_data.h5ad")

2. Create the Sequential Thresholder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize
    seq_gmm = SequentialGMM()

    # Or customize
    seq_gmm = SequentialGMM(
        gmm_kwargs={'random_state': 42},
        thresholding_events_key="cc_phases"
    )

3. Define Feature Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Features to threshold in order
    features = ["PCNA", "CDK2", "Geminin"]

4. Fit the Model
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fit sequentially
    seq_gmm.fit(adata, feature_keys=features)

5. Visualize Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Visualize all thresholding steps
    seq_gmm.plot()

    # Or visualize specific features
    seq_gmm.plot_hist_distribution_with_boundaries(feature_key="PCNA")

6. Access Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get final cell categorization
    categories = adata.obs[seq_gmm.thresholding_events_key]

    # Get the updated AnnData object
    adata_result = seq_gmm.return_adata()

Advanced Options
----------------

Refining with Manual Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add manual thresholds after GMM thresholding:

.. code-block:: python

    # After fitting with GMM
    seq_gmm.refine_labels_with_manual_thresholds(
        adata,
        feature_key="additional_marker",
        manual_thresholds=[50, 200]
    )

Refining with Additional GMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply additional GMM thresholding to specific populations:

.. code-block:: python

    # Refine a specific population
    seq_gmm.refine_labels_with_gmm(
        adata,
        feature_key="refinement_marker",
        parent_label="High_PCNA"
    )

Threshold Entire Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

Apply thresholds to a new dataset using learned parameters:

.. code-block:: python

    # After training on one dataset
    seq_gmm.threshold_entire_dataset(new_adata, features)

Complete Example
----------------

See the :doc:`../tutorials/sequential_thresholding` tutorial for a complete workflow with real data.

API Reference
-------------

For detailed parameter descriptions, see:

* :class:`~cc_mapping.thresholding.SequentialGMM`
