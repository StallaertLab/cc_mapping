Basic Usage
===========

Core Concepts
-------------

The thresholding module works with **AnnData** objects, the standard format for single-cell analysis.

Data Requirements
~~~~~~~~~~~~~~~~~

Your AnnData object should contain:

* **Feature data**: Protein or RNA measurements in ``adata.obs`` or ``adata.X``
* **Cell observations**: Each row represents a cell
* **Feature keys**: Column names identifying your markers (e.g., "PCNA", "CDK2")

Basic Workflow
--------------

1. **Load your data** as an AnnData object
2. **Choose a thresholder**:
   
   - :class:`~cc_mapping.thresholding.GMMThresholding` for single features
   - :class:`~cc_mapping.thresholding.SequentialGMM` for multiple features

3. **Fit the model** with your data and feature key
4. **Visualize results** using built-in plotting methods
5. **Access categorized cells** from ``adata.obs``

Quick Example
-------------

.. code-block:: python

    import scanpy as sc
    from cc_mapping.thresholding import GMMThresholding
    
    # Load your data
    adata = sc.read_h5ad("your_data.h5ad")
    
    # Create and fit thresholder
    gmm = GMMThresholding()
    gmm.fit(adata, feature_key="PCNA")
    
    # Visualize
    gmm.plot()
    
    # Access results
    print(adata.obs["PCNA_categories"].value_counts())

Understanding the Output
------------------------

After fitting, the thresholder adds several columns to your AnnData object:

* ``{feature}_categories``: Categorical labels (e.g., "Low", "Medium", "High")
* ``{feature}_threshold``: The computed threshold values
* Model parameters are stored in ``adata.uns``

Next Steps
----------

* :doc:`single_thresholding` - Learn single feature thresholding in detail
* :doc:`sequential_thresholding` - Learn sequential thresholding for multiple markers
* :doc:`../tutorials/index` - Work through complete examples
