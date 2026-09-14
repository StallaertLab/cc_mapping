.. _cc_mapping-feature_selection_ref:

====================================
:mod:`cc_mapping.feature_selection`
====================================

.. automodule:: cc_mapping.feature_selection
   :no-members:
   :no-inherited-members:

Scikit-learn-style selectors that pick the features that best predict a label, with helpers to prepare the data, plot the results and save fitted selectors.



Selectors
---------




.. currentmodule:: cc_mapping.feature_selection

.. autosummary::
   :toctree: generated
   :nosignatures:


   RFMinMaxSelector

   RFTopNSelector

   FeatureSelector

   SelectionResult




Data Preparation
----------------




.. currentmodule:: cc_mapping.feature_selection

.. autosummary::
   :toctree: generated
   :nosignatures:


   prepare_feature_matrix

   validate_data

   PreparedData




Plotting
--------




.. currentmodule:: cc_mapping.feature_selection

.. autosummary::
   :toctree: generated
   :nosignatures:


   plot_accuracy_curve

   plot_feature_importances




Training
--------




.. currentmodule:: cc_mapping.feature_selection

.. autosummary::
   :toctree: generated
   :nosignatures:


   train_rf_model

   TrainingResult
