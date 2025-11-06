"""
Tests for utility functions in cc_mapping.thresholding.utils

Tests:
    - create_boolean_label_combination with AND, OR, XOR operators
    - generate_thresholding_report in text and dataframe formats
    - Error handling and validation
"""

import numpy as np
import pandas as pd
import pytest
from collections import OrderedDict

import anndata as ad

from cc_mapping.utils import create_boolean_label_combination
from cc_mapping.thresholding import GMMThresholding


@pytest.fixture
def adata_with_labels():
    """Create test AnnData with two categorical labels."""
    adata = ad.AnnData(X=np.random.randn(100, 10))
    
    # Create two categorical labels
    adata.obs['treatment'] = pd.Categorical(['control'] * 50 + ['drug'] * 50)
    adata.obs['cell_cycle'] = pd.Categorical(
        ['G0'] * 25 + ['G1'] * 25 + ['S'] * 25 + ['G2'] * 25
    )
    
    return adata


@pytest.fixture
def gmm_with_thresholding(sample_adata):
    """Create test GMMThresholding instance with completed thresholding."""
    # Use the sample_adata and perform thresholding
    gmm = GMMThresholding(
        adata=sample_adata,
        feature='gene1',
        label_obs_save_str='cell_cycle',
        thresholding_events_key='gmm_thresholding_events'
    )
    gmm.fit(n_components=2)
    gmm.categorize_samples(ordered_labels=['Low', 'High'])
    gmm.return_adata()  # This saves the operation to uns
    
    return gmm


# ===== Tests for create_boolean_label_combination =====

class TestCreateBooleanLabelCombination:
    """Tests for create_boolean_label_combination function."""
    
    def test_and_operator(self, adata_with_labels):
        """Test AND operator combines labels correctly."""
        adata = create_boolean_label_combination(
            adata_with_labels,
            obs_key_1='treatment',
            match_values_1=['control'],
            obs_key_2='cell_cycle',
            match_values_2=['G0'],
            operator='AND',
            output_obs_key='control_G0',
            true_label='yes',
            false_label='no',
        )
        
        # Check output exists
        assert 'control_G0' in adata.obs.columns
        
        # Check correct cells are labeled
        control_mask = adata.obs['treatment'] == 'control'
        g0_mask = adata.obs['cell_cycle'] == 'G0'
        expected_mask = control_mask & g0_mask
        
        actual_positive = adata.obs['control_G0'] == 'yes'
        assert np.array_equal(expected_mask, actual_positive)
        
        # Check it's categorical
        assert isinstance(adata.obs['control_G0'].dtype, pd.CategoricalDtype)
    
    def test_or_operator(self, adata_with_labels):
        """Test OR operator combines labels correctly."""
        adata = create_boolean_label_combination(
            adata_with_labels,
            obs_key_1='treatment',
            match_values_1=['control'],
            obs_key_2='cell_cycle',
            match_values_2=['G0', 'G1'],
            operator='OR',
            output_obs_key='control_or_G0G1',
            true_label='positive',
            false_label='negative',
        )
        
        # Check correct cells are labeled
        control_mask = adata.obs['treatment'] == 'control'
        g0g1_mask = adata.obs['cell_cycle'].isin(['G0', 'G1'])
        expected_mask = control_mask | g0g1_mask
        
        actual_positive = adata.obs['control_or_G0G1'] == 'positive'
        assert np.array_equal(expected_mask, actual_positive)
    
    def test_xor_operator(self, adata_with_labels):
        """Test XOR operator combines labels correctly."""
        adata = create_boolean_label_combination(
            adata_with_labels,
            obs_key_1='treatment',
            match_values_1=['control'],
            obs_key_2='cell_cycle',
            match_values_2=['G0'],
            operator='XOR',
            output_obs_key='xor_result',
            true_label='exactly_one',
            false_label='both_or_neither',
        )
        
        # Check correct cells are labeled
        control_mask = adata.obs['treatment'] == 'control'
        g0_mask = adata.obs['cell_cycle'] == 'G0'
        expected_mask = control_mask ^ g0_mask
        
        actual_positive = adata.obs['xor_result'] == 'exactly_one'
        assert np.array_equal(expected_mask, actual_positive)
    
    def test_multiple_values_per_label(self, adata_with_labels):
        """Test with multiple values in each label."""
        adata = create_boolean_label_combination(
            adata_with_labels,
            obs_key_1='treatment',
            match_values_1=['control', 'drug'],
            obs_key_2='cell_cycle',
            match_values_2=['G0', 'G1'],
            operator='AND',
            output_obs_key='combined',
            true_label='yes',
            false_label='no',
        )
        
        # All cells should be positive (all treatments AND first two phases)
        expected_count = 50  # 25 G0 + 25 G1
        actual_count = (adata.obs['combined'] == 'yes').sum()
        assert actual_count == expected_count
    
    def test_case_insensitive_operator(self, adata_with_labels):
        """Test that operator is case-insensitive."""
        adata = create_boolean_label_combination(
            adata_with_labels,
            obs_key_1='treatment',
            match_values_1=['control'],
            obs_key_2='cell_cycle',
            match_values_2=['G0'],
            operator='and',  # lowercase
            output_obs_key='test',
            true_label='yes',
            false_label='no',
        )
        
        assert 'test' in adata.obs.columns
    
    def test_error_label1_not_found(self, adata_with_labels):
        """Test error when label1 doesn't exist."""
        with pytest.raises(KeyError, match="obs_key_1 'nonexistent' not found"):
            create_boolean_label_combination(
                adata_with_labels,
                obs_key_1='nonexistent',
                match_values_1=['control'],
                obs_key_2='cell_cycle',
                match_values_2=['G0'],
                operator='AND',
                output_obs_key='test',
                true_label='yes',
                false_label='no',
            )
    
    def test_error_label2_not_found(self, adata_with_labels):
        """Test error when label2 doesn't exist."""
        with pytest.raises(KeyError, match="obs_key_2 'nonexistent' not found"):
            create_boolean_label_combination(
                adata_with_labels,
                obs_key_1='treatment',
                match_values_1=['control'],
                obs_key_2='nonexistent',
                match_values_2=['G0'],
                operator='AND',
                output_obs_key='test',
                true_label='yes',
                false_label='no',
            )
    
    def test_error_invalid_operator(self, adata_with_labels):
        """Test error with invalid operator."""
        with pytest.raises(ValueError, match="operator must be one of"):
            create_boolean_label_combination(
                adata_with_labels,
                obs_key_1='treatment',
                match_values_1=['control'],
                obs_key_2='cell_cycle',
                match_values_2=['G0'],
                operator='INVALID',
                output_obs_key='test',
                true_label='yes',
                false_label='no',
            )
    
    def test_error_output_label_exists(self, adata_with_labels):
        """Test error when output_label already exists."""
        with pytest.raises(KeyError, match="output_obs_key 'treatment' already exists"):
            create_boolean_label_combination(
                adata_with_labels,
                obs_key_1='treatment',
                match_values_1=['control'],
                obs_key_2='cell_cycle',
                match_values_2=['G0'],
                operator='AND',
                output_obs_key='treatment',  # Already exists
                true_label='yes',
                false_label='no',
            )
    
    def test_error_label1_values_not_list(self, adata_with_labels):
        """Test error when label1_values is not a list."""
        with pytest.raises(TypeError, match="match_values_1 must be a list"):
            create_boolean_label_combination(
                adata_with_labels,
                obs_key_1='treatment',
                match_values_1='control',  # String instead of list
                obs_key_2='cell_cycle',
                match_values_2=['G0'],
                operator='AND',
                output_obs_key='test',
                true_label='yes',
                false_label='no',
            )
    
    def test_error_value_not_in_label1(self, adata_with_labels):
        """Test error when value doesn't exist in label1."""
        with pytest.raises(ValueError, match="Value 'nonexistent' not found in obs_key_1"):
            create_boolean_label_combination(
                adata_with_labels,
                obs_key_1='treatment',
                match_values_1=['nonexistent'],
                obs_key_2='cell_cycle',
                match_values_2=['G0'],
                operator='AND',
                output_obs_key='test',
                true_label='yes',
                false_label='no',
            )


# ===== Tests for generate_thresholding_report =====

class TestGenerateThresholdingReport:
    """Tests for generate_thresholding_report method."""
    
    def test_text_format_basic(self, gmm_with_thresholding):
        """Test basic text format report."""
        report = gmm_with_thresholding.generate_thresholding_report(output_format='text')
        
        # Check report is a string
        assert isinstance(report, str)
        
        # Check key elements are present
        assert 'Thresholding Report' in report
        assert 'gene1' in report  # Feature name
        assert 'cell_cycle' in report  # Obs label
        assert 'Components: 2' in report
        assert 'Low' in report
        assert 'High' in report
    
    def test_dataframe_format_basic(self, gmm_with_thresholding):
        """Test basic dataframe format report."""
        report = gmm_with_thresholding.generate_thresholding_report(output_format='dataframe')
        
        # Check report is a DataFrame
        assert isinstance(report, pd.DataFrame)
        
        # Check expected columns exist
        expected_cols = ['Operation', 'Type', 'Feature', 'Layer', 'Obs Label',
                        'Components', 'Thresholds', 'Labels', 'Parent', 
                        'Refined From', 'Total Cells']
        assert all(col in report.columns for col in expected_cols)
        
        # Check row count
        assert len(report) == 1
        
        # Check values
        assert report['Feature'].iloc[0] == 'gene1'
        assert report['Type'].iloc[0] == 'standard'
        assert report['Components'].iloc[0] == '2'
    
    def test_text_format_with_refinement(self, sample_adata):
        """Test text report with refinement operation using SequentialGMM."""
        from cc_mapping.thresholding import SequentialGMM
        
        # Create sequential instance
        seq_gmm = SequentialGMM(
            adata=sample_adata,
            thresholding_events_key='test_events'
        )
        
        # First threshold
        seq_gmm.threshold_entire_dataset(
            feature='gene1',
            label_obs_save_str='phase',
            n_components=2,
            ordered_labels=['Low', 'High'],
            operation_name='first_threshold'
        )
        
        # Refine one of the labels
        seq_gmm.refine_labels_with_gmm(
            feature='gene2',
            obs_label='phase',
            value_to_refine='Low',
            n_components=2,
            ordered_labels=['Low_A', 'Low_B'],
            operation_name='refine_low'
        )
        
        report = seq_gmm.generate_thresholding_report(output_format='text')
        
        # Check refinement info is present
        assert 'refine_low' in report
        assert 'Refinement' in report
        assert 'first_threshold' in report  # Parent
        assert 'Low' in report  # Refined from
    
    def test_empty_events(self, sample_adata):
        """Test report with no operations."""
        gmm = GMMThresholding(
            adata=sample_adata,
            feature='gene1',
            label_obs_save_str='labels',
            thresholding_events_key='empty_events'
        )
        
        report = gmm.generate_thresholding_report(output_format='text')
        
        assert isinstance(report, str)
        assert report == "No thresholding operations found."
    
    def test_empty_events_dataframe(self, sample_adata):
        """Test dataframe report with no operations."""
        gmm = GMMThresholding(
            adata=sample_adata,
            feature='gene1',
            label_obs_save_str='labels',
            thresholding_events_key='empty_events'
        )
        
        report = gmm.generate_thresholding_report(output_format='dataframe')
        
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 0
    
    def test_error_key_not_found(self, sample_adata):
        """Test error when thresholding_events_key doesn't exist."""
        # Don't create the key at all - GMMThresholding __init__ creates it
        # So we need to delete it after creation
        gmm = GMMThresholding(
            adata=sample_adata,
            feature='gene1',
            label_obs_save_str='labels',
            thresholding_events_key='nonexistent'
        )
        
        # Delete the key that was auto-created
        del gmm.adata.uns['nonexistent']
        
        with pytest.raises(KeyError, match="thresholding_events_key 'nonexistent' not found"):
            gmm.generate_thresholding_report(output_format='text')
    
    def test_error_invalid_format(self, gmm_with_thresholding):
        """Test error with invalid output_format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            gmm_with_thresholding.generate_thresholding_report(output_format='invalid')
    
    def test_error_not_dict(self, sample_adata):
        """Test error when uns key is not a dict."""
        # Create a GMM instance and manually corrupt the uns key
        gmm = GMMThresholding(
            adata=sample_adata,
            feature='gene1',
            label_obs_save_str='labels',
            thresholding_events_key='corrupt_events'
        )
        gmm.adata.uns['corrupt_events'] = "not a dict"
        
        with pytest.raises(TypeError, match="must be a dict or OrderedDict"):
            gmm.generate_thresholding_report(output_format='text')
    
    def test_cell_counts_with_valid_obs(self, gmm_with_thresholding):
        """Test that cell counts are calculated when obs column exists."""
        report = gmm_with_thresholding.generate_thresholding_report(output_format='text')
        
        # Check cell counts are present
        assert 'Cell counts:' in report
        # The actual counts will vary, just check the format is there
        assert 'Low=' in report or 'High=' in report
    
    def test_manual_thresholds_handling(self, sample_adata):
        """Test report handles manual thresholds (no GMM info)."""
        from cc_mapping.thresholding import SequentialGMM
        
        seq_gmm = SequentialGMM(
            adata=sample_adata,
            thresholding_events_key='manual_events'
        )
        
        # Use manual thresholds
        seq_gmm.threshold_entire_dataset(
            feature='gene1',
            label_obs_save_str='labels',
            n_components=2,
            ordered_labels=['Low', 'High'],
            manual_thresholds=[1.0],
            operation_name='manual_op'
        )
        
        report = seq_gmm.generate_thresholding_report(output_format='text')
        
        # When manual thresholds are used, it still shows the n_components
        assert 'Components: 2' in report
        assert 'Thresholds: [1.0000]' in report
