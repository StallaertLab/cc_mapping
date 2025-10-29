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

from cc_mapping.thresholding.utils import (
    create_boolean_label_combination,
    generate_thresholding_report,
)


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
def adata_with_thresholding_metadata():
    """Create test AnnData with thresholding metadata."""
    adata = ad.AnnData(X=np.random.randn(100, 10))
    
    # Add some labels
    adata.obs['cell_cycle'] = pd.Categorical(['G0'] * 40 + ['G1'] * 30 + ['S'] * 30)
    
    # Add thresholding metadata
    adata.uns['gmm_thresholding_events'] = OrderedDict([
        ('DNA_thresholding', {
            'feature_name': 'DNA_content',
            'gmm_obs_label': 'cell_cycle',
            'layer': None,
            'gmm_info': {
                'gmm_kwargs': {},
                'means': [0.01, 0.03, 0.05],
                'covs': [0.001, 0.001, 0.001],
                'weights': [0.4, 0.3, 0.3],
                'n_components': 3,
                'data_probs': None,
                'condensed_data_probs': None,
            },
            'ordered_gmm_labels': ['G0', 'G1', 'S'],
            'decision_boundaries': {'thresholds': [0.023, 0.045]},
            'condensed_labels': None,
            'operation_type': 'standard',
            'parent_operation': None,
            'refined_from_labels': None,
        }),
    ])
    
    return adata


# ===== Tests for create_boolean_label_combination =====

class TestCreateBooleanLabelCombination:
    """Tests for create_boolean_label_combination function."""
    
    def test_and_operator(self, adata_with_labels):
        """Test AND operator combines labels correctly."""
        adata = create_boolean_label_combination(
            adata_with_labels,
            label1='treatment',
            label1_values=['control'],
            label2='cell_cycle',
            label2_values=['G0'],
            operator='AND',
            output_label='control_G0',
            positive_label='yes',
            negative_label='no',
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
            label1='treatment',
            label1_values=['control'],
            label2='cell_cycle',
            label2_values=['G0', 'G1'],
            operator='OR',
            output_label='control_or_G0G1',
            positive_label='positive',
            negative_label='negative',
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
            label1='treatment',
            label1_values=['control'],
            label2='cell_cycle',
            label2_values=['G0'],
            operator='XOR',
            output_label='xor_result',
            positive_label='exactly_one',
            negative_label='both_or_neither',
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
            label1='treatment',
            label1_values=['control', 'drug'],
            label2='cell_cycle',
            label2_values=['G0', 'G1'],
            operator='AND',
            output_label='combined',
            positive_label='yes',
            negative_label='no',
        )
        
        # All cells should be positive (all treatments AND first two phases)
        expected_count = 50  # 25 G0 + 25 G1
        actual_count = (adata.obs['combined'] == 'yes').sum()
        assert actual_count == expected_count
    
    def test_case_insensitive_operator(self, adata_with_labels):
        """Test that operator is case-insensitive."""
        adata = create_boolean_label_combination(
            adata_with_labels,
            label1='treatment',
            label1_values=['control'],
            label2='cell_cycle',
            label2_values=['G0'],
            operator='and',  # lowercase
            output_label='test',
            positive_label='yes',
            negative_label='no',
        )
        
        assert 'test' in adata.obs.columns
    
    def test_error_label1_not_found(self, adata_with_labels):
        """Test error when label1 doesn't exist."""
        with pytest.raises(KeyError, match="label1 'nonexistent' not found"):
            create_boolean_label_combination(
                adata_with_labels,
                label1='nonexistent',
                label1_values=['control'],
                label2='cell_cycle',
                label2_values=['G0'],
                operator='AND',
                output_label='test',
                positive_label='yes',
                negative_label='no',
            )
    
    def test_error_label2_not_found(self, adata_with_labels):
        """Test error when label2 doesn't exist."""
        with pytest.raises(KeyError, match="label2 'nonexistent' not found"):
            create_boolean_label_combination(
                adata_with_labels,
                label1='treatment',
                label1_values=['control'],
                label2='nonexistent',
                label2_values=['G0'],
                operator='AND',
                output_label='test',
                positive_label='yes',
                negative_label='no',
            )
    
    def test_error_invalid_operator(self, adata_with_labels):
        """Test error with invalid operator."""
        with pytest.raises(ValueError, match="operator must be one of"):
            create_boolean_label_combination(
                adata_with_labels,
                label1='treatment',
                label1_values=['control'],
                label2='cell_cycle',
                label2_values=['G0'],
                operator='INVALID',
                output_label='test',
                positive_label='yes',
                negative_label='no',
            )
    
    def test_error_output_label_exists(self, adata_with_labels):
        """Test error when output_label already exists."""
        with pytest.raises(KeyError, match="output_label 'treatment' already exists"):
            create_boolean_label_combination(
                adata_with_labels,
                label1='treatment',
                label1_values=['control'],
                label2='cell_cycle',
                label2_values=['G0'],
                operator='AND',
                output_label='treatment',  # Already exists
                positive_label='yes',
                negative_label='no',
            )
    
    def test_error_label1_values_not_list(self, adata_with_labels):
        """Test error when label1_values is not a list."""
        with pytest.raises(TypeError, match="label1_values must be a list"):
            create_boolean_label_combination(
                adata_with_labels,
                label1='treatment',
                label1_values='control',  # String instead of list
                label2='cell_cycle',
                label2_values=['G0'],
                operator='AND',
                output_label='test',
                positive_label='yes',
                negative_label='no',
            )
    
    def test_error_value_not_in_label1(self, adata_with_labels):
        """Test error when value doesn't exist in label1."""
        with pytest.raises(ValueError, match="Value 'nonexistent' not found in label1"):
            create_boolean_label_combination(
                adata_with_labels,
                label1='treatment',
                label1_values=['nonexistent'],
                label2='cell_cycle',
                label2_values=['G0'],
                operator='AND',
                output_label='test',
                positive_label='yes',
                negative_label='no',
            )


# ===== Tests for generate_thresholding_report =====

class TestGenerateThresholdingReport:
    """Tests for generate_thresholding_report function."""
    
    def test_text_format_basic(self, adata_with_thresholding_metadata):
        """Test basic text format report."""
        report = generate_thresholding_report(
            adata_with_thresholding_metadata,
            'gmm_thresholding_events',
            output_format='text'
        )
        
        # Check report is a string
        assert isinstance(report, str)
        
        # Check key elements are present
        assert 'Thresholding Report' in report
        assert 'DNA_thresholding' in report
        assert 'DNA_content' in report
        assert 'cell_cycle' in report
        assert 'Components: 3' in report
        assert '0.0230' in report  # Threshold
        assert '0.0450' in report  # Threshold
        assert 'G0' in report
        assert 'G1' in report
        assert 'S' in report
    
    def test_dataframe_format_basic(self, adata_with_thresholding_metadata):
        """Test basic dataframe format report."""
        report = generate_thresholding_report(
            adata_with_thresholding_metadata,
            'gmm_thresholding_events',
            output_format='dataframe'
        )
        
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
        assert report['Feature'].iloc[0] == 'DNA_content'
        assert report['Type'].iloc[0] == 'standard'
        assert report['Components'].iloc[0] == '3'
    
    def test_text_format_with_refinement(self, adata_with_thresholding_metadata):
        """Test text report with refinement operation."""
        # Add a refinement operation
        adata = adata_with_thresholding_metadata
        adata.uns['gmm_thresholding_events']['Plk1_refinement'] = {
            'feature_name': 'Plk1',
            'gmm_obs_label': 'cell_cycle',
            'layer': None,
            'gmm_info': {
                'n_components': 2,
            },
            'ordered_gmm_labels': ['G0_low', 'G0_high'],
            'decision_boundaries': {'thresholds': [120.5]},
            'condensed_labels': None,
            'operation_type': 'refinement',
            'parent_operation': 'DNA_thresholding',
            'refined_from_labels': ['G0'],
        }
        
        report = generate_thresholding_report(
            adata,
            'gmm_thresholding_events',
            output_format='text'
        )
        
        # Check refinement info is present
        assert 'Plk1_refinement' in report
        assert 'Refinement' in report
        assert 'DNA_thresholding' in report  # Parent
        assert 'G0' in report  # Refined from
        assert '120.5' in report  # Threshold
    
    def test_empty_events(self):
        """Test report with no operations."""
        adata = ad.AnnData(X=np.random.randn(10, 10))
        adata.uns['gmm_thresholding_events'] = OrderedDict()
        
        report = generate_thresholding_report(
            adata,
            'gmm_thresholding_events',
            output_format='text'
        )
        
        assert report == "No thresholding operations found."
    
    def test_empty_events_dataframe(self):
        """Test dataframe report with no operations."""
        adata = ad.AnnData(X=np.random.randn(10, 10))
        adata.uns['gmm_thresholding_events'] = OrderedDict()
        
        report = generate_thresholding_report(
            adata,
            'gmm_thresholding_events',
            output_format='dataframe'
        )
        
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 0
    
    def test_error_key_not_found(self):
        """Test error when thresholding_events_key doesn't exist."""
        adata = ad.AnnData(X=np.random.randn(10, 10))
        
        with pytest.raises(KeyError, match="thresholding_events_key 'nonexistent' not found"):
            generate_thresholding_report(
                adata,
                'nonexistent',
                output_format='text'
            )
    
    def test_error_invalid_format(self, adata_with_thresholding_metadata):
        """Test error with invalid output_format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            generate_thresholding_report(
                adata_with_thresholding_metadata,
                'gmm_thresholding_events',
                output_format='invalid'
            )
    
    def test_error_not_dict(self):
        """Test error when uns key is not a dict."""
        adata = ad.AnnData(X=np.random.randn(10, 10))
        adata.uns['gmm_thresholding_events'] = "not a dict"
        
        with pytest.raises(TypeError, match="must be a dict or OrderedDict"):
            generate_thresholding_report(
                adata,
                'gmm_thresholding_events',
                output_format='text'
            )
    
    def test_cell_counts_with_valid_obs(self, adata_with_thresholding_metadata):
        """Test that cell counts are calculated when obs column exists."""
        report = generate_thresholding_report(
            adata_with_thresholding_metadata,
            'gmm_thresholding_events',
            output_format='text'
        )
        
        # Check cell counts are present
        assert 'Cell counts:' in report
        assert 'G0=40' in report
        assert 'G1=30' in report
        assert 'S=30' in report
    
    def test_manual_thresholds_handling(self):
        """Test report handles manual thresholds (no GMM info)."""
        adata = ad.AnnData(X=np.random.randn(10, 10))
        adata.obs['labels'] = pd.Categorical(['Low'] * 5 + ['High'] * 5)
        
        adata.uns['gmm_thresholding_events'] = OrderedDict([
            ('manual_threshold', {
                'feature_name': 'Feature1',
                'gmm_obs_label': 'labels',
                'layer': None,
                'gmm_info': None,  # Manual thresholds
                'ordered_gmm_labels': ['Low', 'High'],
                'decision_boundaries': {'thresholds': [0.5]},
                'condensed_labels': None,
                'operation_type': 'refinement_manual',
                'parent_operation': None,
                'refined_from_labels': None,
            }),
        ])
        
        report = generate_thresholding_report(
            adata,
            'gmm_thresholding_events',
            output_format='text'
        )
        
        assert 'Components: N/A (manual thresholds)' in report
