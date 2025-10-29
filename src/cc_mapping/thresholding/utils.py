"""
Utility functions for GMM thresholding operations.

This module provides standalone utility functions for working with
thresholding results, including label combination and report generation.

Functions:
    create_boolean_label_combination: Combine two categorical labels with boolean operators
    generate_thresholding_report: Generate human-readable thresholding summary
"""

from collections import OrderedDict
from typing import List, Union

import anndata as ad
import numpy as np
import pandas as pd


def create_boolean_label_combination(
    adata: ad.AnnData,
    label1: str,
    label1_values: List[str],
    label2: str,
    label2_values: List[str],
    operator: str,
    output_label: str,
    positive_label: str,
    negative_label: str,
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Combine two categorical labels using boolean operators.
    
    Creates a new binary label based on whether cells match specified values
    in both input labels, using the specified boolean operator.
    
    Args:
        adata (ad.AnnData): AnnData object with observations to combine.
        label1 (str): First label column name in adata.obs.
        label1_values (List[str]): Values in label1 to consider "positive".
        label2 (str): Second label column name in adata.obs.
        label2_values (List[str]): Values in label2 to consider "positive".
        operator (str): Boolean operator - 'AND', 'OR', or 'XOR'.
        output_label (str): Name for new combined label column.
        positive_label (str): Label for cells matching criteria.
        negative_label (str): Label for cells not matching criteria.
        overwrite (bool, optional): If True, overwrites existing output_label. 
            If False, raises error if output_label exists. Defaults to False.
        
    Returns:
        ad.AnnData: Modified AnnData object with new obs column.
        
    Raises:
        KeyError: If label1 or label2 don't exist in adata.obs.
        ValueError: If operator is not 'AND', 'OR', or 'XOR'.
        KeyError: If output_label already exists in adata.obs and overwrite=False.
        TypeError: If label1_values or label2_values are not lists.
        ValueError: If any values in label1_values not found in label1.
        ValueError: If any values in label2_values not found in label2.
        
    Examples:
        # AND: Both conditions must be true
        >>> adata = create_boolean_label_combination(
        ...     adata,
        ...     label1='treatment', label1_values=['control'],
        ...     label2='cell_cycle', label2_values=['G0'],
        ...     operator='AND',
        ...     output_label='control_G0',
        ...     positive_label='control_G0',
        ...     negative_label='other'
        ... )
        # Result: 'control_G0' for cells that are BOTH control AND G0
        
        # OR: Either condition true
        >>> adata = create_boolean_label_combination(
        ...     adata,
        ...     label1='treatment', label1_values=['control', 'vehicle'],
        ...     label2='cell_cycle', label2_values=['G0', 'G1'],
        ...     operator='OR',
        ...     output_label='quiescent_or_control',
        ...     positive_label='positive',
        ...     negative_label='other'
        ... )
        # Result: 'positive' for cells in (control OR vehicle) OR (G0 OR G1)
        
        # XOR: Exactly one condition true (not both)
        >>> adata = create_boolean_label_combination(
        ...     adata,
        ...     label1='marker1', label1_values=['positive'],
        ...     label2='marker2', label2_values=['positive'],
        ...     operator='XOR',
        ...     output_label='single_positive',
        ...     positive_label='single_positive',
        ...     negative_label='other'
        ... )
        # Result: 'single_positive' for cells positive for exactly one marker
        
        # Overwrite existing column
        >>> adata = create_boolean_label_combination(
        ...     adata,
        ...     label1='treatment', label1_values=['control'],
        ...     label2='cell_cycle', label2_values=['G0'],
        ...     operator='AND',
        ...     output_label='control_G0',  # Already exists
        ...     positive_label='control_G0',
        ...     negative_label='other',
        ...     overwrite=True  # Allow overwriting
        ... )
    """
    # Validate label columns exist
    if label1 not in adata.obs.columns:
        raise KeyError(
            f"label1 '{label1}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    if label2 not in adata.obs.columns:
        raise KeyError(
            f"label2 '{label2}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    # Validate operator
    valid_operators = ['AND', 'OR', 'XOR']
    operator = operator.upper()
    if operator not in valid_operators:
        raise ValueError(
            f"operator must be one of {valid_operators}, got '{operator}'"
        )
    
    # Validate output_label doesn't already exist (unless overwrite=True)
    if output_label in adata.obs.columns and not overwrite:
        raise KeyError(
            f"output_label '{output_label}' already exists in adata.obs. "
            "Set overwrite=True to replace it, or choose a different name."
        )
    
    # Validate label values are lists
    if not isinstance(label1_values, list):
        raise TypeError(
            f"label1_values must be a list, got {type(label1_values)}"
        )
    
    if not isinstance(label2_values, list):
        raise TypeError(
            f"label2_values must be a list, got {type(label2_values)}"
        )
    
    # Validate all values exist in their respective labels
    unique_label1 = set(adata.obs[label1].unique())
    for val in label1_values:
        if val not in unique_label1:
            raise ValueError(
                f"Value '{val}' not found in label1 '{label1}'. "
                f"Available values: {sorted(unique_label1)}"
            )
    
    unique_label2 = set(adata.obs[label2].unique())
    for val in label2_values:
        if val not in unique_label2:
            raise ValueError(
                f"Value '{val}' not found in label2 '{label2}'. "
                f"Available values: {sorted(unique_label2)}"
            )
    
    # Create boolean masks
    mask1 = adata.obs[label1].isin(label1_values)
    mask2 = adata.obs[label2].isin(label2_values)
    
    # Apply boolean operator
    if operator == 'AND':
        final_mask = mask1 & mask2
    elif operator == 'OR':
        final_mask = mask1 | mask2
    elif operator == 'XOR':
        final_mask = mask1 ^ mask2
    
    # Create new categorical column
    new_labels = np.where(final_mask, positive_label, negative_label)
    adata.obs[output_label] = pd.Categorical(new_labels)
    
    return adata


def generate_thresholding_report(
    adata: ad.AnnData,
    thresholding_events_key: str,
    output_format: str = 'text',
) -> Union[str, pd.DataFrame]:
    """
    Generate a human-readable report of all thresholding operations.
    
    Reads thresholding metadata from adata.uns and creates a summary showing:
    - Operation names and order
    - Features used
    - Number of components
    - Thresholds calculated
    - Labels assigned
    - Parent operations (for refinements)
    - Cell counts per category (captured at operation time)
    
    Note: Cell counts reflect the state immediately after each operation was performed,
    not the current state of the data. This is important because subsequent refinement
    operations may change labels, but the historical counts are preserved.
    
    Args:
        adata (ad.AnnData): AnnData object with thresholding metadata.
        thresholding_events_key (str): Key in adata.uns containing operations.
        output_format (str, optional): 'text' for formatted string, 'dataframe' 
            for pandas DataFrame. Defaults to 'text'.
        
    Returns:
        Union[str, pd.DataFrame]: Formatted report string or DataFrame.
        
    Raises:
        KeyError: If thresholding_events_key doesn't exist in adata.uns.
        ValueError: If output_format is not 'text' or 'dataframe'.
        TypeError: If adata.uns[thresholding_events_key] is not a dict.
        
    Examples:
        # Generate text report
        >>> report = generate_thresholding_report(
        ...     adata, 
        ...     'sequential_gmm_thresholding_events'
        ... )
        >>> print(report)
        
        # Output:
        # Thresholding Report
        # ===================
        # 
        # 1. DNA_content (Standard Thresholding)
        #    Feature: DNA_content
        #    Layer: None
        #    Components: 3
        #    Thresholds: [0.023, 0.045]
        #    Labels: ['Low', 'Medium', 'High']
        #    Cell counts: Low=1234, Medium=5678, High=910
        # 
        # 2. Plk1_G0_refinement (Refinement of DNA_content)
        #    Feature: Plk1
        #    Layer: None
        #    Parent operation: DNA_content
        #    Refined labels: ['G0']
        #    Components: 2
        #    Thresholds: [120.5]
        #    New labels: ['G0_low', 'G0_high']
        #    Cell counts: G0_low=567, G0_high=667
        
        # Generate DataFrame report
        >>> report_df = generate_thresholding_report(
        ...     adata,
        ...     'sequential_gmm_thresholding_events',
        ...     output_format='dataframe'
        ... )
        >>> report_df.head()
    """
    # Validate thresholding_events_key exists
    if thresholding_events_key not in adata.uns:
        raise KeyError(
            f"thresholding_events_key '{thresholding_events_key}' not found in adata.uns. "
            f"Available keys: {list(adata.uns.keys())}"
        )
    
    # Validate it's a dict-like structure
    events = adata.uns[thresholding_events_key]
    if not isinstance(events, (dict, OrderedDict)):
        raise TypeError(
            f"adata.uns['{thresholding_events_key}'] must be a dict or OrderedDict, "
            f"got {type(events)}"
        )
    
    # Validate output_format
    valid_formats = ['text', 'dataframe']
    if output_format not in valid_formats:
        raise ValueError(
            f"output_format must be one of {valid_formats}, got '{output_format}'"
        )
    
    if len(events) == 0:
        if output_format == 'text':
            return "No thresholding operations found."
        else:
            return pd.DataFrame()
    
    # Build report data
    report_data = []
    
    for idx, (op_name, op_data) in enumerate(events.items(), 1):
        # Extract basic info
        feature = op_data.get('feature_name', 'N/A')
        layer = op_data.get('layer', None)
        obs_label = op_data.get('gmm_obs_label', 'N/A')
        ordered_labels = op_data.get('ordered_gmm_labels', [])
        
        # Extract GMM info
        gmm_info = op_data.get('gmm_info', {})
        if gmm_info is not None:
            n_components = gmm_info.get('n_components', 'N/A')
        else:
            n_components = 'N/A (manual thresholds)'
        
        # Extract thresholds
        decision_boundaries = op_data.get('decision_boundaries', {})
        thresholds = decision_boundaries.get('thresholds', []) if decision_boundaries else []
        
        # Extract operation type and hierarchy
        operation_type = op_data.get('operation_type', 'standard')
        parent_operation = op_data.get('parent_operation', None)
        refined_from_labels = op_data.get('refined_from_labels', None)
        
        # Get cell counts - prefer stored counts from operation time
        cell_counts = op_data.get('cell_counts_after_operation', {})
        
        # Fallback to current obs counts if not stored (backward compatibility)
        if not cell_counts and obs_label in adata.obs.columns:
            counts = adata.obs[obs_label].value_counts()
            # Only include labels from this operation
            for label in ordered_labels:
                if label in counts.index:
                    cell_counts[label] = int(counts[label])
        
        # Store data for this operation
        op_info = {
            'operation_number': idx,
            'operation_name': op_name,
            'operation_type': operation_type,
            'feature': feature,
            'layer': str(layer),
            'obs_label': obs_label,
            'n_components': n_components,
            'thresholds': thresholds,
            'labels': ordered_labels,
            'parent_operation': parent_operation,
            'refined_from_labels': refined_from_labels,
            'cell_counts': cell_counts,
        }
        report_data.append(op_info)
    
    # Generate output based on format
    if output_format == 'dataframe':
        # Create DataFrame
        df_data = []
        for op in report_data:
            df_data.append({
                'Operation': f"{op['operation_number']}. {op['operation_name']}",
                'Type': op['operation_type'],
                'Feature': op['feature'],
                'Layer': op['layer'],
                'Obs Label': op['obs_label'],
                'Components': str(op['n_components']),
                'Thresholds': ', '.join(f"{t:.4f}" for t in op['thresholds']) if op['thresholds'] else 'N/A',
                'Labels': ', '.join(op['labels']),
                'Parent': str(op['parent_operation']),
                'Refined From': ', '.join(op['refined_from_labels']) if op['refined_from_labels'] else 'N/A',
                'Total Cells': sum(op['cell_counts'].values()) if op['cell_counts'] else 'N/A',
            })
        return pd.DataFrame(df_data)
    
    else:  # text format
        lines = []
        lines.append("Thresholding Report")
        lines.append("=" * 50)
        lines.append("")
        
        for op in report_data:
            # Header
            if op['operation_type'] == 'refinement' or op['operation_type'] == 'refinement_manual':
                header = f"{op['operation_number']}. {op['operation_name']} (Refinement)"
                if op['parent_operation']:
                    header += f" of {op['parent_operation']}"
            else:
                header = f"{op['operation_number']}. {op['operation_name']} (Standard Thresholding)"
            
            lines.append(header)
            lines.append("-" * len(header))
            
            # Basic info
            lines.append(f"   Feature: {op['feature']}")
            lines.append(f"   Layer: {op['layer']}")
            lines.append(f"   Obs column: {op['obs_label']}")
            
            # GMM info
            lines.append(f"   Components: {op['n_components']}")
            
            # Thresholds
            if op['thresholds']:
                threshold_str = ', '.join(f"{t:.4f}" for t in op['thresholds'])
                lines.append(f"   Thresholds: [{threshold_str}]")
            else:
                lines.append(f"   Thresholds: None")
            
            # Labels
            labels_str = ', '.join(f"'{label}'" for label in op['labels'])
            lines.append(f"   Labels: [{labels_str}]")
            
            # Refinement-specific info
            if op['refined_from_labels']:
                refined_str = ', '.join(f"'{label}'" for label in op['refined_from_labels'])
                lines.append(f"   Refined from: [{refined_str}]")
            
            # Cell counts
            if op['cell_counts']:
                count_strs = [f"{label}={count}" for label, count in op['cell_counts'].items()]
                lines.append(f"   Cell counts: {', '.join(count_strs)}")
            else:
                lines.append(f"   Cell counts: Not available (obs column may have been modified)")
            
            lines.append("")
        
        # Summary
        lines.append("=" * 50)
        lines.append(f"Total operations: {len(report_data)}")
        
        # Count operation types
        type_counts = {}
        for op in report_data:
            op_type = op['operation_type']
            type_counts[op_type] = type_counts.get(op_type, 0) + 1
        
        if type_counts:
            lines.append("Operation types:")
            for op_type, count in type_counts.items():
                lines.append(f"  - {op_type}: {count}")
        
        return '\n'.join(lines)
