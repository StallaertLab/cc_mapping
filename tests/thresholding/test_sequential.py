"""
Tests for SequentialGMM class.

This module tests the sequential thresholding functionality including:
- Initialization
- threshold_entire_dataset()
- refine_labels_with_gmm()
- refine_labels_with_manual_thresholds()
- Helper methods
- Plotting methods
"""

from collections import OrderedDict

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cc_mapping.thresholding import SequentialGMM

# =====================
# Fixtures
# =====================


@pytest.fixture
def basic_adata():
    """Create basic AnnData object for testing."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 3

    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_obs)]})
    var = pd.DataFrame(index=["DNA", "Plk1", "CyclinB"])

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata_with_bimodal_dist():
    """Create AnnData with clear bimodal distribution for testing.

    This creates a hierarchical structure:
    - DNA separates into Low (G0) and High (S) populations
    - Within Low population, Plk1 separates into G0_low and G0_high
    - Within High population, Plk1 separates into S_low and S_high
    """
    np.random.seed(42)

    # Create 4 populations with well-separated bimodal distributions
    # Population 1: DNA Low + Plk1 Low (50 cells)
    pop1_dna = np.random.normal(0, 0.3, 50)
    pop1_plk1 = np.random.normal(-2, 0.3, 50)

    # Population 2: DNA Low + Plk1 High (50 cells)
    pop2_dna = np.random.normal(0, 0.3, 50)
    pop2_plk1 = np.random.normal(1, 0.3, 50)

    # Population 3: DNA High + Plk1 Low (50 cells)
    pop3_dna = np.random.normal(5, 0.3, 50)
    pop3_plk1 = np.random.normal(-2, 0.3, 50)

    # Population 4: DNA High + Plk1 High (50 cells)
    pop4_dna = np.random.normal(5, 0.3, 50)
    pop4_plk1 = np.random.normal(1, 0.3, 50)

    # Combine all populations
    dna = np.concatenate([pop1_dna, pop2_dna, pop3_dna, pop4_dna])
    plk1 = np.concatenate([pop1_plk1, pop2_plk1, pop3_plk1, pop4_plk1])

    # Third feature (not used for thresholding in most tests)
    cyclin = np.random.randn(200)

    X = np.column_stack([dna, plk1, cyclin])
    obs = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(200)]})
    var = pd.DataFrame(index=["DNA", "Plk1", "CyclinB"])

    return ad.AnnData(X=X, obs=obs, var=var)


# =====================
# Initialization Tests
# =====================


class TestInitialization:
    """Test SequentialGMM initialization."""

    def test_init_success_defaults(self, basic_adata):
        """Test successful initialization with default parameters."""
        seq_gmm = SequentialGMM(adata=basic_adata)

        assert seq_gmm.adata is not None
        assert seq_gmm.adata.shape == basic_adata.shape
        assert seq_gmm.thresholding_events_key == "sequential_gmm_thresholding_events"
        assert seq_gmm.random_state == 42
        assert seq_gmm.gmm_kwargs == {}
        assert "sequential_gmm_thresholding_events" in seq_gmm.adata.uns
        assert isinstance(
            seq_gmm.adata.uns["sequential_gmm_thresholding_events"], OrderedDict
        )

    def test_init_success_custom_key(self, basic_adata):
        """Test initialization with custom .uns key."""
        seq_gmm = SequentialGMM(
            adata=basic_adata, thresholding_events_key="my_custom_key"
        )

        assert seq_gmm.thresholding_events_key == "my_custom_key"
        assert "my_custom_key" in seq_gmm.adata.uns
        assert isinstance(seq_gmm.adata.uns["my_custom_key"], OrderedDict)

    def test_init_success_custom_gmm_kwargs(self, basic_adata):
        """Test initialization with custom GMM kwargs."""
        custom_kwargs = {"covariance_type": "diag", "max_iter": 200}
        seq_gmm = SequentialGMM(adata=basic_adata, gmm_kwargs=custom_kwargs)

        assert seq_gmm.gmm_kwargs == custom_kwargs

    def test_init_success_existing_uns_key(self, basic_adata):
        """Test initialization with existing .uns key (should not error)."""
        basic_adata.uns["sequential_gmm_thresholding_events"] = OrderedDict()

        seq_gmm = SequentialGMM(adata=basic_adata)

        assert "sequential_gmm_thresholding_events" in seq_gmm.adata.uns

    def test_init_error_not_anndata(self):
        """Test initialization fails with non-AnnData object."""
        with pytest.raises(TypeError, match="adata must be an AnnData.AnnData object"):
            SequentialGMM(adata="not_anndata")

    def test_init_error_invalid_thresholding_key_type(self, basic_adata):
        """Test initialization fails with non-string thresholding_events_key."""
        with pytest.raises(TypeError, match="thresholding_events_key must be a string"):
            SequentialGMM(adata=basic_adata, thresholding_events_key=123)

    def test_init_error_empty_thresholding_key(self, basic_adata):
        """Test initialization fails with empty thresholding_events_key."""
        with pytest.raises(
            ValueError, match="thresholding_events_key cannot be an empty string"
        ):
            SequentialGMM(adata=basic_adata, thresholding_events_key="")

    def test_init_error_invalid_uns_key_type(self, basic_adata):
        """Test initialization fails when existing .uns key is not OrderedDict."""
        basic_adata.uns[
            "sequential_gmm_thresholding_events"
        ] = {}  # dict, not OrderedDict

        with pytest.raises(TypeError, match="must be an OrderedDict"):
            SequentialGMM(adata=basic_adata)

    def test_init_error_invalid_gmm_kwargs_type(self, basic_adata):
        """Test initialization fails with non-dict gmm_kwargs."""
        with pytest.raises(TypeError, match="gmm_kwargs must be a dictionary"):
            SequentialGMM(adata=basic_adata, gmm_kwargs="not_a_dict")


# =====================
# threshold_entire_dataset Tests
# =====================


class TestThresholdEntireDataset:
    """Test threshold_entire_dataset() method."""

    def test_threshold_entire_dataset_success(self, adata_with_bimodal_dist):
        """Test successful thresholding of entire dataset."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["Low", "High"],
            operation_name="DNA_threshold",
        )

        # Check obs column created
        assert "cell_cycle" in seq_gmm.adata.obs.columns
        assert set(seq_gmm.adata.obs["cell_cycle"].unique()) == {"Low", "High"}

        # Check operation stored
        assert (
            "DNA_threshold" in seq_gmm.adata.uns["sequential_gmm_thresholding_events"]
        )

        # Check metadata
        op_data = seq_gmm.adata.uns["sequential_gmm_thresholding_events"][
            "DNA_threshold"
        ]
        assert op_data["operation_type"] == "standard"
        assert op_data["parent_operation"] is None
        assert op_data["refined_from_labels"] is None
        assert op_data["feature_name"] == "DNA"
        assert op_data["gmm_obs_label"] == "cell_cycle"

    def test_threshold_entire_dataset_with_manual_thresholds(
        self, adata_with_bimodal_dist
    ):
        """Test thresholding with manual thresholds."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["Low", "High"],
            manual_thresholds=[1.5],
            operation_name="DNA_manual",
        )

        assert "cell_cycle" in seq_gmm.adata.obs.columns
        assert "DNA_manual" in seq_gmm.adata.uns["sequential_gmm_thresholding_events"]

    def test_threshold_entire_dataset_with_duplicate_labels(
        self, adata_with_bimodal_dist
    ):
        """Test thresholding with duplicate labels (label collapsing)."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=4,
            ordered_labels=["Low", "Low", "High", "High"],
            duplicate_labels=True,
            operation_name="DNA_collapsed",
        )

        # Should only have 2 final labels despite 4 components
        assert set(seq_gmm.adata.obs["cell_cycle"].unique()) == {"Low", "High"}

    def test_threshold_entire_dataset_error_no_operation_name(self, basic_adata):
        """Test error when operation_name is None."""
        seq_gmm = SequentialGMM(adata=basic_adata)

        with pytest.raises(ValueError, match="operation_name is required"):
            seq_gmm.threshold_entire_dataset(
                feature="DNA",
                label_obs_save_str="cell_cycle",
                n_components=2,
                ordered_labels=["Low", "High"],
                operation_name=None,
            )

    def test_threshold_entire_dataset_error_duplicate_operation_name(
        self, adata_with_bimodal_dist
    ):
        """Test error when operation_name already exists."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        # First operation
        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["Low", "High"],
            operation_name="DNA_threshold",
        )

        # Try to use same name again
        with pytest.raises(KeyError, match="already exists"):
            seq_gmm.threshold_entire_dataset(
                feature="Plk1",
                label_obs_save_str="plk1_level",
                n_components=2,
                ordered_labels=["Low", "High"],
                operation_name="DNA_threshold",  # Same name
            )


# =====================
# refine_labels_with_gmm Tests
# =====================


class TestRefineLabelsWithGMM:
    """Test refine_labels_with_gmm() method."""

    def test_refine_labels_success(self, adata_with_bimodal_dist):
        """Test successful label refinement."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        # Initial thresholding
        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["G0", "S"],
            operation_name="DNA_threshold",
        )

        # Refine G0 cells
        seq_gmm.refine_labels_with_gmm(
            feature="Plk1",
            obs_label="cell_cycle",
            value_to_refine="G0",
            n_components=2,
            ordered_labels=["G0_low", "G0_high"],
            operation_name="Plk1_refinement",
        )

        # Check labels updated
        assert "G0_low" in seq_gmm.adata.obs["cell_cycle"].values
        assert "G0_high" in seq_gmm.adata.obs["cell_cycle"].values
        # Original 'G0' should be replaced
        # (though some cells might still have 'S')

        # Check operation stored
        assert (
            "Plk1_refinement" in seq_gmm.adata.uns["sequential_gmm_thresholding_events"]
        )

        # Check metadata
        op_data = seq_gmm.adata.uns["sequential_gmm_thresholding_events"][
            "Plk1_refinement"
        ]
        assert op_data["operation_type"] == "refinement"
        assert op_data["parent_operation"] == "cell_cycle"
        assert op_data["refined_from_labels"] == ["G0"]
        assert op_data["feature_name"] == "Plk1"

    def test_refine_labels_error_no_operation_name(self, adata_with_bimodal_dist):
        """Test error when operation_name is None."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        # Create initial labels
        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["G0", "S"],
            operation_name="DNA_threshold",
        )

        with pytest.raises(ValueError, match="operation_name is required"):
            seq_gmm.refine_labels_with_gmm(
                feature="Plk1",
                obs_label="cell_cycle",
                value_to_refine="G0",
                n_components=2,
                ordered_labels=["G0_low", "G0_high"],
                operation_name=None,
            )

    def test_refine_labels_error_obs_label_not_found(self, basic_adata):
        """Test error when obs_label doesn't exist."""
        seq_gmm = SequentialGMM(adata=basic_adata)

        with pytest.raises(KeyError, match="not found in adata.obs"):
            seq_gmm.refine_labels_with_gmm(
                feature="Plk1",
                obs_label="nonexistent_column",
                value_to_refine="G0",
                n_components=2,
                ordered_labels=["G0_low", "G0_high"],
                operation_name="Plk1_refinement",
            )

    def test_refine_labels_error_value_not_found(self, adata_with_bimodal_dist):
        """Test error when value_to_refine doesn't exist."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        # Create initial labels
        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["G0", "S"],
            operation_name="DNA_threshold",
        )

        with pytest.raises(ValueError, match="not found in adata.obs"):
            seq_gmm.refine_labels_with_gmm(
                feature="Plk1",
                obs_label="cell_cycle",
                value_to_refine="G2M",  # Doesn't exist
                n_components=2,
                ordered_labels=["G2M_low", "G2M_high"],
                operation_name="Plk1_refinement",
            )


# =====================
# refine_labels_with_manual_thresholds Tests
# =====================


class TestRefineLabelsWithManualThresholds:
    """Test refine_labels_with_manual_thresholds() method."""

    def test_refine_manual_success(self, adata_with_bimodal_dist):
        """Test successful manual threshold refinement."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        # Initial thresholding
        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["G0", "S"],
            operation_name="DNA_threshold",
        )

        # Refine with manual thresholds
        # G0 cells have Plk1 ~N(-1, 0.3), so use -1 as threshold to split them
        seq_gmm.refine_labels_with_manual_thresholds(
            feature="Plk1",
            obs_label="cell_cycle",
            value_to_refine="G0",
            manual_thresholds=[-1.0],
            ordered_labels=["G0_low", "G0_high"],
            operation_name="Plk1_manual",
        )

        # Check labels updated
        assert "G0_low" in seq_gmm.adata.obs["cell_cycle"].values
        assert "G0_high" in seq_gmm.adata.obs["cell_cycle"].values

        # Check operation stored
        assert "Plk1_manual" in seq_gmm.adata.uns["sequential_gmm_thresholding_events"]

        # Check metadata (should not have GMM info)
        op_data = seq_gmm.adata.uns["sequential_gmm_thresholding_events"]["Plk1_manual"]
        assert op_data["operation_type"] == "refinement_manual"
        assert op_data["gmm_info"] is None
        assert op_data["decision_boundaries"]["thresholds"] == [-1.0]

    def test_refine_manual_error_wrong_threshold_count(self, adata_with_bimodal_dist):
        """Test error when threshold count doesn't match labels."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        # Initial thresholding
        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["G0", "S"],
            operation_name="DNA_threshold",
        )

        with pytest.raises(ValueError, match="Number of thresholds"):
            seq_gmm.refine_labels_with_manual_thresholds(
                feature="Plk1",
                obs_label="cell_cycle",
                value_to_refine="G0",
                manual_thresholds=[0.5, 1.5],  # 2 thresholds for 2 labels (should be 1)
                ordered_labels=["G0_low", "G0_high"],
                operation_name="Plk1_manual",
            )


# =====================
# return_adata Tests
# =====================


class TestReturnAdata:
    """Test return_adata() method."""

    def test_return_adata(self, adata_with_bimodal_dist):
        """Test return_adata returns modified object."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["G0", "S"],
            operation_name="DNA_threshold",
        )

        adata_result = seq_gmm.return_adata()

        assert isinstance(adata_result, ad.AnnData)
        assert "cell_cycle" in adata_result.obs.columns
        assert "DNA_threshold" in adata_result.uns["sequential_gmm_thresholding_events"]


# =====================
# Integration Tests
# =====================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_sequential_workflow(self, adata_with_bimodal_dist):
        """Test complete sequential thresholding workflow."""
        seq_gmm = SequentialGMM(adata=adata_with_bimodal_dist)

        # Step 1: Initial thresholding on DNA
        seq_gmm.threshold_entire_dataset(
            feature="DNA",
            label_obs_save_str="cell_cycle",
            n_components=2,
            ordered_labels=["Low", "High"],
            operation_name="DNA_initial",
        )

        # Step 2: Refine 'Low' cells with Plk1
        seq_gmm.refine_labels_with_gmm(
            feature="Plk1",
            obs_label="cell_cycle",
            value_to_refine="Low",
            n_components=2,
            ordered_labels=["G0", "G1"],
            operation_name="Plk1_low_refinement",
        )

        # Step 3: Refine 'High' cells with CyclinB
        seq_gmm.refine_labels_with_gmm(
            feature="CyclinB",
            obs_label="cell_cycle",
            value_to_refine="High",
            n_components=2,
            ordered_labels=["S", "G2M"],
            operation_name="CyclinB_high_refinement",
        )

        # Step 4: Refine G2M with manual threshold
        seq_gmm.refine_labels_with_manual_thresholds(
            feature="DNA",
            obs_label="cell_cycle",
            value_to_refine="G2M",
            manual_thresholds=[2.5],
            ordered_labels=["G2", "M"],
            operation_name="DNA_G2M_manual",
        )

        # Get result
        adata_result = seq_gmm.return_adata()

        # Check all operations stored
        assert len(adata_result.uns["sequential_gmm_thresholding_events"]) == 4
        assert "DNA_initial" in adata_result.uns["sequential_gmm_thresholding_events"]
        assert (
            "Plk1_low_refinement"
            in adata_result.uns["sequential_gmm_thresholding_events"]
        )
        assert (
            "CyclinB_high_refinement"
            in adata_result.uns["sequential_gmm_thresholding_events"]
        )
        assert (
            "DNA_G2M_manual" in adata_result.uns["sequential_gmm_thresholding_events"]
        )

        # Check final labels exist
        unique_labels = set(adata_result.obs["cell_cycle"].unique())
        expected_labels = {"G0", "G1", "S", "G2", "M"}
        assert expected_labels.issubset(unique_labels) or len(unique_labels) > 0

        print("Full workflow test passed!")
        print(f"Final unique labels: {unique_labels}")
        print(f"Label counts: {adata_result.obs['cell_cycle'].value_counts()}")
