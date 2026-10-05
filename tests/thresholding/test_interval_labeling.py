"""Tests for how GMM probabilities are turned into labels along a feature.

Samples are sorted by the feature and each one's most likely GMM component is
taken. Walking along the feature, a step up to a component above the current
interval's component starts a new interval (threshold at the midpoint of the
step), and every sample gets the label of the interval it falls in. Each label
therefore covers one contiguous range of the feature, a label whose component
never wins stays empty, and samples whose most likely component differs from
their interval's label are counted as reassigned.

The probability matrices are hand-made: 300 samples at feature values 0..299,
labels ['A', 'B', 'C'], and a list of each sample's most likely component in
feature order. Samples are stored shuffled, so the code has to sort them.
"""

import re
import warnings

import anndata as ad
import numpy as np
import pytest

from cc_mapping.thresholding import GMMThresholding, SequentialGMM
from cc_mapping.thresholding.base import GaussianMixtureModelBase
from tests.helpers import probs_from_winners, use_fake_gaussian_mixture

FEATURE = "marker"
LABELS = ["A", "B", "C"]
FEATURE_VALUES = np.arange(300, dtype=float)
SHUFFLE = np.random.default_rng(0).permutation(FEATURE_VALUES.size)
SEQUENTIAL_KEY = "sequential_gmm_thresholding_events"

THIRDS = ["A"] * 100 + ["B"] * 100 + ["C"] * 100

# Each case: the most likely component of every sample in feature order, and the
# result expected from the rule, derived by hand (labels are in feature order).
CASES = {
    "clean": dict(
        winners=[0] * 100 + [1] * 100 + [2] * 100,
        labels=THIRDS,
        thresholds=[99.5, 199.5],
        interval_components=[0, 1, 2],
        empty_labels=[],
        n_reassigned=0,
        warnings=set(),
    ),
    "B_never_wins": dict(
        winners=[0] * 150 + [2] * 150,
        labels=["A"] * 150 + ["C"] * 150,
        thresholds=[149.5],
        interval_components=[0, 2],
        empty_labels=["B"],
        n_reassigned=0,
        warnings={"empty"},
    ),
    "A_wins_again_in_high_tail": dict(
        winners=[0] * 100 + [1] * 100 + [2] * 95 + [0] * 5,
        labels=THIRDS,
        thresholds=[99.5, 199.5],
        interval_components=[0, 1, 2],
        empty_labels=[],
        n_reassigned=5,
        warnings={"reassigned"},
    ),
    "C_wins_in_low_tail": dict(
        winners=[2] * 5 + [0] * 95 + [1] * 100 + [2] * 100,
        labels=THIRDS,
        thresholds=[99.5, 199.5],
        interval_components=[0, 1, 2],
        empty_labels=[],
        n_reassigned=5,
        warnings={"reassigned"},
    ),
    "A_and_B_alternate_before_C": dict(
        winners=[0] * 100 + [1] * 10 + [0] * 10 + [1] * 80 + [2] * 100,
        labels=THIRDS,
        thresholds=[99.5, 199.5],
        interval_components=[0, 1, 2],
        empty_labels=[],
        n_reassigned=10,
        warnings={"reassigned"},
    ),
    # No step up at all: the most common component (A, 200 samples) takes everything
    "no_step_up": dict(
        winners=[2] * 100 + [0] * 200,
        labels=["A"] * 300,
        thresholds=[],
        interval_components=[0],
        empty_labels=["B", "C"],
        n_reassigned=100,
        warnings={"empty", "reassigned"},
    ),
}

parametrize_cases = pytest.mark.parametrize(
    "case", list(CASES.values()), ids=list(CASES.keys())
)

# Label A spread over two GMM components, collapsed with duplicate_labels=True
DUPLICATE_LABELS = ["A", "A", "B", "C"]


def _spread_a_over_two_components(winners):
    """Maps label winners to 4 components: A samples alternate between components 0 and 1."""
    winners = np.asarray(winners)
    components = winners + 1
    is_a = winners == 0
    components[is_a] = np.arange(is_a.sum()) % 2
    return components


@pytest.fixture(params=["unique_labels", "duplicate_labels"])
def label_setup(request):
    """Ordered labels for the classes, and how label winners map to GMM components."""
    if request.param == "unique_labels":
        return dict(
            ordered_labels=LABELS,
            duplicate_labels=False,
            to_components=np.asarray,
        )
    return dict(
        ordered_labels=DUPLICATE_LABELS,
        duplicate_labels=True,
        to_components=_spread_a_over_two_components,
    )


def _warning_kinds(caught):
    """Which of the two labeling warnings were issued."""
    kinds = set()
    for w in caught:
        message = str(w.message).lower()
        if "no samples" in message:
            kinds.add("empty")
        if "reassigned" in message:
            kinds.add("reassigned")
    return kinds


def _boundaries_from_probs(winners):
    """Runs the base rule on shuffled samples; returns (base, boundaries, warnings)."""
    base = GaussianMixtureModelBase()
    probs = probs_from_winners(winners, n_components=len(LABELS))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        boundaries = base._calculate_decision_boundaries_from_probs(
            feature_values=FEATURE_VALUES[SHUFFLE],
            data_probs=probs[SHUFFLE],
            ordered_labels=LABELS,
            feature_name=FEATURE,
        )
    return base, boundaries, caught


def _adata():
    """AnnData with the feature stored in shuffled order."""
    adata = ad.AnnData(X=FEATURE_VALUES[SHUFFLE].reshape(-1, 1))
    adata.var_names = [FEATURE]
    adata.obs_names = [f"sample_{i}" for i in range(adata.n_obs)]
    return adata


def _adata_with_group():
    """The 300 samples to refine plus 20 samples of another group that must be left alone."""
    values = np.concatenate([FEATURE_VALUES[SHUFFLE], 1000.0 + np.arange(20)])
    adata = ad.AnnData(X=values.reshape(-1, 1))
    adata.var_names = [FEATURE]
    adata.obs_names = [f"sample_{i}" for i in range(adata.n_obs)]
    adata.obs["group"] = ["to_refine"] * 300 + ["other"] * 20
    return adata


def _expected_in_storage_order(labels):
    return np.array(labels)[SHUFFLE].tolist()


### The rule in the base class ###


@parametrize_cases
def test_rule_gives_each_label_one_contiguous_range(case):
    base, boundaries, _ = _boundaries_from_probs(case["winners"])

    label_indices = base._assign_label_indices(
        FEATURE_VALUES, boundaries.thresholds, boundaries.interval_components
    )

    assert np.array(LABELS)[label_indices].tolist() == case["labels"]
    assert boundaries.thresholds == case["thresholds"]
    assert boundaries.interval_components == case["interval_components"]


@parametrize_cases
def test_rule_reports_empty_labels_and_reassigned_samples(case):
    _, boundaries, caught = _boundaries_from_probs(case["winners"])

    assert boundaries.empty_labels == case["empty_labels"]
    assert boundaries.n_reassigned_samples == case["n_reassigned"]
    assert _warning_kinds(caught) == case["warnings"]


def test_empty_label_warning_names_the_labels_and_feature():
    _, _, caught = _boundaries_from_probs(CASES["B_never_wins"]["winners"])

    (message,) = [
        str(w.message) for w in caught if "no samples" in str(w.message).lower()
    ]
    assert "'B'" in message
    assert "'A'" not in message and "'C'" not in message
    assert FEATURE in message


def test_reassigned_warning_gives_count_share_and_feature_range():
    _, _, caught = _boundaries_from_probs(CASES["A_wins_again_in_high_tail"]["winners"])

    (message,) = [
        str(w.message) for w in caught if "reassigned" in str(w.message).lower()
    ]
    assert re.search(r"\b5\b", message)  # 5 samples ...
    assert re.search(r"1\.67?%", message)  # ... of 300
    assert "295" in message and "299" in message  # feature range they fall in
    assert FEATURE in message


### Through GMMThresholding and SequentialGMM ###


@parametrize_cases
def test_gmm_thresholding_labels_follow_the_rule(case, label_setup, monkeypatch):
    n_components = len(label_setup["ordered_labels"])
    component_winners = label_setup["to_components"](case["winners"])
    use_fake_gaussian_mixture(
        monkeypatch, FEATURE_VALUES, probs_from_winners(component_winners, n_components)
    )
    gmm = GMMThresholding(adata=_adata(), feature=FEATURE, label_obs_save_str="labels")
    gmm.fit(n_components=n_components)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gmm.categorize_samples(
            ordered_labels=label_setup["ordered_labels"],
            duplicate_labels=label_setup["duplicate_labels"],
        )

    assert gmm.adata.obs["labels"].tolist() == _expected_in_storage_order(
        case["labels"]
    )
    boundaries = gmm._internal_data.decision_boundaries
    assert boundaries.thresholds == case["thresholds"]
    assert boundaries.interval_components == case["interval_components"]
    assert boundaries.empty_labels == case["empty_labels"]
    assert boundaries.n_reassigned_samples == case["n_reassigned"]
    assert _warning_kinds(caught) == case["warnings"]


@parametrize_cases
def test_sequential_refinement_labels_follow_the_rule(case, label_setup, monkeypatch):
    n_components = len(label_setup["ordered_labels"])
    component_winners = label_setup["to_components"](case["winners"])
    use_fake_gaussian_mixture(
        monkeypatch, FEATURE_VALUES, probs_from_winners(component_winners, n_components)
    )
    seq_gmm = SequentialGMM(adata=_adata_with_group())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seq_gmm.refine_labels_with_gmm(
            feature=FEATURE,
            obs_label="group",
            value_to_refine="to_refine",
            n_components=n_components,
            ordered_labels=label_setup["ordered_labels"],
            duplicate_labels=label_setup["duplicate_labels"],
            operation_name="refine",
        )

    labels = seq_gmm.adata.obs["group"].astype(str).tolist()
    assert labels[:300] == _expected_in_storage_order(case["labels"])
    assert labels[300:] == ["other"] * 20
    boundaries = seq_gmm.adata.uns[SEQUENTIAL_KEY]["refine"]["decision_boundaries"]
    assert boundaries["thresholds"] == case["thresholds"]
    assert boundaries["interval_components"] == case["interval_components"]
    assert boundaries["empty_labels"] == case["empty_labels"]
    assert boundaries["n_reassigned_samples"] == case["n_reassigned"]
    assert _warning_kinds(caught) == case["warnings"]


def test_threshold_entire_dataset_follows_the_rule(monkeypatch):
    case = CASES["B_never_wins"]
    use_fake_gaussian_mixture(
        monkeypatch, FEATURE_VALUES, probs_from_winners(case["winners"], 3)
    )
    seq_gmm = SequentialGMM(adata=_adata())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the empty-label warning is expected here
        seq_gmm.threshold_entire_dataset(
            feature=FEATURE,
            label_obs_save_str="labels",
            n_components=3,
            ordered_labels=LABELS,
            operation_name="initial",
        )

    assert seq_gmm.adata.obs["labels"].tolist() == _expected_in_storage_order(
        case["labels"]
    )
    boundaries = seq_gmm.adata.uns[SEQUENTIAL_KEY]["initial"]["decision_boundaries"]
    assert boundaries["interval_components"] == [0, 2]
    assert boundaries["empty_labels"] == ["B"]
    assert boundaries["n_reassigned_samples"] == 0


### Manual thresholds keep one label per interval ###


def test_manual_thresholds_map_each_interval_to_its_own_label():
    gmm = GMMThresholding(adata=_adata(), feature=FEATURE, label_obs_save_str="labels")

    gmm.categorize_samples(ordered_labels=["low", "high"], manual_thresholds=[149.5])

    expected = ["low"] * 150 + ["high"] * 150
    assert gmm.adata.obs["labels"].tolist() == _expected_in_storage_order(expected)
    boundaries = gmm._internal_data.decision_boundaries
    assert boundaries.interval_components == [0, 1]
    assert boundaries.empty_labels == []
    assert boundaries.n_reassigned_samples == 0


def test_sequential_manual_refinement_stores_one_label_per_interval():
    seq_gmm = SequentialGMM(adata=_adata_with_group())

    seq_gmm.refine_labels_with_manual_thresholds(
        feature=FEATURE,
        obs_label="group",
        value_to_refine="to_refine",
        manual_thresholds=[149.5],
        ordered_labels=["low", "high"],
        operation_name="manual",
    )

    boundaries = seq_gmm.adata.uns[SEQUENTIAL_KEY]["manual"]["decision_boundaries"]
    assert boundaries["interval_components"] == [0, 1]
    assert boundaries["empty_labels"] == []
    assert boundaries["n_reassigned_samples"] == 0


### Storage in .uns and .h5ad files ###


def _write_and_read(adata, tmp_path):
    """Round-trips adata through an .h5ad file."""
    path = tmp_path / "thresholded.h5ad"
    adata.write_h5ad(path)
    return ad.read_h5ad(path)


def test_gmm_thresholding_fields_survive_return_adata_and_h5ad(monkeypatch, tmp_path):
    case = CASES["A_wins_again_in_high_tail"]
    use_fake_gaussian_mixture(
        monkeypatch, FEATURE_VALUES, probs_from_winners(case["winners"], 3)
    )
    gmm = GMMThresholding(adata=_adata(), feature=FEATURE, label_obs_save_str="labels")
    gmm.fit(n_components=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # reassigned samples are expected here
        gmm.categorize_samples(ordered_labels=LABELS)

    adata = gmm.return_adata()
    stored = adata.uns["gmm_thresholding_events"][FEATURE]["decision_boundaries"]
    assert stored == {
        "thresholds": [99.5, 199.5],
        "interval_components": [0, 1, 2],
        "empty_labels": [],
        "n_reassigned_samples": 5,
    }

    read_back = _write_and_read(adata, tmp_path)
    event = read_back.uns["gmm_thresholding_events"][FEATURE]
    boundaries = event["decision_boundaries"]
    assert list(boundaries["interval_components"]) == [0, 1, 2]
    assert len(boundaries["empty_labels"]) == 0
    assert boundaries["n_reassigned_samples"] == 5


def test_sequential_fields_survive_return_adata_and_h5ad(monkeypatch, tmp_path):
    case = CASES["B_never_wins"]
    use_fake_gaussian_mixture(
        monkeypatch, FEATURE_VALUES, probs_from_winners(case["winners"], 3)
    )
    seq_gmm = SequentialGMM(adata=_adata_with_group())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the empty-label warning is expected here
        seq_gmm.refine_labels_with_gmm(
            feature=FEATURE,
            obs_label="group",
            value_to_refine="to_refine",
            n_components=3,
            ordered_labels=LABELS,
            operation_name="refine",
        )

    read_back = _write_and_read(seq_gmm.return_adata(), tmp_path)

    boundaries = read_back.uns[SEQUENTIAL_KEY]["refine"]["decision_boundaries"]
    assert list(boundaries["interval_components"]) == [0, 2]
    assert list(boundaries["empty_labels"]) == ["B"]
    assert boundaries["n_reassigned_samples"] == 0
