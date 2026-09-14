"""Checks on the package metadata in pyproject.toml."""

from pathlib import Path

import numpy as np
import pytest

import cc_mapping
from cc_mapping.feature_selection import RFTopNSelector, _persistence

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture
def poetry():
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["tool"]["poetry"]


def test_package_version_matches_pyproject(poetry):
    assert cc_mapping.__version__ == poetry["version"]


def test_every_optional_dependency_belongs_to_an_extra(poetry):
    """Poetry turns an optional dependency that no extra lists into a hard
    requirement of the wheel."""
    optional = {
        name
        for name, spec in poetry["dependencies"].items()
        if isinstance(spec, dict) and spec.get("optional")
    }
    in_extras = {dep for deps in poetry.get("extras", {}).values() for dep in deps}

    assert optional <= in_extras, f"not in any extra: {optional - in_extras}"


def test_saving_without_skops_names_the_persistence_extra(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    selector = RFTopNSelector(
        n_features=2, rf_params={"n_estimators": 5}, verbose=False
    ).fit(rng.normal(size=(60, 3)), np.repeat(["a", "b"], 30), ["f0", "f1", "f2"])
    monkeypatch.setattr(_persistence, "SKOPS_AVAILABLE", False)

    with pytest.raises(ImportError, match=r"cc-mapping\[persistence\]"):
        selector.save(tmp_path / "selector.skops")
