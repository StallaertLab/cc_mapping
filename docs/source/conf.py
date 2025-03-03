from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../..", "src").resolve()))

import importlib.metadata

project = "cc_mapping"
copyright = "2024, Dante Poe"
author = "My Name"
version = release = importlib.metadata.version("cc_mapping")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "nbsphinx",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "scanpydoc"
html_theme_options = {
    "use_repository_button": True,
    "repository_url": "https://github.com/scverse/anndata",
    "repository_branch": "main",
    "navigation_with_keys": False,  # https://github.com/pydata/pydata-sphinx-theme/issues/1492
}

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

nbsphinx_execute = "auto"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png2x'}",
    "--InlineBackend.rc=figure.dpi=96",
]

nbsphinx_kernel_name = "python3"

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
