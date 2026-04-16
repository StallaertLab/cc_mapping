# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("."))

# Mock optional dependencies that may not be installed in the docs environment
autodoc_mock_imports = ["phate"]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cc-mapping"
copyright = "2025, ddpoe"
author = "ddpoe"

version = "0.2.0"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpydoc",  # Use numpydoc instead of napoleon
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
    # 'scanpydoc',  # Remove scanpydoc to avoid duplication
]

# Intersphinx mapping to link to other docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

# Autosummary settings
autosummary_generate = True
add_module_names = False  # Show short names in autosummary tables

# NBSphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Continue even if there are errors

# Numpydoc settings (like scikit-learn)
numpydoc_show_class_members = False  # Don't show class members in a separate table
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False  # Show members in-page instead of separate pages

# Napoleon settings for Google/NumPy style docstrings
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False  # Don't include __init__ in documentation

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__init__",  # Exclude __init__ from documentation
}

# Type hints
typehints_fully_qualified = False
always_document_param_types = True

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "cc-mapping"
html_logo = "_static/logo.png"

# PyData theme options (used by scikit-learn)
html_theme_options = {
    "logo": {
        "text": "cc-mapping",
    },
    "github_url": "https://github.com/StallaertLab/cc_mapping",
    "collapse_navigation": False,
    "navigation_depth": 4,
    "navbar_align": "left",
    "show_nav_level": 2,
    "show_toc_level": 2,
}

# Control how autosummary items appear in the TOC
toc_object_entries_show_parents = "hide"

# Add GitHub links
html_context = {
    "display_github": True,
    "github_user": "StallaertLab",
    "github_repo": "cc_mapping",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Generate API reference pages from templates ----------------------------
import jinja2  # noqa: E402
from pathlib import Path  # noqa: E402
from api_reference import API_REFERENCE  # noqa: E402

# Define templates for API reference pages
rst_templates = [
    (
        "api/index",
        "api/index",
        {
            "API_REFERENCE": sorted(API_REFERENCE.items(), key=lambda x: x[0]),
        },
    ),
]

# Add each module's API reference page
for module in API_REFERENCE:
    rst_templates.append(
        (
            "api/module",
            f"api/{module}",
            {"module": module, "module_info": API_REFERENCE[module]},
        )
    )

# Generate RST files from templates
for rst_template_name, rst_target_name, kwargs in rst_templates:
    template_path = (
        Path(__file__).parent / "templates" / f"{rst_template_name}.rst.template"
    )
    target_path = Path(__file__).parent / f"{rst_target_name}.rst"

    if template_path.exists():
        with template_path.open("r", encoding="utf-8") as f:
            t = jinja2.Template(f.read())

        with target_path.open("w", encoding="utf-8") as f:
            f.write(t.render(**kwargs))
