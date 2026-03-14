# Configuration file for the Sphinx documentation builder.
import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
project = 'DP-EVA'
copyright = '2026, DP-EVA Developers'
author = 'DP-EVA Developers'
version = '0.6.6'
release = '0.6.6'

# -- Path setup --------------------------------------------------------------
# Add src to sys.path to allow autodoc to find modules
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.autodoc_pydantic',
    'sphinxcontrib.mermaid',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []
language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "DP-EVA Documentation"

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3

# -- Autodoc Pydantic configuration ------------------------------------------
autodoc_pydantic_model_show_json = False
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_field_show_default = True
autodoc_pydantic_field_show_required = True
