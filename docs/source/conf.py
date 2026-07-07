# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
project = 'DP-EVA'
copyright = '2026, DP-EVA Developers'
author = 'DP-EVA Developers'
version = '0.8.1'
release = '0.8.1'

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
exclude_patterns = [
    'reports/**',
    'plans/**',
    'archive/**',
]
language = 'zh_CN'

# Optional scientific backends are runtime dependencies for specific workflows,
# but docs must build in a lightweight environment.
autodoc_mock_imports = [
    'deepmd',
]

# External services may reject GitHub Actions linkcheck requests even when the
# links are valid in browsers. Keep these citations visible in docs but exclude
# them from CI link probing.
linkcheck_ignore = [
    r'https://github\.com/QuantumMisaka/dpeva/.*',
    r'https://doi\.org/10\.1093/ce/zkag029',
    r'https://academic\.oup\.com/ce/advance-article/doi/10\.1093/ce/zkag029/.*',
]

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
