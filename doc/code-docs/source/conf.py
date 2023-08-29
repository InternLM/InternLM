# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = "InternLM"
copyright = "2023, InternLM Team"
author = "InternLM Team"
release = "v0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "recommonmark",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
]

pygments_style = "sphinx"

# autodoc_pyandtic config
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_field_signature_prefix = " "
autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_summary_list_order = "bysource"
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_field_list_validators = False

templates_path = ["_templates"]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

sys.path.insert(0, os.path.abspath("../../../"))

# Prepend module names to class descriptions
add_module_names = True

autoclass_content = "init"

autodoc_mock_imports = ["apex", "torch"]
