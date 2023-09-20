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

with open("../../../version.txt", "r") as f:
    release = f.readline().rstrip()

master_doc = "index"

autodoc_member_order = "bysource"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "myst_parser",
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

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ["_templates"]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

# GitHub integration
html_context = {
    "display_github": True,
    "github_user": "InternLM",
    "github_repo": "InternLM",
    "github_version": "main",
    "conf_py_path": "/doc/code-docs/source/",
}

sys.path.insert(0, os.path.abspath("../../../"))

# Prepend module names to class descriptions
add_module_names = True

autoclass_content = "class"

autodoc_mock_imports = [
    "apex",
    "torch",
    "numpy",
]

# support multi-language docs
language = "zh_CN"
locale_dirs = ["../locales/"]  # path is example but recommended.
gettext_compact = False  # optional.
gettext_uuid = False  # optional.
