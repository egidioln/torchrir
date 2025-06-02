import sys
import os


sys.path.append(os.path.abspath(__file__ + "/../.."))
# Configuration file for the Sphinx documentation builder.

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "torchRIR"
copyright = "2025, Lucas Egidio"
author = "Lucas Egidio"
release = "0.0.1-dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "autodoc2",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = []

autodoc2_packages = [
    {
        "path": "../../torchrir",
        "auto_mode": True,
    },
]
autodoc2_hidden_objects = ["undoc", "dunder", "private", "inherited"]

autodoc2_docstring_parser_regexes = [
    # this will render all docstrings as Markdown
    (r".*", "source.docstrings_parser"),
]
myst_enable_extensions = [
    "dollarmath",
    "fieldlist",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
# import solar_theme

# html_theme = "solar_theme"
# html_theme_path = [solar_theme.theme_path]

# html_theme = "stanford_theme"
# html_theme_path = [sphinx_theme.get_html_theme_path("stanford-theme")]
html_theme = "shibuya"
html_static_path = ["_static"]
