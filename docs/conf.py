# Configuration file for the Sphinx documentation builder.
#
import os, sys, pathlib
#
# Add the project src/ to Python path so autodoc can import the package
sys.path.insert(0, os.path.abspath('../src/'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TorchTDSCHA'
copyright = '2025, Francesco Libbi'
author = 'Francesco Libbi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Numpy/Google docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",         # LaTeX math
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",     # gh-pages support
    "sphinx_autodoc_typehints",   # types in signatures
    "myst_parser",                # Markdown support
    "sphinx_copybutton",          # nice copy buttons on code blocks
]
autosummary_generate = True
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# Optional theme tweaks
html_title = "TorchTDSCHA"
html_theme_options = {
    # Furo examples:
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    # RTD examples (if using rtd theme):
    # "collapse_navigation": False,
    # "navigation_depth": 4,
    # "style_nav_header_background": "#222",
}

# Nice code highlighting
pygments_style = "default"
pygments_dark_style = "native"

# Inter-project links (optional)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
