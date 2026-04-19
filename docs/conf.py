# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'deepSTRF'
copyright = '2025, Ulysse Rancon'
author = 'Ulysse Rancon'
release = '0.0.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'nbsphinx',
              'sphinx.ext.mathjax',  # Optional, for math support
              'sphinx.ext.viewcode' # Optional, shows code
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/deepstrf_logo.png"


# Markdown is handled by myst_parser; .ipynb by nbsphinx.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'jupyter_notebook',
}

# Don't re-execute notebooks at build time (we commit pre-run outputs).
nbsphinx_execute = 'never'



# -- Options for autodoc -------------------------------------------------
# https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-config.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
#sys.path.insert(0, os.path.abspath(os.path.join('..', 'deepSTRF')))
