# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..')) # . is docs/source

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ot_markov_distances'
copyright = '2023, Tristan Brugère'
author = 'Tristan Brugère'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "myst_parser", "sphinxcontrib.bibtex", 'sphinx_math_dollar', 'sphinx.ext.mathjax']
napoleon_custom_sections = [('Returns', 'params_style')]
autodoc_member_order = 'bysource'
autoclass_content = "both"
bibtex_bibfiles = ['bibliography.bib']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
