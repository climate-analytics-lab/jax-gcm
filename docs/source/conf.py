# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../../')) 
import jcm
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Jax-GCM'
copyright = '2025 Jax-GCM team'
author = 'J. Varan Madan, Ellen Davenport, Nicholas Ho, Rebecca Gjini, Duncan Watson-Parris'
release = jcm.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # MyST lets sphinx parse the design/*.md reference docs alongside
    # the .rst pages. Without it the design folder is invisible to
    # readthedocs.
    'myst_parser',
]

# Pick up both reStructuredText and CommonMark/MyST source.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Auto-generate anchors for h1/h2/h3 in markdown so intra-doc links
# like ``[Validation](#validation)`` resolve without hand-written
# ``.. _label:`` blocks.
myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate=True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
