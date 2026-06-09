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

project = 'JAX-GCM'
copyright = '2025 JAX-GCM team'
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


# -- Generated content -------------------------------------------------------
# The SPEEDY variable-translation page is generated from
# ``jcm/physics/speedy/units_table.csv`` at *build time* (here), rather than
# being committed to the repo by a CI job (which bypassed the PR rule — see
# issue #394). Read the Docs, ``make html`` and any CI build all regenerate it
# in sync with the CSV; the output is gitignored.

def setup(app):
    """Regenerate ``speedy_translation.rst`` from the units CSV before reading sources."""
    docs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # docs/
    if docs_dir not in sys.path:
        sys.path.insert(0, docs_dir)
    import generate_docs

    app.connect('builder-inited', lambda _app: generate_docs.generate())


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
