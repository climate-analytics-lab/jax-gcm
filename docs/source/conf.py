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
    # jcm's docstrings are sectioned Google/NumPy style (``Args:``,
    # ``Returns:``). Plain docutils reads such a section as a block quote
    # and then rejects the continuation lines of a multi-line argument
    # description as "Unexpected indentation" — seven of the fourteen
    # problems the strict build reported in #829 were exactly that, in the
    # two dycore class docstrings. Napoleon translates the sections into
    # field lists before docutils sees them, which both clears the errors
    # and renders the arguments as a proper parameter list.
    'sphinx.ext.napoleon',
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
autosummary_generate = True

# The API tree is built with ``:recursive:`` from ``api.rst``. jcm co-locates
# its tests with the modules they cover (``*_test.py``, plus a per-package
# ``conftest.py``), so a naive recursive walk imports and publishes the whole
# test suite. ``_templates/autosummary/module.rst`` filters those leaves out of
# the walk; see the comment at the top of that template for why filtering is
# preferred here over ``autodoc_mock_imports = ['pyses']`` (mocking would hide
# a genuinely broken import of a module we *do* document).
#
# Note there is deliberately no ``suppress_warnings`` entry anywhere in this
# file: ``run_docs.yaml`` builds the full tree with ``-W --keep-going``, and a
# blanket suppression would make that gate meaningless. Fix the source instead.

# Google/NumPy sections are translated to field lists; leave everything else at
# Napoleon's defaults. ``napoleon_use_param`` is the default True, which renders
# ``Args:`` as ``:param:`` fields rather than a single definition list.
napoleon_google_docstring = True
napoleon_numpy_docstring = True


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
