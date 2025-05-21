# Sphinx Documentation Guide

A quick guide to using **Sphinx** for writing and organizing your documentation.

## Titles and Headings

Sphinx (via reStructuredText) uses specific characters to denote title levels.

```rst
Main Title
==========

Sub Title
---------
```

## How `toctree` Works

The `toctree` directive creates navigation menus by linking `.rst` files together.

```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   usage
   api
```

This would expect `intro.rst`, `usage.rst`, and `api.rst` to exist in the same folder.

## Making a New Page

1. Create a new `.rst` file, e.g., `faq.rst`
2. Add it to the desired `toctree`:

```rst
.. toctree::

   faq
```

## Including HTML in Sphinx

You can add raw HTML using the `raw` directive:

```rst
.. raw:: html

   <p style="color: red;">This is red text.</p>
```

## Notes and Code Blocks

**Notes**:

```rst
.. note::

   This is a note box.
```

**Code Blocks**:

```rst
.. code-block:: python

   def hello():
       print("Hello, world!")
```

## External and Internal Links

**External**:

```rst
`Python Website <https://www.python.org>`_
```

**Internal** (to a section labeled `usage`):

```rst
:ref:`usage`
```

Make sure to label the target section:

```rst
.. _usage:

Usage Guide
-----------
```

## Building the Docs

After making changes, run:

```bash
make clean
make html
```

To preview, open the generated file:

```
build/html/index.html
```
