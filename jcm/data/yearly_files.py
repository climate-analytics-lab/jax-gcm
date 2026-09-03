"""``{year}`` file-pattern expansion for the yearly transient forcing bundles.

The transient AMIP/ERA5 bundles are laid out one file per year (issue #610:
download only what you run, append new years without rewriting history), so a
config points at a ``{year}`` pattern plus an inclusive ``forcing.years`` range
and the loader expands it to the concrete yearly files.

This module is deliberately free of any intra-package (``jcm``) import — the
same invariant :mod:`jcm.data.bundle_names` maintains — so it can be loaded in
isolation:

* :mod:`jcm.forcing` re-exports :func:`expand_yearly_files` (its historical
  home), and :mod:`jcm.runners` imports it from there, so the model build path
  is unchanged.
* ``tools/benchmark.py`` loads it **by file path** (exactly as it loads
  ``jcm/data/bundle_names.py`` and ``jcm/data/remote.py``) to expand the same
  patterns for its pre-GPU prefetch **without importing** ``jcm`` — importing
  the package initialises a JAX backend, which preallocates the GPU before the
  free-card gate. ``jcm.forcing`` itself imports JAX/dinosaur/``jcm`` at module
  top, so it cannot be the shared file-path-loaded source; this leaf can.

Keeping the expansion here means the build-time resolver and the prefetch
enumerator cannot drift apart.
"""

from __future__ import annotations


def expand_yearly_files(file_spec, years, available=None):
    """Expand a ``{year}`` file pattern into the yearly-bundle file list.

    The transient AMIP bundles are one file per year (issue #610:
    download only what you run, append new years without rewriting
    history), so config points at a pattern plus an inclusive range:
    ``file: hf://bundles/t63/forcing_amip/{year}.nc`` with
    ``years: [1979, 1983]``. A pattern without ``years`` raises rather
    than silently running with a literal ``{year}`` path. Non-pattern
    specs (plain paths, lists, ``None``) pass through untouched even when
    ``years`` is set — a run may mix yearly SST files with a static dust
    climatology, all sharing one ``forcing.years`` range.

    ``available`` (``forcing.available_years``, the product's inclusive
    source coverage) widens the expansion by one year on each side,
    clipped to that coverage: the yearly files hold *mid-month* samples,
    so a run starting Jan 1 needs the previous December's sample (and a
    run ending Dec 31 the next January's) for ``by_date_interp`` to
    bracket the boundary instead of clamping to the nearest mid-month
    value for ~half a month.

    This expands a **single** product (one scalar spec). A ``{year}``
    pattern becomes that product's list of yearly files — one product
    concatenated along a single time axis downstream. A **list** spec
    (e.g. ``emissions_file`` carrying a biomass-burning product plus an
    anthropogenic one) names *several* products and is not flattened here:
    :func:`jcm.runners._forcing_products` splits it and expands each element
    through this function, so each product is opened and time-aligned on its
    own (a transient ``{year}`` product and a 12-month climatology in the same
    list must not share one time axis — see that function).
    """
    has_pattern = isinstance(file_spec, str) and "{year}" in file_spec
    if not has_pattern:
        return file_spec
    if years is None:
        raise ValueError(
            f"forcing file pattern {file_spec!r} contains {{year}} but "
            "no year range is set — add e.g. forcing.years=[1979,1983]")
    first, last = int(years[0]), int(years[-1])
    if last < first:
        raise ValueError(f"forcing.years range is reversed: {years!r}")
    if available is not None:
        lo, hi = int(available[0]), int(available[-1])
        first, last = max(first - 1, lo), min(last + 1, hi)
        # A requested range entirely outside coverage would invert here
        # and expand to nothing; clamp to the nearest edge file instead
        # (the time lookup then clamps to its first/last sample).
        first, last = min(first, hi), max(last, lo)
    return [file_spec.format(year=y) for y in range(first, last + 1)]
