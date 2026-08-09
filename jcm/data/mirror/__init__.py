"""Maintainer-side builders for the hosted forcing/boundary-condition mirror.

Everything here runs on NCAR Glade against local source data (CESM
``inputdata``, ``input4MIPs_raw``, RDA ERA5, one GMTED2010 download) and
produces the artifacts hosted on Hugging Face:

* Tier A — grid-independent processed products (super-sectored CEDS+BB
  emissions at 0.5°, AMIP SST/ice, FZJ ozone, ERA5 land climatology,
  dust/DMS sources) as chunked float32 zarr.
* Tier B — ready-to-run per-grid bundles (t63/t106 x l47/l95, ne30):
  ``terrain.nc``, ``forcing.nc``, ``ozone.nc``, ``oxidants.nc``,
  ``emissions.nc`` on the exact model grid.

Users never run these: they pull artifacts via ``jcm.data.remote``.
See ``SOURCES.md`` for provenance and ``registry.json`` for hashes.
"""
