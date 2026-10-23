"""Generate the synthetic CAM6 SOAG fixture used by ``jcm/data/mirror``'s
offline-network tests (#1152).

The three generated files (``anthro.nc``, ``biogenic.nc``, ``bb.nc``) are
NOT a subset of NCAR's real inventory -- they are built from scratch here,
entirely synthetically, with the same variable name, units attribute,
dimension order and historical time coverage that
:func:`jcm.data.emissions.cam6_soa.prepare_cam6_soa` requires of the real
``svn-ccsm-inputdata.cgd.ucar.edu`` files it names in ``HISTORICAL_SOURCES``.
Passing this fixture's paths as ``prepare_cam6_soa``'s ``sources=`` argument
therefore exercises the exact same fetch/validate/regrid code path the real
files do (:func:`jcm.data.emissions.downloader.fetch` returns a local path
verbatim, so no network call happens), without shipping any real NCAR data
or touching the network at test time.

Grid: a regular 4x8 lat/lon grid (``prepare_speciated_emissions`` falls back
to a unit-area grid for the ``(lat, lon)``-trailing regular case the real
CAM6 SOAG files use -- see its ``area_name`` branch), as the SAME minimal
shape the existing ``jcm.data.emissions.cam6_soa_test`` unit tests already
use, scaled up slightly so the conservative regridder sees more than one
cell per model-grid latitude band.

Time coverage: exactly the two decades
:func:`jcm.data.mirror.bundles.add_cam6_soa` ever requests --
1850-1859 (the "pi" era) and 2005-2014 (the "pd" era) -- at one mid-month
record per month (240 records total), rather than the real files' full
1750-2015 span, since nothing under test ever looks outside those decades
and keeping the coverage minimal is most of why the fixture fits in a few
KB rather than the ~100s of MB the real files are.

Regenerate (only needed if the fixture's grid/time coverage must change)::

    python -m jcm.data.test.cam6_soag_fixture.generate_fixture
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import xarray as xr

#: The three CAM6 SOAG source categories ``prepare_cam6_soa`` requires --
#: see ``jcm.data.emissions.cam6_soa.SOURCES``. A different nonzero flux
#: level per category (rather than one shared RNG stream) makes a test that
#: accidentally mixed up which file is which fail loudly instead of by
#: coincidence matching.
_CATEGORIES = {"anthro": 1.0e10, "biogenic": 2.0e10, "bb": 3.0e10}
_NLAT, _NLON = 4, 8
_YEARS = (*range(1850, 1860), *range(2005, 2015))  # the "pi"/"pd" decades
_SEED = 1152
# One fixed offset per category so each file draws an independent stream
# from the documented seed (recorded in ``provenance.json``).
_CATEGORY_SEED_OFFSET = {"anthro": 1, "biogenic": 2, "bb": 3}


def _time_axis() -> tuple[np.ndarray, dict]:
    """240 mid-month day-offsets (``1850-1859`` + ``2005-2014``) plus the
    CF ``units``/``calendar`` attrs :func:`xarray.decode_cf` needs to read
    them back into calendar years/months, matching the check
    ``prepare_cam6_soa`` applies to the real historical files.
    """
    epoch = datetime.date(1850, 1, 1)
    days = [
        (datetime.date(year, month, 15) - epoch).days
        for year in _YEARS for month in range(1, 13)
    ]
    attrs = {"units": "days since 1850-01-01", "calendar": "gregorian"}
    return np.asarray(days, dtype=np.float64), attrs


def build_dataset(category: str, flux_scale: float) -> xr.Dataset:
    """One category's synthetic SOAG source, shaped like the real file."""
    # Fixed per-category offsets, not ``hash(category)``: Python salts str
    # hashes per interpreter, so a hash-derived seed would give a different
    # payload on every regeneration and the committed files could never be
    # reproduced from this generator and ``provenance.json``.
    rng = np.random.default_rng(_SEED + _CATEGORY_SEED_OFFSET[category])
    lat = np.linspace(-67.5, 67.5, _NLAT)
    lon = np.linspace(0.0, 360.0, _NLON, endpoint=False)
    time, time_attrs = _time_axis()
    data = rng.uniform(0.5 * flux_scale, 1.5 * flux_scale,
                       (time.size, _NLAT, _NLON)).astype(np.float64)
    ds = xr.Dataset(
        {f"emiss_{category}": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["time"].attrs.update(time_attrs)
    ds[f"emiss_{category}"].attrs.update(
        units="molecules/cm2/s",
        long_name=f"synthetic CAM6 SOAG {category} fixture (#1152, not real NCAR data)",
    )
    ds.attrs.update(
        title=f"Synthetic CAM6 SOAG {category} fixture for offline jcm tests",
        provenance="jcm/data/test/cam6_soag_fixture/generate_fixture.py (#1152)",
    )
    return ds


def generate() -> None:
    out_dir = Path(__file__).parent
    for category, flux_scale in _CATEGORIES.items():
        ds = build_dataset(category, flux_scale)
        ds.to_netcdf(out_dir / f"{category}.nc")
        print("wrote", out_dir / f"{category}.nc", flush=True)


if __name__ == "__main__":
    generate()
