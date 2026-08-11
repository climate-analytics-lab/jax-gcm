"""ERA5 land-surface monthly climatology from the NCAR RDA mirror (d633001).

Produces the Tier A land-surface product at native 0.25°: a 12-month
climatology of skin temperature, soil temperature and moisture, snow
depth (water equivalent) and albedo, plus the invariant land-sea mask.
Per-grid regridding and translation into jcm forcing variables happens
at bundle assembly.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

RDA_MODA = "/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc"
RDA_INVARIANT = ("/glade/campaign/collections/rda/data/d633000/"
                 "e5.oper.invariant/197901")

# ERA5 GRIB table 128 codes for the fields the jcm forcing needs.
FIELDS = {
    "skt": "128_235_skt",      # skin temperature [K]
    "stl1": "128_139_stl1",    # soil temperature level 1 [K]
    "swvl1": "128_039_swvl1",  # volumetric soil water level 1 [m3/m3]
    "swvl2": "128_040_swvl2",  # volumetric soil water level 2 [m3/m3]
    "swvl3": "128_041_swvl3",  # volumetric soil water level 3 [m3/m3]
    "sd": "128_141_sd",        # snow depth, water equivalent [m]
    "fal": "128_243_fal",      # forecast albedo [1]
}


def _open_year(field_code: str, year: int) -> xr.DataArray:
    path = (f"{RDA_MODA}/{year}/e5.moda.an.sfc.{field_code}.ll025sc."
            f"{year}010100_{year}120100.nc")
    ds = xr.open_dataset(path)
    (name,) = [v for v in ds.data_vars if ds[v].ndim == 3]
    return ds[name]


# Invariant fields: land fraction + low/high vegetation cover (the
# SPEEDY soil-availability formula weights the deep layer by vegetation).
INVARIANTS = {"lsm": "128_172_lsm", "cvl": "128_027_cvl",
              "cvh": "128_028_cvh"}


def load_invariant(name: str) -> xr.DataArray:
    path = (f"{RDA_INVARIANT}/e5.oper.invariant.{INVARIANTS[name]}."
            f"ll025sc.1979010100_1979010100.nc")
    ds = xr.open_dataset(path)
    (var,) = [v for v in ds.data_vars if ds[v].ndim >= 2]
    return ds[var].squeeze(drop=True).rename(name)


def build_climatology(years=range(2005, 2015)) -> xr.Dataset:
    """12-month mean over ``years`` of the five land fields + lsm."""
    out = {}
    for var, code in FIELDS.items():
        acc = None
        for year in years:
            da = _open_year(code, year).load()
            acc = da.values if acc is None else acc + da.values
            template = da
        clim = acc / len(list(years))
        out[var] = xr.DataArray(
            clim, dims=("month", "latitude", "longitude"),
            coords={"month": np.arange(1, 13),
                    "latitude": template.latitude,
                    "longitude": template.longitude},
            attrs=template.attrs)
    ds = xr.Dataset(out)
    for name in INVARIANTS:
        ds[name] = load_invariant(name)
    ds.attrs = {
        "source": "ERA5 monthly means (NCAR RDA d633001), 0.25 deg",
        "climatology_years": f"{min(years)}-{max(years)}",
    }
    return ds
