"""Grid-free volcanic SO₂ product (#1190): CMIP6 continuous degassing + CMIP7 eruptions.

Both sources are stored as point lists, so :func:`jcm.forcing.read_volcanic_emissions`
maps them onto whatever grid a run uses (Gaussian or pySES columns) when the
forcing loads; nothing here depends on a model grid.

**Continuous degassing** is the CMIP6/CESM2 ``contvolcano`` file, built by
NCAR's ``create_files.ncl`` from Diehl's ``volc_continuous.nc`` (after Andres &
Kasgnoc 1998) scaled by 2.6325 to the AeroCom continuous total, 12.6 Tg S/yr
(Textor's ×1.5 for an underestimate, applied to the SO₂ plus non-SO₂ sulfur).
The script spreads each volcano evenly over the 500 m layers between two
thirds of its vent altitude and the vent, and stores the layers as pressures
``1013.25·exp(−z/7600)`` hPa over a fixed 1013.25 hPa surface. The file is
constant in time and carries 97.5 % of the sulfur as SO₂, in molecules cm⁻³
s⁻¹ over each layer's volume; the other 2.5 % is in its separate SO₄ files.

The product inverts that construction exactly: each non-zero 0.9°×1.25° cell
becomes one source whose rate per layer [kg s⁻¹] is the inventory's TOTAL
sulfur expressed as SO₂, the share the model emits as primary sulfate being
its own parameter. The layer interfaces are the script's altitudes a.s.l.

**Eruptions** are the CMIP7 input4MIPs ``utsvolcemis`` list (UOEXETER-CMIP-2-2-1;
Aubry et al. 2026, GMD preprint egusphere-2026-546), copied as published: day,
volcano position, injection altitude a.s.l. and depth, and SO₂ mass per eruption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import xarray as xr

#: Mirror product name (under ``products/``).
VOLCANIC_PRODUCT = "volcanic_so2_cmip.nc"

#: The degassing file, relative to a CESM inputdata root.
DEGASSING_RELPATH = (
    "atm/cam/chem/emis/CMIP6_emissions_1750_2015/"
    "emissions-cmip6_SO2_contvolcano_vertical_850-5000_0.9x1.25_c20170724.nc")

#: The eruption list, relative to an input4MIPs root.
ERUPTIONS_RELPATH = (
    "CMIP7/CMIP/uoexeter/UOEXETER-CMIP-2-2-1/atmos/day/utsvolcemis/gn/"
    "v20250521/utsvolcemis_input4MIPs_emissions_CMIP_UOEXETER-CMIP-2-2-1_gn_"
    "17500101-20231120.nc")

# The constants create_files.ncl built the degassing file with, so the inverse
# returns the inventory's own kg/s.
_AVOGADRO = 6.02214e23       # mol⁻¹
_SO2_MOLAR_MASS = 64.0e-3    # kg mol⁻¹
_EARTH_RADIUS = 6.37122e6    # m (CESM's shr_const_rearth)
_SCALE_HEIGHT = 7600.0       # m
_P_SURFACE = 101325.0        # Pa
_SO2_SHARE = 0.975           # the file's SO2 share; 2.5 % went to its SO4 files


def scale_height_altitude(pressure):
    """Altitude a.s.l. [m] the degassing file encoded as ``pressure`` [Pa].

    ``create_files.ncl`` wrote its 500 m layers as ``1013.25·exp(−z/7600)`` hPa.
    """
    return -_SCALE_HEIGHT * np.log(np.asarray(pressure) / _P_SURFACE)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def degassing_sources(ds: xr.Dataset) -> xr.Dataset:
    """Point sources from the CESM ``contvolcano`` file ``ds``.

    Returns the non-zero cells as sources with their per-layer rate of total
    volcanic sulfur as SO₂ [kg s⁻¹] and the layers' altitude interfaces
    [m a.s.l.], bottom first. Raises if the file is not the time-constant,
    fixed-surface-pressure field this conversion assumes.
    """
    ps = np.asarray(ds["PS"].values)
    if not np.allclose(ps, _P_SURFACE, atol=1.0):
        raise ValueError("contvolcano PS is not the fixed 1013.25 hPa this "
                         "conversion assumes")
    field = ds["contvolcano"]
    first = np.asarray(field.isel(time=0).values)
    for t in range(1, ds.sizes["time"]):
        if not np.array_equal(np.asarray(field.isel(time=t).values), first):
            raise ValueError(f"contvolcano varies in time (record {t}); the "
                             "product stores one constant field")
    p0 = float(ds["P0"])
    # ilev runs top first (271.8 -> 1013 hPa); the product is bottom first.
    p_int = np.asarray(ds["hyai"].values) * p0 + np.asarray(ds["hybi"].values) * _P_SURFACE
    z_int = scale_height_altitude(p_int)[::-1]
    dz_cm = np.diff(z_int) * 100.0                      # (layer,), bottom first
    molec = first[::-1]                                 # (layer, lat, lon), bottom first

    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    # The script's cell area, r²·cos(lat)·dlat·dlon.
    dlat = np.deg2rad(float(lat[2] - lat[1]))
    dlon = np.deg2rad(float(lon[2] - lon[1]))
    area_cm2 = (_EARTH_RADIUS ** 2 * np.cos(np.deg2rad(lat)) * dlat * dlon) * 1.0e4

    kg_per_molec = _SO2_MOLAR_MASS / _AVOGADRO
    rate = (molec * dz_cm[:, None, None] * area_cm2[None, :, None]
            * kg_per_molec / _SO2_SHARE)
    ilat, ilon = np.nonzero(rate.sum(axis=0) > 0.0)
    return xr.Dataset(
        {
            "degassing_lat": ("degassing_source", lat[ilat]),
            "degassing_lon": ("degassing_source", np.mod(lon[ilon], 360.0)),
            "degassing_so2": (("degassing_source", "degassing_layer"),
                              rate[:, ilat, ilon].T),
            "degassing_altitude_interface": ("degassing_interface", z_int),
        })


def eruption_list(ds: xr.Dataset) -> xr.Dataset:
    """Return the CMIP7 ``utsvolcemis`` eruptions in ``ds``, as published.

    Raises unless every eruption lasts exactly one day, the duration the
    dataset's injection recommendation (and the model term) assumes.
    """
    t0 = ds["time_bnds"].values[:, 0]
    t1 = ds["time_bnds"].values[:, 1]
    if not np.all((t1 - t0) == np.timedelta64(1, "D")):
        raise ValueError("utsvolcemis eruptions are not all one day long")
    return xr.Dataset(
        {
            "eruption_time": ("eruption", t0.astype("datetime64[ns]")),
            "eruption_lat": ("eruption", np.asarray(ds["lat"].values, dtype=float)),
            "eruption_lon": ("eruption",
                             np.mod(np.asarray(ds["lon"].values, dtype=float), 360.0)),
            "eruption_height": ("eruption", np.asarray(ds["height"].values, dtype=float)),
            "eruption_depth": ("eruption", np.asarray(ds["depth"].values, dtype=float)),
            "eruption_so2": ("eruption", np.asarray(ds["utsvolcemis"].values, dtype=float)),
        })


_ATTRS = {
    "degassing_lat": {"units": "degrees_north",
                      "long_name": "degassing source cell centre latitude"},
    "degassing_lon": {"units": "degrees_east",
                      "long_name": "degassing source cell centre longitude"},
    "degassing_so2": {"units": "kg s-1",
                      "long_name": "continuous volcanic degassing, total sulfur "
                                   "as SO2 mass, per source cell and layer"},
    "degassing_altitude_interface": {
        "units": "m", "positive": "up",
        "long_name": "degassing layer interface altitude above sea level, "
                     "bottom first"},
    "eruption_lat": {"units": "degrees_north", "long_name": "volcano latitude"},
    "eruption_lon": {"units": "degrees_east", "long_name": "volcano longitude"},
    "eruption_height": {"units": "m",
                        "long_name": "SO2 injection altitude above sea level"},
    "eruption_depth": {"units": "m",
                       "long_name": "SO2 injection depth: the profile is "
                                    "exp(-((z - height)/depth)^2)"},
    "eruption_so2": {"units": "kg", "long_name": "SO2 mass emitted by the eruption"},
    "eruption_time": {"long_name": "start of the eruption day (UTC); the SO2 is "
                                   "injected over 24 hours"},
}


def build_volcanic_product(degassing_path, eruptions_path) -> xr.Dataset:
    """Assemble the grid-free volcanic SO₂ product from its two source files."""
    with xr.open_dataset(degassing_path, decode_times=False) as ds:
        degassing = degassing_sources(ds)
    with xr.open_dataset(eruptions_path) as ds:
        eruptions = eruption_list(ds)
    out = xr.merge([degassing, eruptions])
    for name, attrs in _ATTRS.items():
        out[name].attrs.update(attrs)
    total = float(out["degassing_so2"].sum()) * 365.25 * 86400.0 / 1.0e9
    out.attrs.update({
        "title": "jax-gcm volcanic SO2: CMIP6 continuous degassing and CMIP7 "
                 "eruptions as point sources",
        "degassing_source": f"{Path(degassing_path).name} (sha256 "
                            f"{_sha256(degassing_path)})",
        "degassing_note": "AeroCom continuous degassing (Diehl's "
                          "volc_continuous.nc after Andres & Kasgnoc 1998, x2.6325 "
                          "to 12.6 Tg S/yr) as CESM2 builds it; constant in time; "
                          f"total sulfur {total:.2f} Tg SO2/yr, of which the "
                          "file carries 97.5 % as SO2",
        "eruptions_source": f"{Path(eruptions_path).name} (sha256 "
                            f"{_sha256(eruptions_path)})",
        "eruptions_note": "CMIP7 utsvolcemis UOEXETER-CMIP-2-2-1 (Aubry et al. "
                          "2026, egusphere-2026-546), copied as published",
        "history": "jcm.data.mirror.volcanic.build_volcanic_product",
    })
    return out
