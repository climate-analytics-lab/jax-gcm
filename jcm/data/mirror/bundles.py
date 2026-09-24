"""Assemble per-grid boundary-condition bundles from the Tier A products.

For each Gaussian grid this produces the two files the spectral model
reads directly:

* ``terrain.nc`` — jcm-canonical layout (lowercase vars, ``(lon, lat)``
  axis order, ascending latitudes): GMTED2010 SSO statistics masked with
  the ERA5 invariant land-sea mask.
* ``forcing_<era>.nc`` — ``(lon, lat, time)`` 12-month climatology:
  SST + sea ice from PCMDI-AMIP-1-1-10 (PD 2005–2014; "PI" uses
  1870–1879, the earliest observed decade — flagged in the file attrs
  since no observational 1850 SST exists), land fields from the ERA5
  climatology (identical for both eras).

Unit translations into the conventions the packaged t63 files establish
(and ``_validate_bc_fields`` checks):

* ``sst``   = tos [degC] + 273.15, land filled with nearest-ocean value
* ``icec``  = siconc [%] / 100, land = 0
* ``stl``   = ERA5 stl1 [K]
* ``snowc`` = snow-cover fraction ``min(1, sd·1000/sd2sc)`` — the
  ``jcm.data.bc.compile`` convention (``sd2sc`` = 60 mm w.e. for full
  cover), zeroed where snow persists year-round (ice sheets): their
  high albedo lives in ``alb``, and blending toward fresh-snow albedo
  would darken them
* ``alb``   = per-cell minimum monthly ERA5 fal — the snow-free
  background albedo (snow brightening is applied dynamically from
  ``snowc``; an annual mean would double-count it)
* ``snowc`` on the target grid is the snow-covered fraction of the
  NON-glacier land: the zeroed ice-sheet source points dilute the regridded
  share in a cell that is partly ice sheet, so it is divided by ``1 - glac``
  (:func:`snow_cover_of_non_glacier_land`). Total snow cover of the land is
  ``glac + (1 - glac)·snowc`` (``jcm.forcing.land_snow_cover``)
* ``forest`` = ERA5 high-vegetation cover ``cvh`` as a fraction of the
  land — the forest fraction JSBACH's broadband land albedo masks the snow
  albedo with (static)
* ``glac``  = the permanent-snow (ice-sheet) mask above, as a cell fraction
  — the glacier tiles whose albedo is the ECHAM glacier albedo (static).
  Same mask that zeroes ``snowc``, so a cell's snow is either seasonal
  (``snowc``) or glacier (``glac``), never both
* ``soilw_am`` = SPEEDY availability fraction in [0, 1] per
  ``jcm.data.bc.compile``:
  ``min(1, (swvl1 + veg·3·max(0, swvl2 − swwil)) / (swcap + 3·(swcap − swwil)))``
  with ``veg = cvh + 0.8·cvl``. ERA5's layer depths (7 cm, 21 cm) match
  the formula's 1:3 layer weighting.
* ``soilw_rel`` = ECHAM-like relative soil wetness ``ws/wsmx`` in [0, 1]:
  ``min(1, swvl1 / θ_cap(slt))`` — the 0-7 cm volumetric water content over
  the HTESSEL field capacity of that cell's own soil type. Numerator and
  denominator are water contents of the same layer, so 1 means a soil at its
  own field capacity whatever its texture: that is what ECHAM's ``ws/wsmx``
  means, and what the Tegen dust saturation cut-off is defined against
  (#787). Normalising every texture by one constant instead would read a
  saturated desert sand (θ_cap = 0.244) as 0.70 and never fire the cut-off
  in exactly the cells that emit dust.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from jcm.data.regridding import (conservative_to_gaussian, fill_nearest,
                                 interp_to)

AMIP_ROOT = ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10")
TOS = (f"{AMIP_ROOT}/ocean/mon/tos/gn/v20250807/"
       "tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_"
       "187001-202212.nc")
SICONC = (f"{AMIP_ROOT}/seaIce/mon/siconc/gn/v20250807/"
          "siconc_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_"
          "187001-202212.nc")

ERA_YEARS = {"pd": ("2005-01-01", "2014-12-31"),
             "pi": ("1870-01-01", "1879-12-31")}

# snow/soil conversion constants live with the SPEEDY physics
from jcm.physics.speedy.physical_constants import (sd2sc,  # noqa: E402
                                                   swcap, swwil)

SSO_FIELDS = ("orog", "orostd", "orosig", "orogam", "orothe",
              "oropic", "oroval")

# 12 month-start timestamps: interpolate_to_daily requires pd.infer_freq
# "MS"/"M". The bundle is a manifest ``climatology`` product, so align: auto
# resolves it to WRAP_YEAR from the manifest (#884) and the year itself is
# arbitrary.
CLIMO_TIME = np.array([np.datetime64(f"2014-{m:02d}-01")
                       for m in range(1, 13)])





def _monthly_clim(da: xr.DataArray, era: str) -> xr.DataArray:
    t0, t1 = ERA_YEARS[era]
    return (da.sel(time=slice(t0, t1)).groupby("time.month").mean("time")
            .rename(month="time"))


def _to_lonlat(da2d: xr.DataArray) -> tuple:
    """(lat, lon[, time]) DataArray -> jcm-canonical (lon, lat[, time])."""
    dims = ("lon", "lat") + tuple(d for d in da2d.dims
                                  if d not in ("lat", "lon"))
    return dims, da2d.transpose(*dims).values


#: HTESSEL volumetric field capacity θ_cap [m³/m³] indexed by ERA5's soil-type
#: code ``slt`` (Balsamo et al. 2009, "A revised hydrology for the ECMWF
#: model", Table 1 / IFS documentation Part IV): 1 coarse, 2 medium, 3 medium
#: fine, 4 fine, 5 very fine, 6 organic, 7 tropical organic. Index 0 is ERA5's
#: "no soil" code (ocean) and index 7 reuses the organic entry, which is the
#: closest documented hydrology for the tropical-peat class. Slot 0 keeps the
#: medium value so an ocean cell still divides by a real capacity rather than
#: by zero; dust never reads one (``pot_source`` is zero there).
HTESSEL_THETA_CAP = np.array(
    [0.347, 0.244, 0.347, 0.383, 0.448, 0.541, 0.663, 0.663])


def _field_capacity(slt: xr.DataArray) -> xr.DataArray:
    """Per-cell HTESSEL field capacity [m³/m³] from ERA5's soil-type code."""
    index = np.clip(np.rint(np.asarray(slt.values)).astype(int),
                    0, len(HTESSEL_THETA_CAP) - 1)
    return xr.DataArray(HTESSEL_THETA_CAP[index], dims=slt.dims,
                        coords=slt.coords)


def translate_land(era5: xr.Dataset, permanent_snow: xr.DataArray) -> dict:
    """ERA5 land fields -> jcm ``stl``/``soilw_am``/``soilw_rel``/``snowc``
    (module docstring formulas). The single translation point for
    climatological and transient builders, so the products cannot drift apart.

    ``permanent_snow`` (bool, no time axis) marks cells whose snow never
    melts (ice sheets): their high albedo lives in ``alb`` and blending
    toward fresh-snow albedo would darken them, so ``snowc`` is zeroed
    there. Compute it from a fixed multi-year climatology — a per-year
    minimum flickers with snowy winters and would make ice sheets blink
    in and out of a transient product (#629).

    ``soilw_am`` and ``soilw_rel`` are two different quantities from the same
    ERA5 water, not a refinement of one another: the first is SPEEDY's
    vegetation-weighted root-zone availability index driving land evaporation,
    the second ECHAM's relative wetness of the *saltation* layer driving the
    dust saturation cut-off. Both are published (#787).
    """
    veg = (era5.cvh + 0.8 * era5.cvl).clip(0.0, 1.0)
    soilw = ((era5.swvl1 + veg * 3.0 * (era5.swvl2 - swwil).clip(min=0.0))
             / (swcap + 3.0 * (swcap - swwil))).clip(0.0, 1.0)
    soilw_rel = (era5.swvl1 / _field_capacity(era5.slt)).clip(0.0, 1.0)
    snowc = (era5.sd * 1000.0 / sd2sc).clip(0.0, 1.0).where(
        ~permanent_snow, 0.0)
    return {"stl": era5.stl1, "soilw_am": soilw, "soilw_rel": soilw_rel,
            "snowc": snowc}



def land_cover_fields(era5: xr.Dataset, permanent_snow: xr.DataArray,
                      lats, lons) -> dict:
    """Build the static ``forest`` / ``glac`` fractions on the Gaussian grid.

    The ECHAM land albedo (JSBACH ``update_land_surface_fast``) needs a
    forest fraction and a glacier mask besides ``alb``/``snowc``. ERA5's
    invariant high-vegetation cover ``cvh`` is the forest fraction; the
    glacier mask is the same ``permanent_snow`` definition
    :func:`translate_land` zeroes ``snowc`` with, so the two channels
    partition the snow. The single source for every bundle builder (#672).

    Both are fractions *of the land*, as JSBACH's are: ERA5 carries ``cvh``
    on its land points (``lsm > 0.5``) and zero at sea, so a plain regrid of
    a coastal target cell would dilute the land value with sea zeros. Each
    field is therefore regridded together with the ERA5 land mask and
    divided by it.
    """
    land = (era5.lsm > 0.5).astype(np.float64)
    land_frac = interp_to(land, lats, lons)
    has_land = land_frac > 1e-6

    def per_land(field):
        frac = interp_to(field.astype(np.float64) * land, lats, lons)
        return (frac / land_frac.where(has_land, 1.0)).where(
            has_land, 0.0).clip(0.0, 1.0)

    return {
        "forest": per_land(era5.cvh.clip(0.0, 1.0)),
        "glac": per_land(permanent_snow),
    }


def snow_cover_of_non_glacier_land(snowc: xr.DataArray,
                                   glac: xr.DataArray) -> xr.DataArray:
    """Regridded ``snowc`` -> snow-covered fraction of the non-glacier land.

    :func:`translate_land` zeroes ``snowc`` on permanent-snow source points,
    so after regridding a cell that is partly ice sheet carries its seasonal
    snow as a share of ALL its land. The jcm convention (``ForcingData``) is
    the share of the non-glacier land, JSBACH's tiling, so divide by
    ``1 - glac``; an all-glacier cell has no such land and gets 0.
    """
    open_land = 1.0 - glac
    has_open = open_land > 1e-6
    return (snowc / open_land.where(has_open, 1.0)).where(
        has_open, 0.0).clip(0.0, 1.0)


_EMIS_SPECIES = ("so2", "bc", "oc")
_ANTHRO_SECTORS = ("surface_combustion", "elevated_industrial", "shipping")


def build_emissions_nc(ceds_zarr: str, bb_zarr: str, era: str,
                       lats, lons, out_path: str) -> None:
    """Per-grid emissions file keyed ``emis_<super_sector>_<species>``.

    All four model super-sectors (see
    ``jcm.physics.aerosol.jam.emissions.sectors``) — the three CEDS
    anthropogenic groups keep their distinct injection altitudes
    (elevated_industrial ~50 m) plus biomass burning.
    """
    ceds = xr.open_zarr(ceds_zarr)
    bb = xr.open_zarr(bb_zarr)
    ds = xr.Dataset(coords={"lat": lats, "lon": lons,
                            "time": CLIMO_TIME})
    for sp in _EMIS_SPECIES:
        up = sp.upper()
        channels = [(sector, ceds[f"{up}_{sector}_{era}_clim"])
                    for sector in _ANTHRO_SECTORS]
        channels.append(("biomass_burning", bb[f"{up}_{era}_clim"]))
        for prefix, da in channels:
            da = da.load()
            arr = conservative_to_gaussian(
                np.nan_to_num(da.values), da.lat.values, da.lon.values,
                lats, lons)
            ds[f"emis_{prefix}_{sp}"] = (
                ("time", "lon", "lat"), arr.transpose(0, 2, 1),
                {"units": "kg m-2 s-1"})
    ds.attrs = {
        "title": ("jax-gcm prescribed emissions (bulk per-super-sector "
                  "surface flux)"),
        "era": era,
        "source": "CEDS-CMIP-2025-04-18 + DRES-CMIP-BB4CMIP7-2-0",
    }
    ds.to_netcdf(out_path)
    print("wrote", out_path, flush=True)


def build_terrain(sso_path: str, era5_path: str, out_path: str) -> None:
    """Gaussian terrain.nc: GMTED SSO fields + fractional ERA5 land mask.

    ``lsm`` stays a fraction — ``TerrainData``'s ``fmask`` is consumed
    fractionally (coastal flux blending) and the packaged climatology is
    fractional too. SSO fields are zeroed only below 10% land (matching
    ``get_terrain``'s snap threshold): open-ocean cells lose the
    shoreline-step artifacts of the DEM while islands and coasts keep
    their orography.
    """
    sso = xr.open_dataset(sso_path)
    era5 = xr.open_dataset(era5_path)
    lats, lons = sso.lat.values, sso.lon.values
    lsm_frac = np.clip(interp_to(era5.lsm, lats, lons).values, 0.0, 1.0)
    keep = lsm_frac >= 0.1
    ds = xr.Dataset(coords={"lat": lats, "lon": lons})
    ds["lsm"] = (("lon", "lat"), lsm_frac.T)
    for name in SSO_FIELDS:
        ds[name] = (("lon", "lat"), np.where(keep, sso[name].values, 0.0).T)
    ds.attrs = dict(sso.attrs)
    ds.attrs["lsm_source"] = "ERA5 invariant land fraction (0.25 deg)"
    ds.to_netcdf(out_path)
    print("wrote", out_path, flush=True)


def build_forcing(era5_path: str, era: str, lats, lons,
                  out_path: str) -> None:
    era5 = xr.open_dataset(era5_path).rename(month="time")

    tos = xr.open_dataset(TOS).tos
    sic = xr.open_dataset(SICONC).siconc
    sst_c = _monthly_clim(tos, era)
    sst = fill_nearest(sst_c.values, sst_c.lat.values, sst_c.lon.values)
    sst_da = xr.DataArray(sst + 273.15, dims=("time", "lat", "lon"),
                          coords={"time": CLIMO_TIME,
                                  "lat": sst_c.lat, "lon": sst_c.lon})
    icec_c = _monthly_clim(sic, era) / 100.0
    icec_da = icec_c.fillna(0.0)

    # For a climatology the ice-sheet mask comes from its own window (a
    # fixed multi-year period, as translate_land requires).
    permanent_snow = era5.sd.min("time") >= 0.1
    land = translate_land(era5, permanent_snow=permanent_snow)

    fields = {
        "sst": interp_to(sst_da, lats, lons),
        "icec": interp_to(icec_da, lats, lons).clip(0.0, 1.0),
        "stl": interp_to(land["stl"], lats, lons),
        "soilw_am": interp_to(land["soilw_am"], lats, lons).clip(0.0, 1.0),
        "soilw_rel": interp_to(land["soilw_rel"], lats, lons).clip(0.0, 1.0),
        "snowc": interp_to(land["snowc"], lats, lons).clip(0.0, 1.0),
        "alb": interp_to(era5.fal.min("time"), lats, lons),
        **land_cover_fields(era5, permanent_snow, lats, lons),
    }
    fields["snowc"] = snow_cover_of_non_glacier_land(fields["snowc"],
                                                     fields["glac"])
    ds = xr.Dataset(coords={"lat": lats, "lon": lons,
                            "time": CLIMO_TIME})
    for name, da in fields.items():
        ds[name] = _to_lonlat(da)
    ds.attrs = {
        "source": ("SST/ice: PCMDI-AMIP-1-1-10; land: ERA5 monthly "
                   "climatology 2005-2014"),
        "era": era,
        "note": ("PI SST/ice is the 1870-1879 mean — the earliest "
                 "observed decade, not a true 1850 state"
                 if era == "pi" else "PD climatology 2005-2014"),
    }
    ds.to_netcdf(out_path)
    print("wrote", out_path, flush=True)
