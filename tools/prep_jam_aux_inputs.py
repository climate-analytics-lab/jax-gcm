"""Prepare JAM natural-emission / oxidant aux inputs from CESM inputdata.

Run ONCE on a login node (needs the on-disk CESM inputdata tree). Produces
three small netCDFs in the jcm reader-contract layouts (see
``jcm.forcing.read_dms_seawater`` / ``read_dust_source`` /
``read_oxidant_vmr``), which the pySES ``build_forcing`` then interpolates
onto the physics columns at model-build time — so the files stay on their
native lon/lat grids here.

Sources (all under /glade/campaign/cesm/cesmdata/inputdata):

* DMS seawater concentration: ``Csw_DMS_Lana2011_f09f09_1750_2100`` — the
  Lana et al. (2011) surface-ocean climatology in nmol/L (verified constant
  across years; any 12-month block is *the* climatology).
* Dust erodibility: ``dst_1.9x2.5_c090203.nc`` ``mbl_bsn_fct_geo`` — CAM's
  static geomorphic mobilization-basin factor, used as the [0, 1]
  potential-source map ``DustEmissions`` expects.
* Oxidants: ``oxid_1.9x2.5_L26_1850-2015`` OH/NO3/O3/H2O2 [mol/mol] on CAM
  L26 hybrid levels, sampled as 12-month blocks per selected year (1849,
  1855, ..., 2005, 2015). The block nearest ``--year`` is vertically
  interpolated in log-pressure onto the ECHAM L47 hybrid grid the pySES
  backend runs (``full_echam_hybrid``), because ``read_oxidant_vmr`` maps
  levels one-to-one onto the model grid. Above the CAM ~3.5 hPa top the
  profile is clamped to the top value — acceptable for JAM's tropospheric /
  lower-stratospheric sulfur chemistry, which is where the oxidant fields
  are consumed.

Usage:
    python tools/prep_jam_aux_inputs.py [--year 2014] \
        [--outdir /glade/derecho/scratch/$USER/jam_inputs]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Prefer this script's repo over any installed jcm: the pyses backend lives on
# this branch/worktree and may be absent from the editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_INPUTDATA = "/glade/campaign/cesm/cesmdata/inputdata"
_DMS_SRC = f"{_INPUTDATA}/atm/cam/chem/ocnexch/Csw_DMS_Lana2011_f09f09_1750_2100_20200717a.nc"
_DUST_SRC = f"{_INPUTDATA}/atm/cam/dst/dst_1.9x2.5_c090203.nc"
_OXID_SRC = (f"{_INPUTDATA}/atm/cam/chem/trop_mozart_aero/oxid/"
             "oxid_1.9x2.5_L26_1850-2015_c20181106.nc")

# 12 monthly mid-month timestamps: gives the readers a clean one-year span so
# ``align_mode='auto'`` resolves to WRAP_YEAR (climatology) indexing.
_CLIMO_TIME = np.array([np.datetime64(f"2014-{m:02d}-15") for m in range(1, 13)])


def prep_dms(out: Path) -> None:
    """Slice one 12-month block of the (year-constant) Lana climatology."""
    ds = xr.open_dataset(_DMS_SRC)
    da = ds["DMS_monthly_nM_out"].isel(time=slice(0, 12)).load()
    out_ds = xr.Dataset(
        {"DMS_sea": (("time", "lat", "lon"), da.values,
                     {"units": "nanomol l-1",
                      "long_name": "Lana et al. (2011) seawater DMS concentration"})},
        coords={"time": _CLIMO_TIME,
                "lat": ds["lat"].values, "lon": ds["lon"].values},
        attrs={"source": _DMS_SRC,
               "history": "prep_jam_aux_inputs.py: 12-month slice, renamed to "
                          "the jcm DMS_sea contract (read_dms_seawater)"},
    )
    out_ds.to_netcdf(out)
    print(f"wrote {out} {dict(out_ds.sizes)} max={np.nanmax(da.values):.2f} nM")


def prep_dust(out: Path) -> None:
    """Write the static erodibility map: mbl_bsn_fct_geo -> pot_source (lat, lon)."""
    ds = xr.open_dataset(_DUST_SRC)
    arr = np.clip(np.nan_to_num(ds["mbl_bsn_fct_geo"].values), 0.0, 1.0)
    out_ds = xr.Dataset(
        {"pot_source": (("lat", "lon"), arr,
                        {"units": "1",
                         "long_name": "CAM geomorphic dust source (mbl_bsn_fct_geo)"})},
        coords={"lat": ds["lat"].values, "lon": ds["lon"].values},
        attrs={"source": _DUST_SRC,
               "history": "prep_jam_aux_inputs.py: renamed to the jcm "
                          "pot_source contract (read_dust_source, static path)"},
    )
    out_ds.to_netcdf(out)
    print(f"wrote {out} {dict(out_ds.sizes)} max={arr.max():.2f}")


def prep_oxidants(out: Path, year: int, nlev: int = 47) -> None:
    """Vertically remap the CAM L26 oxidant climatology onto ECHAM L47."""
    from jcm.dycore.pyses.coords import full_echam_hybrid

    ds = xr.open_dataset(_OXID_SRC, decode_times=False)
    dates = ds["date"].values  # YYYYMMDD ints
    years = np.unique(dates // 10000)
    src_year = int(years[np.argmin(np.abs(years - year))])
    sel = np.nonzero(dates // 10000 == src_year)[0]
    assert sel.size == 12, f"expected a 12-month block for {src_year}, got {sel.size}"
    print(f"oxidants: using {src_year} block (nearest to {year}) from {years.min()}–{years.max()}")

    p0 = float(ds["P0"])
    hyam = ds["hyam"].values          # normalized (× P0 -> Pa), top→bottom
    hybm = ds["hybm"].values
    ps = ds["PS"].isel(time=sel).values                 # (12, lat, lon)
    # Source mid-level pressures per (month, lat, lon): (12, 26, lat, lon)
    p_src = (hyam[None, :, None, None] * p0
             + hybm[None, :, None, None] * ps[:, None, :, :])

    # Target ECHAM L47 mid-level pressures with the *same* surface pressure
    # field, so the mapping is a pure vertical-coordinate change. The model's
    # instantaneous ps will differ — read_oxidant_vmr's documented contract is
    # level-for-level anyway, so climatological ps is the consistent choice.
    a_b, b_b = full_echam_hybrid(nlev)                  # boundaries, a in Pa
    a_mid = 0.5 * (np.asarray(a_b)[:-1] + np.asarray(a_b)[1:])
    b_mid = 0.5 * (np.asarray(b_b)[:-1] + np.asarray(b_b)[1:])
    p_tgt = (a_mid[None, :, None, None]
             + b_mid[None, :, None, None] * ps[:, None, :, :])

    nt, _, nlat, nlon = p_src.shape
    logs = np.log(p_src).reshape(nt, -1, nlat * nlon)   # (12, 26, N)
    logt = np.log(p_tgt).reshape(nt, nlev, nlat * nlon)

    out_vars = {}
    for src_name, dst_name in [("OH", "OH_VMR_avrg"), ("NO3", "NO3_VMR_avrg"),
                               ("O3", "O3_VMR_avrg"), ("H2O2", "H2O2_VMR_avrg")]:
        v = np.nan_to_num(ds[src_name].isel(time=sel).values)  # (12, 26, lat, lon)
        vflat = v.reshape(nt, -1, nlat * nlon)
        res = np.empty((nt, nlev, nlat * nlon))
        for t in range(nt):
            for c in range(nlat * nlon):
                # np.interp clamps outside the source range: above the CAM
                # top the L47 mesospheric levels hold the top value.
                res[t, :, c] = np.interp(logt[t, :, c], logs[t, :, c], vflat[t, :, c])
        out_vars[dst_name] = (("time", "mlev", "lat", "lon"),
                              res.reshape(nt, nlev, nlat, nlon),
                              {"units": "mole/mole"})

    out_ds = xr.Dataset(
        out_vars,
        coords={"time": _CLIMO_TIME, "mlev": np.arange(1, nlev + 1),
                "lat": ds["lat"].values, "lon": ds["lon"].values},
        attrs={"source": _OXID_SRC, "source_year": src_year,
               "history": "prep_jam_aux_inputs.py: CAM L26 -> ECHAM L47 "
                          "log-p vertical remap (clamped above CAM top); "
                          "MACC-layout for jcm read_oxidant_vmr"},
    )
    # ECHAM convention: hyam in Pa (p = hyam + hybm*ps), top→bottom.
    out_ds["hyam"] = ("mlev", a_mid, {"units": "Pa"})
    out_ds["hybm"] = ("mlev", b_mid, {"units": "1"})
    out_ds["p0"] = ((), 101325.0, {"units": "Pa"})
    out_ds.to_netcdf(out)
    print(f"wrote {out} {dict(out_ds.sizes)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2014)
    ap.add_argument("--outdir",
                    default=f"/glade/derecho/scratch/{os.environ['USER']}/jam_inputs")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prep_dms(outdir / "dms_lana2011_climo.nc")
    prep_dust(outdir / "dust_erodibility_cam_f19.nc")
    prep_oxidants(outdir / f"oxidants_cam_echam_l47_{args.year}.nc", args.year)


if __name__ == "__main__":
    main()
