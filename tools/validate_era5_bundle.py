"""Validate the transient ERA5 bundles against independent references.

Glade-only companion to ``build_mirror --stage era5-transient`` (#629);
run it on the Tier A intermediates and emitted bundles before ``--stage
upload``:

    python tools/validate_era5_bundle.py --years 1982,2003,2010,2022,2024

Checks, in order:

1. **AIMIP regression** — the Tier A month-start SST/ice against the
   published AIMIP Phase-1 forcing (Zenodo doi:10.5281/zenodo.16782372,
   ``--aimip`` path). Both are built from ERA5 with the same
   centred-window construction on the same 0.25° grid, so ocean-mean
   |diff| should be ~0.01 K and the max well under 0.1 K. A big residual
   means the window arithmetic drifted from the published recipe.
2. **Land realism** — land-mean ``stl`` must differ year to year (the
   whole point of #629: the AMIP bundle repeats a climatology), and the
   2003 European / 2010 Russian heatwaves must appear as warm monthly
   anomalies relative to the other sample years.
3. **Bundle sanity** — every emitted ``forcing_era5/<year>.nc`` passes
   ``jcm.forcing._validate_bc_fields`` with no warnings, fractions stay
   in [0, 1], and no field carries NaN.

Exits non-zero on any failure so it can gate a build pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.analysis import area_weights

ROOT = Path(os.environ.get(
    "JCM_MIRROR_ROOT",
    f"/glade/derecho/scratch/{os.environ.get('USER', '')}/hf_mirror"))
AIMIP_DEFAULT = (ROOT / "sources" / "aimip" /
                 "ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc")

FAILURES: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("PASS " if ok else "FAIL "), message, flush=True)
    if not ok:
        FAILURES.append(message)


def _pick_var(ds: xr.Dataset, *needles: str) -> xr.DataArray:
    """First data variable whose name contains any needle (case-insensitive)."""
    for needle in needles:
        for name in ds.data_vars:
            if needle in name.lower() and ds[name].ndim >= 3:
                return ds[name]
    raise KeyError(f"no variable matching {needles} in {list(ds.data_vars)}")


def check_aimip_regression(aimip_path: Path, years: list[int]) -> None:
    ref = xr.open_dataset(aimip_path)
    sst_ref = _pick_var(ref, "sea_surface_temperature", "sst", "tos")
    ice_ref = _pick_var(ref, "sea_ice_cover", "siconc", "sic")
    for year in years:
        ours = xr.open_dataset(ROOT / "build" / "era5_sstice" / f"{year}.nc")
        span = slice(f"{year}-01-01", f"{year}-12-31")
        # Measured on 1982: max|d| = 2e-4 K — the published file uses this
        # exact construction, so the tolerances are float32-rounding slack,
        # not physics slack.
        for name, ref_da, our_da, tol_mean, tol_max in (
                ("sst", sst_ref, ours.sstk, 0.005, 0.05),
                ("ice", ice_ref, ours.ci, 0.001, 0.01)):
            r = ref_da.sel(time=span)
            if r.sizes.get("time") != 12:
                check(False, f"AIMIP file has {r.sizes.get('time')} steps "
                             f"for {year}")
                continue
            # Compare on the common ocean mask; align latitudes by value
            # (the two products may store opposite orientations).
            rv = r.values
            ov = our_da.values
            ref_lat = r[r.dims[-2]].values
            if not np.allclose(ref_lat, ours.latitude.values):
                if np.allclose(ref_lat[::-1], ours.latitude.values):
                    rv = rv[:, ::-1, :]
                else:
                    check(False, f"{name} {year}: AIMIP grid does not "
                                 f"match 0.25° source grid")
                    continue
            both = np.isfinite(rv) & np.isfinite(ov)
            diff = np.abs(rv - ov)[both]
            check(diff.mean() < tol_mean and diff.max() < tol_max,
                  f"AIMIP {name} {year}: mean|d|={diff.mean():.4f} "
                  f"max|d|={diff.max():.3f} over {both.sum()} pts "
                  f"(tol {tol_mean}/{tol_max})")


def _land_monthly_stl(year: int) -> xr.DataArray:
    """Area-weighted land-mean stl1 per month from the Tier A land file."""
    land = xr.open_dataset(
        ROOT / "build" / "era5_land_transient" / f"{year}.nc")
    clim = xr.open_dataset(ROOT / "build" /
                           "era5_land_climo_2005-2014_0p25.nc")
    # Shared area weights (#640). ERA5's regular 0.25° grid is not
    # Gauss-Legendre, so area_weights returns cos(lat) — value-identical to
    # the previous inline weighting — carried onto ERA5's ``latitude`` dim and
    # masked to land.
    lat_w = xr.DataArray(np.asarray(area_weights(land.latitude.values)),
                         dims="latitude", coords={"latitude": land.latitude})
    w = lat_w * (clim.lsm > 0.5)
    return (land.stl1 * w).sum(("latitude", "longitude")) / w.sum()


def check_land_realism(years: list[int]) -> None:
    means = {y: float(_land_monthly_stl(y).isel(time=slice(1, None))
                      .mean("time")) for y in years}
    print("   land-mean stl by year:",
          {y: round(v, 3) for y, v in means.items()}, flush=True)
    spread = max(means.values()) - min(means.values())
    check(spread > 0.1,
          f"land stl varies between years (spread {spread:.3f} K — the "
          f"climatology-tiled AMIP bundle would give 0)")
    if all(y in years for y in (1982, 2022)):
        check(means[2022] > means[1982],
              f"land stl warmed 1982→2022 "
              f"({means[1982]:.2f} → {means[2022]:.2f} K)")

    def region_month(year, month, lat0, lat1, lon0, lon1):
        land = xr.open_dataset(
            ROOT / "build" / "era5_land_transient" / f"{year}.nc")
        da = land.stl1.sel(time=f"{year}-{month:02d}-01",
                           latitude=slice(lat1, lat0),   # ERA5 lat descends
                           longitude=slice(lon0, lon1))
        return float(da.mean())

    others = [y for y in years if y not in (2003,)]
    if 2003 in years and others:
        base = np.mean([region_month(y, 8, 44, 50, 2, 8) for y in others])
        anom = region_month(2003, 8, 44, 50, 2, 8) - base
        check(anom > 2.0, f"2003 European heatwave visible in stl "
                          f"(Aug France anomaly {anom:+.2f} K)")
    others = [y for y in years if y not in (2010,)]
    if 2010 in years and others:
        base = np.mean([region_month(y, 7, 50, 60, 35, 55) for y in others])
        anom = region_month(2010, 7, 50, 60, 35, 55) - base
        check(anom > 2.0, f"2010 Russian heatwave visible in stl "
                          f"(Jul anomaly {anom:+.2f} K)")


def check_bundles(years: list[int], grids: tuple[str, ...]) -> None:
    from jcm.forcing import _validate_bc_fields

    for grid in grids:
        for year in years:
            path = ROOT / "upload" / "bundles" / grid / "forcing_era5" / \
                f"{year}.nc"
            ds = xr.open_dataset(path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    _validate_bc_fields(ds)
                    err = None
                except ValueError as e:
                    err = e
            check(err is None and not caught,
                  f"{grid}/{year}: _validate_bc_fields clean "
                  f"({err or [str(w.message)[:60] for w in caught] or 'ok'})")
            finite = all(np.isfinite(ds[v].values).all()
                         for v in ("sst", "icec", "stl", "soilw_am",
                                   "snowc", "alb", "co2", "ch4", "n2o"))
            frac = all(float(ds[v].min()) >= 0.0
                       and float(ds[v].max()) <= 1.0
                       for v in ("icec", "soilw_am", "snowc", "alb"))
            check(finite and frac,
                  f"{grid}/{year}: finite everywhere, fractions in [0,1]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="1982,2003,2010,2022,2024")
    ap.add_argument("--aimip", default=str(AIMIP_DEFAULT))
    ap.add_argument("--grids", default="t63,t106")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",")]

    if Path(args.aimip).exists():
        check_aimip_regression(Path(args.aimip), years)
    else:
        print(f"SKIP AIMIP regression ({args.aimip} not present)")
    check_land_realism(years)
    check_bundles(years, tuple(args.grids.split(",")))

    if FAILURES:
        sys.exit(f"{len(FAILURES)} validation failure(s)")
    print("all validations passed", flush=True)


if __name__ == "__main__":
    main()
