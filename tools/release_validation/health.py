"""Climatological health check for a finished jcm run directory.

Usage:
    python check_run_health.py <run_dir with *_dayNNN.nc chunks> [--log FILE]

Computes annual, area-weighted global means from the saved chunks and
checks loose climatological ranges (spin-up tolerant — this is a
"did the model produce a climate" gate, not a tuning target):

    TOA net  = radiation.toa_sw_down - toa_sw_up - toa_lw_up   |net| <= 10 W/m2
    precip   = clouds.precip_rain + precip_snow + convection.precip_conv
               (kg/m2/s -> mm/day)                              2 - 4 mm/day
    cloud    = column max of clouds.cloud_fraction              0.4 - 0.8
    near-sfc T = temperature at the lowest level                278 - 295 K

Also scans every saved variable for NaN/Inf and, with --log, reports the
settled sim-days/hr (last chunk wall) for runtime-regression tracking.
Exit code 0 = all checks pass.
"""
import argparse
import glob
import re
import sys

import numpy as np
import xarray as xr

RANGES = {
    "toa_net_wm2": (-10.0, 10.0),
    "precip_mm_day": (2.0, 4.0),
    "cloud_cover": (0.4, 0.8),
    "near_surface_T": (278.0, 295.0),
    # Global-mean 550 nm AOD: JAM's jam_band_optics.aod_550 or MACv2-SP's
    # aerosol.aod_total. Wide gate — a from-zero JAM spin-up year sits low,
    # so use --last-n to score the settled months.
    "aod_550": (0.02, 0.35),
}

# Global-mean column burdens [mg/m2] per species, JAM runs only (summed
# over the interstitial modes; cloud-borne lives in the carry, not the
# saved tracers). Very loose gates around AeroCom-model ranges — the
# check is "does each species hold a plausible planetary loading", not
# a tuning target. From-zero spin-up: score with --last-n.
BURDEN_RANGES = {
    "so4": (0.5, 8.0),
    "bc": (0.05, 1.0),
    "poa": (0.2, 4.0),
    "soa": (0.2, 6.0),
    "ss": (2.0, 30.0),
    "du": (3.0, 60.0),
}


def wmean(da, lat):
    w = np.cos(np.deg2rad(lat))
    w = xr.DataArray(w, dims=["lat"], coords={"lat": lat})
    return float(da.weighted(w).mean(
        [d for d in da.dims if d != "time"]).mean("time")
        if "time" in da.dims else da.weighted(w).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--log")
    ap.add_argument("--last-n", type=int, default=None,
                    help="use only the last N chunks (default: all)")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.run_dir}/*_day*.nc"),
                   key=lambda f: int(re.search(r"day(\d+)", f).group(1)))
    if not files:
        print(f"FAIL  no chunk files in {a.run_dir}")
        return 1
    if a.last_n:
        files = files[-a.last_n:]
    ds = xr.open_mfdataset(files, combine="by_coords")
    lat = ds.lat.values

    ok = True

    def check(name, value, lo, hi):
        nonlocal ok
        good = lo <= value <= hi
        print(f"{'PASS' if good else 'FAIL'}  {name} = {value:.2f} "
              f"(expected [{lo}, {hi}])")
        ok = ok and good

    # NaN scan over everything saved.
    bad = []
    for v in ds.data_vars:
        arr = ds[v].isel(time=-1).values
        if not np.all(np.isfinite(arr)):
            bad.append(v)
    print(f"{'PASS' if not bad else 'FAIL'}  NaN scan: "
          f"{len(bad)}/{len(ds.data_vars)} variables non-finite "
          f"{bad[:5] if bad else ''}")
    ok = ok and not bad

    speedy = "longwave_rad.ftop" in ds       # SPEEDY field dialect
    if speedy:
        # shortwave_rad.ftop = net downward SW at TOA; longwave_rad.ftop
        # = OUTGOING LW at TOA (speedy_longwave.py docstring; verified on
        # a full model year: global mean +219 W/m², everywhere positive —
        # the units_table's "net downward" row for it is a copy-paste of
        # the shortwave description). Net TOA = SW_net_down − OLR.
        toa = ds["shortwave_rad.ftop"] - ds["longwave_rad.ftop"]
    else:
        toa = (ds["radiation.toa_sw_down"] - ds["radiation.toa_sw_up"]
               - ds["radiation.toa_lw_up"])
    check("toa_net_wm2", wmean(toa, lat), *RANGES["toa_net_wm2"])

    if speedy:
        # SPEEDY precls/precnv are g/m²/s → ×86.4 for mm/day.
        precip = (ds.get("condensation.precls", 0)
                  + ds.get("convection.precnv", 0)) * 86.4
    else:
        precip = (ds.get("clouds.precip_rain", 0)
                  + ds.get("clouds.precip_snow", 0)
                  + ds.get("convection.precip_conv", 0)) * 86400.0
    check("precip_mm_day", wmean(precip, lat), *RANGES["precip_mm_day"])

    if speedy:
        cf = ds["shortwave_rad.cloudc"]
    else:
        cf = ds["clouds.cloud_fraction"].max("level")
    check("cloud_cover", wmean(cf, lat), *RANGES["cloud_cover"])

    # Lowest model level (level index orientation: take the max-pressure end;
    # jcm output has level index 0 = lowest layer).
    t_low = ds["temperature"].isel(level=0)
    check("near_surface_T", wmean(t_low, lat), *RANGES["near_surface_T"])

    # 550 nm AOD — JAM publishes jam_band_optics.aod_550, MACv2-SP runs
    # publish aerosol.aod_total; whichever is present is the scheme's AOD.
    aod = None
    for key in ("jam_band_optics.aod_550", "aerosol.aod_total"):
        if key in ds:
            aod = ds[key]
            break
    if aod is not None:
        check("aod_550", wmean(aod, lat), *RANGES["aod_550"])
    else:
        print("NOTE  no AOD field found (aod_550/aod_total); skipping")

    # Per-species global burdens (JAM only): column-integrate the
    # interstitial mass tracers m_<species>_<mode> with the saved
    # air_density × layer_thickness air mass.
    mass_keys = [v for v in ds.data_vars if re.fullmatch(r"m_\w+_\w+", v)]
    if mass_keys:
        if "air_density" in ds and "layer_thickness" in ds:
            dm_air = ds["air_density"] * ds["layer_thickness"]
        else:
            dm_air = None
            print("NOTE  air_density/layer_thickness not saved; "
                  "skipping burdens")
        if dm_air is not None:
            by_species = {}
            for v in mass_keys:
                sp = v.split("_")[1]
                by_species.setdefault(sp, []).append(ds[v])
            for sp, (lo, hi) in BURDEN_RANGES.items():
                if sp not in by_species:
                    print(f"NOTE  no {sp} mass tracers; skipping burden")
                    continue
                total = sum(by_species[sp])
                burden = (total.clip(min=0) * dm_air).sum("level") * 1e6
                check(f"burden_{sp}_mg_m2", wmean(burden, lat), lo, hi)

    if a.log:
        walls = re.findall(r"Wall: ([0-9.]+)s this chunk", open(a.log).read())
        if len(walls) >= 2:
            w = float(walls[-1])
            chunk_days = 5.0
            print(f"INFO  settled rate ~ {chunk_days * 3600 / w:.0f} "
                  f"sim-days/hr (last chunk {w:.0f}s; compare vs the "
                  "derecho-jcm-runs reference points)")

    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
