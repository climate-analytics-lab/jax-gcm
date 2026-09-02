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
from pathlib import Path

import numpy as np
import xarray as xr

# Shared weighting/column-integration machinery lives in jcm.analysis (#640);
# the species table and the mode-summing burden() are tool domain and stay in
# tools/jam_burden_report.py (it includes cloud-borne tracers and the
# pressure_half level-orientation handling).
from jcm.analysis import area_weights, global_mean  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1]))
from jam_burden_report import _SPECIES, burden  # noqa: E402

#: Gate slack: the release gate is the climatological anchor range from
#: the shared species table widened by this factor each way — a "did the
#: model produce a plausible planetary loading" gate, not a tuning target.
_BURDEN_SLACK = 3.0

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

# Burden gates derive from the shared anchor table (see _BURDEN_SLACK).
BURDEN_RANGES = {
    sp: (lo / _BURDEN_SLACK, hi * _BURDEN_SLACK)
    for sp, (_modes, (lo, hi)) in _SPECIES.items()
}


def wmean(da, weights):
    """Time-mean, area-weighted global mean (over the horizontal dims)."""
    if "time" in da.dims:
        da = da.mean("time")
    return float(global_mean(da, weights))


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
    weights = area_weights(ds)

    ok = True

    def check(name, value, lo, hi):
        nonlocal ok
        good = lo <= value <= hi
        print(f"{'PASS' if good else 'FAIL'}  {name} = {value:.2f} "
              f"(expected [{lo:g}, {hi:g}])")
        ok = ok and good

    # NaN scan over everything saved, across the WHOLE opened window —
    # a run that NaN'd mid-year and was restarted can end on a finite
    # chunk, so the last time step alone is not evidence of health.
    bad = []
    for v in ds.data_vars:
        if not bool(np.isfinite(ds[v].values).all()):
            bad.append(v)
    print(f"{'PASS' if not bad else 'FAIL'}  NaN scan: "
          f"{len(bad)}/{len(ds.data_vars)} variables non-finite "
          f"{bad[:5] if bad else ''}")
    ok = ok and not bad

    speedy = "longwave_rad.ftop" in ds       # SPEEDY field dialect
    if speedy:
        # shortwave_rad.ftop is the net downward SW at TOA and
        # longwave_rad.ftop the OUTGOING LW (see speedy_longwave.py), so
        # net TOA = SW_net_down − OLR.
        toa = ds["shortwave_rad.ftop"] - ds["longwave_rad.ftop"]
    else:
        toa = (ds["radiation.toa_sw_down"] - ds["radiation.toa_sw_up"]
               - ds["radiation.toa_lw_up"])
    check("toa_net_wm2", wmean(toa, weights), *RANGES["toa_net_wm2"])

    if speedy:
        # SPEEDY precls/precnv are g/m²/s → ×86.4 for mm/day.
        precip = (ds.get("condensation.precls", 0)
                  + ds.get("convection.precnv", 0)) * 86.4
    else:
        precip = (ds.get("clouds.precip_rain", 0)
                  + ds.get("clouds.precip_snow", 0)
                  + ds.get("convection.precip_conv", 0)) * 86400.0
    check("precip_mm_day", wmean(precip, weights), *RANGES["precip_mm_day"])

    if speedy:
        cf = ds["shortwave_rad.cloudc"]
    else:
        cf = ds["clouds.cloud_fraction"].max("level")
    check("cloud_cover", wmean(cf, weights), *RANGES["cloud_cover"])

    # Lowest model level (level index orientation: take the max-pressure end;
    # jcm output has level index 0 = lowest layer).
    t_low = ds["temperature"].isel(level=0)
    check("near_surface_T", wmean(t_low, weights), *RANGES["near_surface_T"])

    # 550 nm AOD — JAM publishes jam_band_optics.aod_550, MACv2-SP runs
    # publish aerosol.aod_total; whichever is present is the scheme's AOD.
    aod = None
    for key in ("jam_band_optics.aod_550", "aerosol.aod_total"):
        if key in ds:
            aod = ds[key]
            break
    if aod is not None:
        check("aod_550", wmean(aod, weights), *RANGES["aod_550"])
    else:
        print("NOTE  no AOD field found (aod_550/aod_total); skipping")

    # Per-species global burdens (JAM runs): the shared ``burden`` sums
    # interstitial + cloud-borne mass over the species' modes and
    # integrates q·Δp/g from the file's own pressure_half.
    if any(re.fullmatch(r"m_\w+_\w+", v) for v in ds.data_vars):
        for sp, (lo, hi) in BURDEN_RANGES.items():
            col = burden(ds, sp, _SPECIES[sp][0])
            if col is None:
                print(f"NOTE  no {sp} mass tracers; skipping burden")
                continue
            check(f"burden_{sp}_mg_m2",
                  float(global_mean(col.compute(), weights)), lo, hi)

    if a.log:
        walls = re.findall(r"Wall: ([0-9.]+)s this chunk", open(a.log).read())
        # Chunk length from the day-numbered filenames themselves, not a
        # constant that can drift from the run's actual chunk_days.
        day_nums = sorted(int(re.search(r"day(\d+)", f).group(1))
                          for f in files)
        spacings = {b - a_ for a_, b in zip(day_nums, day_nums[1:])}
        if len(walls) >= 2 and len(spacings) == 1:
            w = float(walls[-1])
            chunk_days = spacings.pop()
            print(f"INFO  settled rate ~ {chunk_days * 3600 / w:.0f} "
                  f"sim-days/hr (last chunk {w:.0f}s, {chunk_days}-day "
                  "chunks; compare vs the recorded baselines)")

    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
